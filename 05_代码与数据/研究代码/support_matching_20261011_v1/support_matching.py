"""Bounded reference-feasibility cost for native cross-stage matching.

No detector forward, prediction grid, parameter, or loss weight is changed.
The native matching code is source-bound and receives exactly one extra cost.
"""
import hashlib
import inspect
import textwrap
import types

import torch


def support_penalty(reference, targets, up, reg_scale):
    """Return [B*Q, sum(G)] bounded overflow, with no GT-size threshold.

    Native d_left = (cx - GT_x1)/(w/s) - s/2. The prediction support is
    [-2*abs(up)*s, +2*abs(up)*s]. Penalize the largest violating edge by
    min(1, overflow / support_halfwidth). This is a matching preference,
    not a probability, localization loss, or guaranteed attainable IoU.
    """
    with torch.no_grad(), torch.autocast(device_type=reference.device.type, enabled=False):
        r = reference.detach().float().flatten(0, 1)
        g = torch.cat([t['boxes'].detach().float() for t in targets], 0)
        assert r.ndim == g.ndim == 2 and r.shape[1] == g.shape[1] == 4
        assert torch.isfinite(r).all() and torch.isfinite(g).all()
        assert (r[:, 2:] > 0).all() and (g[:, 2:] > 0).all()
        s = reg_scale.detach().float().abs().reshape(())
        bound = 2 * up.detach().float().abs().reshape(()) * s
        assert s > 0 and bound > 0
        xy1, xy2 = g[:, :2] - g[:, 2:] / 2, g[:, :2] + g[:, 2:] / 2
        delta_left = (r[:, None, :2] - xy1[None]) / (r[:, None, 2:] / s + 1e-16) - s / 2
        delta_right = (xy2[None] - r[:, None, :2]) / (r[:, None, 2:] / s + 1e-16) - s / 2
        distance = torch.cat((delta_left, delta_right), -1)
        penalty = ((distance.abs().amax(-1) - bound).clamp(min=0) / bound).clamp(max=1)
        assert torch.isfinite(penalty).all()
        return penalty


def compile_matcher_forward(native_class, expected_file_sha):
    source_file = inspect.getsourcefile(native_class)
    assert hashlib.sha256(open(source_file, 'rb').read()).hexdigest() == expected_file_sha
    source = textwrap.dedent(inspect.getsource(native_class.forward))
    anchor = 'C = C.view(bs, num_queries, -1).cpu()'
    assert source.count(anchor) == 1
    replacement = ('if getattr(self, "_support_matching_penalty", None) is not None:\n'
                   '            assert C.shape == self._support_matching_penalty.shape\n'
                   '            C = C + self.cost_giou * self._support_matching_penalty\n'
                   '        ' + anchor)
    source = source.replace(anchor, replacement)
    namespace = dict(native_class.forward.__wrapped__.__globals__)
    exec(compile(source, '<source-bound-support-matcher>', 'exec'), namespace)
    return namespace['forward'], hashlib.sha256(source.encode()).hexdigest()


class SupportMatchingBinding:
    """Install on one criterion instance; all five ordinary match calls share it.

    Original GO frequency reconciliation and all DN supervision stay native.
    A bounded penalty does not force a feasible assignment or enlarge the pool.
    """
    def __init__(self, criterion, expected_matcher_sha):
        self.criterion = criterion
        self.matcher = criterion.matcher
        self.original_criterion_forward = criterion.forward
        self.original_matcher_forward = self.matcher.forward
        self.modified_forward, self.compiled_source_sha = compile_matcher_forward(
            type(self.matcher), expected_matcher_sha)
        assert self.matcher.use_focal_loss and not self.matcher.change_matcher
        assert self.matcher.cost_giou == 2 and not self.matcher.mask_point_sample_ratio
        self.calls = 0

    def install(self):
        assert not hasattr(self.matcher, '_support_matching_penalty')
        self.matcher._support_matching_penalty = None
        self.matcher.forward = types.MethodType(self.modified_forward, self.matcher)
        binding = self

        def criterion_forward(module, outputs, targets, **kwargs):
            assert binding.matcher._support_matching_penalty is None
            if 'pre_outputs' not in outputs:
                return binding.original_criterion_forward(outputs, targets, **kwargs)
            assert module.use_uni_set and module.losses == ['mal', 'boxes', 'local']
            assert 'mgsa_outputs' not in outputs and 'o2m_t_outputs' not in outputs
            refs = outputs['pre_outputs']['pred_boxes']
            assert refs.shape[1] == 300 and len(outputs['aux_outputs']) == 2
            assert len(outputs['enc_aux_outputs']) == 1
            binding.matcher._support_matching_penalty = support_penalty(
                refs, targets, outputs['up'], outputs['reg_scale'])
            binding.calls += 1
            try:
                return binding.original_criterion_forward(outputs, targets, **kwargs)
            finally:
                binding.matcher._support_matching_penalty = None

        self.criterion.forward = types.MethodType(criterion_forward, self.criterion)
        return self

    def close(self):
        self.criterion.forward = self.original_criterion_forward
        self.matcher.forward = self.original_matcher_forward
        del self.matcher._support_matching_penalty

    def metadata(self):
        return dict(compiled_native_matcher_sha256=self.compiled_source_sha,
            coefficient='native cost_giou=2', penalty='min(1,max_edge_overflow/support_halfwidth)',
            ordinary_match_calls_per_batch=5, native_GO_rule_preserved=True,
            added_parameters=0, native_FDR_grid_unchanged=True, DN_unchanged=True)
