"""CPU checks for the P2 phase controls (hgnetv2_light_phase.py).

Loads the backbone modules directly so pycocoeval etc. are not required.
Run: python tests/engine/test_p2_phase_controls.py
"""
import importlib.util
import os
import sys
import types
import unittest

import torch
from torch.nn import functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BACKBONE = os.path.join(ROOT, 'engine', 'backbone')


def _load_isolated():
    """Import light/phase modules with a stub `register` and stub HGNetv2 deps."""
    pkg = types.ModuleType('engine'); pkg.__path__ = [os.path.join(ROOT, 'engine')]
    core = types.ModuleType('engine.core'); core.register = lambda *a, **k: (lambda c: c)
    bb = types.ModuleType('engine.backbone'); bb.__path__ = [BACKBONE]
    sys.modules.update({'engine': pkg, 'engine.core': core, 'engine.backbone': bb})

    class HGNetv2(torch.nn.Module):  # minimal stand-in: B0 channel layout only
        arch_configs = {'B0': {'stem_channels': [3, 16, 16]}}
        def __init__(self, **kw):
            super().__init__()
            self._out_channels = [64, 256, 512, 1024]
    hg = types.ModuleType('engine.backbone.hgnetv2'); hg.HGNetv2 = HGNetv2
    sys.modules['engine.backbone.hgnetv2'] = hg

    mods = {}
    for name in ('hgnetv2_light', 'hgnetv2_light_control', 'hgnetv2_light_phase'):
        spec = importlib.util.spec_from_file_location(
            f'engine.backbone.{name}', os.path.join(BACKBONE, f'{name}.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


M = _load_isolated()
C = M['hgnetv2_light'].P2ToP3Residual
LowPass = M['hgnetv2_light_control'].P2ToP3LowPassResidual
Ablate = M['hgnetv2_light_phase'].P2ToP3PhaseAblation
Single = M['hgnetv2_light_phase'].P2ToP3SinglePhaseResidual
Backbone = M['hgnetv2_light_phase'].HGNetv2_Light_Phase
CMean = M['hgnetv2_light_phase'].P2ToP3PhaseMeanResidual
LowRank = M['hgnetv2_light_phase'].P2ToP3LowRankResidual
# Archived rank16 branch actually used in the 12-epoch runs (reference for equivalence).
LOWRANK_REF = os.environ.get(
    'LOWRANK_REF',
    os.path.join(ROOT, 'tests', 'engine', 'lowrank_branch_ref_20261011_v3.py'))


def _trained_c(seed=0):
    torch.manual_seed(seed)
    c = C()
    torch.nn.init.normal_(c.restore.weight)  # emulate trained (non-zero) restore
    return c.eval()


def _inputs(h=16, w=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(2, 64, h, w, generator=g),
            torch.randn(2, 256, (h + 1) // 2, (w + 1) // 2, generator=g))


def _num(m):
    return sum(p.numel() for p in m.parameters())


class PhaseControls(unittest.TestCase):
    def test_ablation_none_equals_c_with_same_state_dict(self):
        c = _trained_c()
        a = Ablate(phase_eval='none').eval()
        a.load_state_dict(c.state_dict(), strict=True)
        p2, p3 = _inputs()
        with torch.no_grad():
            self.assertTrue(torch.equal(a(p2, p3), c(p2, p3)))

    def test_state_dict_keys_match_c(self):
        self.assertEqual(set(Ablate().state_dict()), set(C().state_dict()))

    def test_mean_equals_avgpool_lowpass(self):
        c = _trained_c()
        a = Ablate(phase_eval='mean').eval(); a.load_state_dict(c.state_dict())
        p2, p3 = _inputs()
        with torch.no_grad():
            pooled = F.avg_pool2d(c.reduce(p2), 2, 2)          # [B,16,H/2,W/2]
            expect = p3 + c.restore(pooled.repeat_interleave(4, dim=1))
            self.assertLess((a(p2, p3) - expect).abs().max().item(), 1e-5)

    def test_shuffle_moves_every_phase(self):
        c = _trained_c()
        a = Ablate(phase_eval='shuffle').eval(); a.load_state_dict(c.state_dict())
        p2, p3 = _inputs()
        with torch.no_grad():
            d = F.pixel_unshuffle(c.reduce(p2), 2).view(2, 16, 4, 8, 8)
            perm = d[:, :, [1, 2, 3, 0]].reshape(2, 64, 8, 8)
            self.assertLess((a(p2, p3) - (p3 + c.restore(perm))).abs().max().item(), 1e-5)
            self.assertGreater((a(p2, p3) - c(p2, p3)).abs().max().item(), 1e-3)

    def test_zero_returns_p3(self):
        a = Ablate(phase_eval='zero').eval()
        p2, p3 = _inputs()
        self.assertTrue(torch.equal(a(p2, p3), p3))

    def test_ablation_rejected_in_training(self):
        a = Ablate(phase_eval='mean').train()
        with self.assertRaises(RuntimeError):
            a(*_inputs())

    def test_single_phase_takes_top_left(self):
        torch.manual_seed(0)
        s = Single().eval(); torch.nn.init.normal_(s.restore.weight)
        p2, p3 = _inputs()
        with torch.no_grad():
            expect = p3 + s.restore(s.expand(s.reduce(p2)[:, :, 0::2, 0::2]))
            self.assertTrue(torch.allclose(s(p2, p3), expect))
            p2b = p2.clone(); p2b[:, :, 1::2, :] += 5; p2b[:, :, :, 1::2] += 5
            self.assertTrue(torch.allclose(s(p2b, p3), expect))  # other phases ignored

    def test_zero_init_identity_and_odd_sizes(self):
        for mod in (Single(), C()):
            for h, w in ((16, 16), (15, 17)):
                p2, p3 = _inputs(h, w)
                self.assertTrue(torch.equal(mod.eval()(p2, p3), p3))

    def test_single_phase_capacity_matches_lowpass(self):
        self.assertEqual(_num(Single()), _num(LowPass()))
        print(f'\nparams C={_num(C())} lowpass={_num(LowPass())} single={_num(Single())}')

    def test_gradients_reach_bridge(self):
        s = Single().train()
        p2, p3 = _inputs()
        s(p2.requires_grad_(), p3).sum().backward()
        self.assertGreater(s.restore.weight.grad.abs().sum().item(), 0)

    # ---- C-mean / rank16 / rank16-mean (trainable controls) ----
    def test_c_mean_same_params_keys_and_init_as_c(self):
        torch.manual_seed(3); c = C()
        torch.manual_seed(3); cm = CMean()
        self.assertEqual(_num(cm), _num(c))
        self.assertEqual(set(cm.state_dict()), set(c.state_dict()))
        for k, v in c.state_dict().items():
            self.assertTrue(torch.equal(cm.state_dict()[k], v), k)
        self.assertEqual(torch.count_nonzero(cm.restore.weight).item(), 0)

    def test_c_mean_equals_inference_mean_ablation(self):
        c = _trained_c()
        cm = CMean().eval(); cm.load_state_dict(c.state_dict())
        a = Ablate(phase_eval='mean').eval(); a.load_state_dict(c.state_dict())
        p2, p3 = _inputs()
        with torch.no_grad():
            self.assertTrue(torch.allclose(cm(p2, p3), a(p2, p3)))

    def test_c_mean_phase_grouping_and_replication(self):
        cm = CMean().eval()
        x = torch.randn(2, 16, 8, 8)
        d = F.pixel_unshuffle(x, 2)                       # [2,64,4,4], channel c*4+phase
        m = M['hgnetv2_light_phase']._phase_mean(d, 16).view(2, 16, 4, 4, 4)
        ref = F.avg_pool2d(x, 2, 2)                        # per-channel 2x2 mean
        for ph in range(4):
            self.assertTrue(torch.allclose(m[:, :, ph], ref, atol=1e-6))

    def test_c_mean_ignores_within_cell_differences(self):
        torch.manual_seed(0)
        cm = CMean().eval(); torch.nn.init.normal_(cm.restore.weight)
        p2, p3 = _inputs()
        with torch.no_grad():
            r = cm.reduce(p2)
            # same reduced 2x2 means, different within-cell pattern -> same output
            pert = torch.zeros_like(r); pert[:, :, 0::2, 0::2] = 1; pert[:, :, 1::2, 1::2] = -1
            out1 = p3 + cm.restore(M['hgnetv2_light_phase']._phase_mean(F.pixel_unshuffle(r, 2), 16))
            out2 = p3 + cm.restore(M['hgnetv2_light_phase']._phase_mean(F.pixel_unshuffle(r + pert, 2), 16))
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))
            self.assertTrue(torch.allclose(cm(p2, p3), out1, atol=1e-5))

    def test_c_mean_trains_and_grad_flows(self):
        cm = CMean().train()
        p2, p3 = _inputs()
        cm(p2, p3).sum().backward()
        self.assertGreater(cm.restore.weight.grad.abs().sum().item(), 0)
        # restore columns of the 4 phases of a channel get identical gradients
        g = cm.restore.weight.grad.view(256, 16, 4)
        self.assertTrue(torch.allclose(g[:, :, 0], g[:, :, 3], atol=1e-6))

    def test_inference_guard_still_active(self):
        with self.assertRaises(RuntimeError):
            Ablate(phase_eval='mean').train()(*_inputs())

    def test_rank16_matches_archived_lowrank_branch(self):
        spec = importlib.util.spec_from_file_location('lowrank_ref', LOWRANK_REF)
        ref_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref_mod)
        torch.manual_seed(5); ref = ref_mod.LowRankP2Residual()
        torch.manual_seed(5); r = LowRank()
        self.assertEqual(set(r.state_dict()), set(ref.state_dict()))
        for k, v in ref.state_dict().items():
            self.assertTrue(torch.equal(r.state_dict()[k], v), k)
        self.assertEqual(_num(r), 6176)
        torch.nn.init.normal_(ref.restore.weight); r.load_state_dict(ref.state_dict())
        p2, p3 = _inputs()
        with torch.no_grad():
            self.assertTrue(torch.equal(r.eval()(p2, p3), ref.eval()(p2, p3)))

    def test_rank16_mean_same_params_init_and_rank(self):
        torch.manual_seed(5); r = LowRank()
        torch.manual_seed(5); rm = LowRank(phase_mean=True)
        self.assertEqual(_num(rm), _num(r))
        for k, v in r.state_dict().items():
            self.assertTrue(torch.equal(rm.state_dict()[k], v), k)
        p2, p3 = _inputs()
        # restore is zero-init: at step 0 only restore receives gradient (same as rank16/C)
        rm.train()(p2, p3).sum().backward()
        self.assertGreater(rm.restore.weight.grad.abs().sum().item(), 0)
        self.assertEqual(rm.mix.weight.grad.abs().sum().item(), 0)
        torch.nn.init.normal_(rm.restore.weight); rm.zero_grad()
        rm(p2, p3).sum().backward()
        self.assertGreater(rm.mix.weight.grad.abs().sum().item(), 0)
        # mix columns of the 4 phases of a channel get identical gradients under phase mean
        gm = rm.mix.weight.grad.view(16, 16, 4)
        self.assertTrue(torch.allclose(gm[:, :, 0], gm[:, :, 2], atol=1e-6))
        with torch.no_grad():
            d = F.pixel_unshuffle(rm.eval().reduce(p2), 2)
            expect = p3 + rm.restore(rm.mix(M['hgnetv2_light_phase']._phase_mean(d, 16)))
            self.assertTrue(torch.allclose(rm(p2, p3), expect, atol=1e-5))
            self.assertGreater((rm(p2, p3) - (p3 + rm.restore(rm.mix(d)))).abs().max().item(), 1e-4)

    def test_new_modes_zero_init_identity(self):
        for mod in (CMean(), LowRank(), LowRank(phase_mean=True)):
            for h, w in ((16, 16), (15, 17)):
                p2, p3 = _inputs(h, w)
                self.assertTrue(torch.equal(mod.eval()(p2, p3), p3))

    def test_backbone_new_modes(self):
        self.assertIsInstance(Backbone('B0', p2_skip_mode='c_mean').p2_skip, CMean)
        self.assertFalse(Backbone('B0', p2_skip_mode='rank16').p2_skip.phase_mean)
        self.assertTrue(Backbone('B0', p2_skip_mode='rank16_mean').p2_skip.phase_mean)
        with self.assertRaises(ValueError):
            Backbone('B0', p2_skip_mode='c_mean', p2_phase_eval='mean')

    def test_backbone_mode_switch(self):
        self.assertIsInstance(Backbone('B0', p2_skip_mode='c').p2_skip, Ablate)
        self.assertIsInstance(Backbone('B0', p2_skip_mode='single_phase').p2_skip, Single)
        with self.assertRaises(ValueError):
            Backbone('B0', p2_skip_mode='single_phase', p2_phase_eval='mean')
        with self.assertRaises(ValueError):
            Backbone('B0', p2_skip_mode='bad')


if __name__ == '__main__':
    unittest.main(verbosity=2)
