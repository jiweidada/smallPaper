"""Shared paired-run audits and fail-closed checkpoint guards."""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

INIT_SHA = '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
NEW_KEY = 'decoder.query_recruitment.project.weight'


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def source_manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_sha(p)
            for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}


def canonical_cfg(cfg):
    v = copy.deepcopy(cfg)
    for k in ('config', 'output_dir', 'summary_dir', 'resume', 'tuning',
              'test_only', 'path', 'mode', 'print_method', 'print_rank', 'local_rank'):
        v.pop(k, None)
    for split in ('train', 'val'):
        d = v[f'{split}_dataloader']['dataset']
        d['img_folder'] = f'<DATA>/{split}/images'
        d['ann_file'] = f'<DATA>/{split}/annotations.json'
    return v


def strict_common_init(model, path):
    if file_sha(path) != INIT_SHA:
        raise RuntimeError('Common initialization SHA256 mismatch')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    current = model.state_dict()
    missing = set(current)-set(state)
    allowed = {NEW_KEY} if hasattr(model.decoder, 'query_recruitment') else set()
    extra = set(state)-set(current)
    bad = [k for k in set(current)&set(state) if current[k].shape != state[k].shape or current[k].dtype != state[k].dtype]
    if missing != allowed or extra or bad:
        raise RuntimeError(f'Non-matching common initialization: missing={missing}, extra={extra}, bad={bad}')
    if allowed and torch.count_nonzero(current[NEW_KEY]).item() != 0:
        raise RuntimeError('P2 projection must start at zero')
    model.load_state_dict(state, strict=not allowed)
    return {'sha256': INIT_SHA, 'loaded_keys': len(state), 'new_zero_keys': sorted(allowed)}


def tensor_digest(value, h):
    if torch.is_tensor(value):
        a = value.detach().cpu().contiguous()
        h.update(str((tuple(a.shape), str(a.dtype))).encode())
        h.update(a.numpy().tobytes())
    elif isinstance(value, dict):
        for k in sorted(value):
            h.update(str(k).encode()); tensor_digest(value[k], h)
    elif isinstance(value, (list, tuple)):
        for a in value: tensor_digest(a, h)
    else:
        h.update(repr(value).encode())


def install_runtime(cfg, receipt_dir):
    from engine.solver._solver import BaseSolver
    from engine.solver.det_solver import DetSolver
    from engine.solver import det_solver
    from engine.misc import dist_utils
    root = Path(__file__).resolve().parent
    receipt_dir = Path(receipt_dir); receipt_dir.mkdir(parents=True, exist_ok=True)
    identity = {'schema': 'p2_native_full132_v1', 'config': canonical_cfg(cfg.yaml_cfg),
                'sources': source_manifest(root), 'init_sha256': INIT_SHA}
    (receipt_dir/'resolved_config.json').write_text(json.dumps(cfg.yaml_cfg, indent=2, default=str))
    (receipt_dir/'run_identity.json').write_text(json.dumps(identity, indent=2))
    native_state = DetSolver.state_dict
    native_load = DetSolver.load_state_dict
    native_train = DetSolver.train

    def guarded_state(self):
        state = native_state(self)
        state['screen_identity'] = copy.deepcopy(identity)
        return state

    def guarded_load(self, state):
        if state.get('screen_identity') != identity:
            raise RuntimeError('Checkpoint configuration/source identity mismatch; no state loaded')
        for name in ('model', 'ema'):
            incoming = state[name]['module'] if name == 'ema' else state[name]
            module = self.ema.module if name == 'ema' else dist_utils.de_parallel(self.model)
            current = module.state_dict()
            if set(incoming) != set(current) or any(incoming[k].shape != current[k].shape or incoming[k].dtype != current[k].dtype for k in current):
                raise RuntimeError(f'Checkpoint {name} tensor schema mismatch; no state loaded')
        if hasattr(self, 'optimizer'):
            groups = state.get('optimizer', {}).get('param_groups', [])
            if len(groups) != len(self.optimizer.param_groups) or any(len(a['params']) != len(b['params']) for a,b in zip(groups, self.optimizer.param_groups)):
                raise RuntimeError('Checkpoint optimizer schema mismatch; no state loaded')
        if self.scaler is not None and 'scaler' not in state:
            raise RuntimeError('Checkpoint lacks scaler state; no state loaded')
        return native_load(self, state)

    def guarded_tuning(self, path):
        receipt = strict_common_init(dist_utils.de_parallel(self.model), path)
        (receipt_dir/'init_load.json').write_text(json.dumps(receipt, indent=2))

    def audited_train(self):
        native_train(self)
        model = dist_utils.de_parallel(self.model)
        all_ids = [id(p) for g in self.optimizer.param_groups for p in g['params']]
        assert len(all_ids) == len(set(all_ids)), 'Optimizer contains duplicate parameters'
        assert set(all_ids) == {id(p) for p in model.parameters() if p.requires_grad}
        record = {'optimizer_parameter_coverage': True, 'total_parameters': sum(p.numel() for p in model.parameters()),
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'group_sizes': [sum(p.numel() for p in g['params']) for g in self.optimizer.param_groups]}
        (receipt_dir/'optimizer_coverage.json').write_text(json.dumps(record, indent=2))
        steps = {'inputs': 0, 'attempts': 0, 'updates': 0}
        def prehook(_m, args, kwargs):
            if not _m.training or steps['inputs'] >= 100: return
            h = hashlib.sha256(); tensor_digest(args[0], h); tensor_digest(kwargs.get('targets'), h)
            with (receipt_dir/'first100_inputs.jsonl').open('a') as f:
                f.write(json.dumps({'index': steps['inputs'], 'sha256': h.hexdigest(), 'shape': list(args[0].shape)})+'\n')
            steps['inputs'] += 1
        model.register_forward_pre_hook(prehook, with_kwargs=True)
        self.optimizer.register_step_post_hook(lambda _o, _a, _k: steps.__setitem__('updates', steps['updates']+1))
        original_scaler_step = self.scaler.step
        def audited_scaler_step(optimizer, *args, **kwargs):
            before = steps['updates']; scale = self.scaler.get_scale()
            found = self.scaler._found_inf_per_device(optimizer)
            inf = sum(float(v.item()) for v in found.values())
            result = original_scaler_step(optimizer, *args, **kwargs)
            steps['attempts'] += 1
            if steps['attempts'] <= 100 or steps['attempts'] % 100 == 0 or steps['updates'] == before:
                branch = getattr(model.decoder, 'query_recruitment', None)
                with (receipt_dir/'optimizer_updates.jsonl').open('a') as f:
                    f.write(json.dumps({'attempt': steps['attempts'], 'actual_updates': steps['updates'],
                                       'updated': steps['updates'] > before, 'scale': scale, 'found_inf': inf,
                                       'p2_weight_norm': float(branch.project.weight.detach().float().norm()) if branch else None,
                                       'p2_info': getattr(model.decoder, '_p2_last_info', None)})+'\n')
            return result
        self.scaler.step = audited_scaler_step

    # Profiling differs between arms, and must not consume the training RNG.
    native_stats = det_solver.stats
    def rng_neutral_stats(*args, **kwargs):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
                candidate_stats = native_stats(*args, **kwargs)
        finally:
            random.setstate(python_state); np.random.set_state(numpy_state)
        # The existing formal132 baseline uses the native profiler, including
        # its CPU/GPU RNG advance. Replay that same BASE profiler after the P2
        # cost profile, rather than changing the baseline's data/DN RNG stream.
        if cfg.yaml_cfg.get('profiling_rng_mode') == 'native_reference_replay' and kwargs.get('module') is None:
            from engine.deim.deim import DEIM
            active_cfg = args[0]
            current = active_cfg._model
            decoder = copy.deepcopy(current.decoder)
            del decoder.query_recruitment
            del decoder._get_decoder_input  # restores the native class method
            decoder._p2_context = None
            reference = DEIM(copy.deepcopy(current.backbone), copy.deepcopy(current.encoder), decoder,
                             o2m_c=False, num_classes=active_cfg.yaml_cfg['num_classes'])
            before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            try:
                active_cfg._model = reference
                native_stats(active_cfg)
            finally:
                active_cfg._model = current
            after = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            (receipt_dir/'profiling_rng_replay.json').write_text(json.dumps({
                'mode':'native_reference_replay','reference_model':'DEIM',
                'CPU_rng_before':before,'CPU_rng_after':after},indent=2))
        return candidate_stats
    det_solver.stats = rng_neutral_stats
    DetSolver.state_dict, DetSolver.load_state_dict, DetSolver.train = guarded_state, guarded_load, audited_train
    BaseSolver.load_tuning_state = guarded_tuning
    return identity
