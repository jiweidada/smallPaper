"""Standard-library safety/recipe tests; never import torch or run a model."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import replay_original_ema as r

HERE = Path(__file__).resolve().parent
INDEX = HERE.parent.parent/'data'/'00_实验计划与证据索引'
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejects(name, call):
    try:
        call()
    except (ValueError, RuntimeError):
        checks.append(name)
        return
    raise AssertionError('Expected rejection: '+name)


def main():
    module = SimpleNamespace(training=False)
    batch = (SimpleNamespace(shape=(4, 3, 640, 640)),)
    budget = r.ForwardBudget()
    for _ in range(137):
        budget.before(module, batch)
    check('exactly 137 batches account for 548 images', (budget.batches, budget.images) == (137, 548))
    rejects('138th batch rejected before model execution', lambda: budget.before(module, batch))
    check('rejected excess batch is not added to executed budget', (budget.batches, budget.images) == (137, 548))
    rejects('training mode rejected', lambda: r.ForwardBudget().before(SimpleNamespace(training=True), batch))
    for shape in ((1, 3, 640, 640), (4, 3, 800, 800), (4, 4, 640, 640)):
        rejects('non-native batch shape rejected '+str(shape), lambda shape=shape: r.ForwardBudget().before(module, (SimpleNamespace(shape=shape),)))
    base = [.2]*12
    check('identical metrics pass original gate', r.reproduction_gate(base, base)['gate_0_02pp_passed'])
    high = base.copy(); high[0] += .00021
    low = base.copy(); low[3] -= .00021
    rejects('non-finite native metric rejected', lambda: r.reproduction_gate([float('nan')]*12, base))
    rejects('wrong metric vector size rejected', lambda: r.reproduction_gate(base[:6], base))
    check('positive AP drift above 0.02pp fails', not r.reproduction_gate(high, base)['gate_0_02pp_passed'])
    check('negative APs drift above 0.02pp fails', not r.reproduction_gate(low, base)['gate_0_02pp_passed'])
    near = base.copy(); near[0] += .00019
    check('0.019pp within original threshold passes', r.reproduction_gate(near, base)['gate_0_02pp_passed'])
    targets = r.read(HERE/'source_binding.json')['targets']
    for seed in (1, 2):
        root = INDEX/'正式132轮基线模型下载_20261008_v1'/f'seed{seed}'/'run'
        args = r.read(root/'args.json')
        before = deepcopy(args)
        paths = {'output': Path('/new/replay/output'), 'val_images': Path('/existing/images'), 'val_ann': Path('/existing/val.json')}
        config = r.bound_config(args, paths)
        check(f'seed{seed} original args never mutated', args == before)
        # Compare the full dictionary after reversing precisely allowed portability/run-mode changes.
        restored = deepcopy(config)
        for key in ('resume', 'tuning', 'device', 'output_dir', 'use_swanlab', '__include__', 'config'):
            if key in before['yaml_cfg']:
                restored[key] = deepcopy(before['yaml_cfg'][key])
            else:
                restored.pop(key, None)
        for key in ('img_folder', 'ann_file'):
            restored['val_dataloader']['dataset'][key] = before['yaml_cfg']['val_dataloader']['dataset'][key]
        check(f'seed{seed} recipe exactly retained beyond explicit path/run changes', restored == before['yaml_cfg'])
        check(f'seed{seed} native yolo metric branch retained', config['yolo_metrice'] == before['yaml_cfg']['yolo_metrice'])
        check(f'seed{seed} args/log identities match known archive', r.sha(root/'args.json', lf=True) == targets[str(seed)]['original_args_LF_sha256']
              and r.sha(root/'log.txt', lf=True) == targets[str(seed)]['original_log_LF_sha256'])
        altered = deepcopy(args); altered['yaml_cfg']['ram_cache'] = True
        rejects(f'seed{seed} alternate cached validation path rejected', lambda: r.bound_config(altered, paths))
    runtime, evaluator = r.load_support(HERE.parent/'fine_residual_geometry_20261008_v5')
    native_root = HERE.parent/'p2_candidate_seed1_full132_20261007_v1'
    source_binding = runtime.verify_training_snapshot(native_root)
    check('real bound full native snapshot remains intact', len(source_binding['files']) == 468)
    tree = ast.parse((HERE/'replay_original_ema.py').read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    direct = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'evaluate']
    check('one direct original evaluate call with test_only false', len(direct) == 1
          and next(k.value.value for k in direct[0].keywords if k.arg == 'test_only') is False)
    names = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    check('worker has no deploy, solver val/fit/train, backward or optimizer step calls',
          not names.intersection({'deploy', 'val', 'fit', 'train', 'backward', 'step', 'zero_grad'}))
    attributes = {n.attr for n in ast.walk(worker) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'cfg'}
    check('lazy optimizer/scaler/EMA/training loader are never requested',
          not attributes.intersection({'optimizer', 'ema', 'scaler', 'train_dataloader', 'lr_scheduler', 'lr_warmup_scheduler'}))
    check('worker retains strict checkpoint load and no-grad native path', any(
        isinstance(n.func, ast.Attribute) and n.func.attr == 'load_state_dict' and any(k.arg == 'strict' and isinstance(k.value, ast.Constant) and k.value.value is True for k in n.keywords)
        for n in calls))
    if 'ORIGINAL_EMA_REPLAY_WORKER_TOKEN' in os.environ:
        raise RuntimeError('Synthetic checks require no running GPU-worker token')
    rejects('private worker cannot bypass supervisor', lambda: r.worker('/missing/unused/bundle'))
    check('one-hour supervisor limit is explicit', r.TIMEOUT_SECONDS == 3600)
    with tempfile.TemporaryDirectory(prefix='gpu_replay_static_') as tmp:
        p = Path(tmp)/'identity.txt'; p.write_bytes(b'a\r\nb\r\n')
        q = Path(tmp)/'identity_lf.txt'; q.write_bytes(b'a\nb\n')
        check('LF normalization verifies text transport without changing raw evidence', r.sha(p, lf=True) == r.sha(q) and r.sha(p) != r.sha(q))
        # These temporary schema fixtures do NOT contain GPU results. They test
        # the admission logic only and are removed with this temporary directory.
        def fixture(seed, suffix, drift=0., mode='native_gpu_replay'):
            out = Path(tmp)/suffix; out.mkdir()
            stats = targets[str(seed)]['original_stats'].copy(); stats[0] += drift
            gate = r.reproduction_gate(stats, targets[str(seed)]['original_stats'])
            r.write(out/'execution_identity.json', {'receipt': {'seed': seed, 'target': targets[str(seed)]}})
            r.write(out/'native_metrics.json', {'stats': stats, **gate})
            r.write(out/'completion.json', {'status': 'ORIGINAL_EMA_REPLAY_COMPLETED', 'mode': mode,
                'seed': seed, 'state_unchanged': True, 'images': 548, 'batch_forwards': 137,
                'optimizer_constructions': 0, 'parameter_or_EMA_updates': 0, **gate})
            r.write(out/'predictions.json', {'synthetic_schema_fixture': True})
            (out/'precision_recall.npz').write_bytes(b'SCHEMA FIXTURE ONLY, NOT REAL ARRAYS')
            r.write(out/'forward_counts.json', {'synthetic_schema_fixture': True})
            r.write(out/'manifest.json', {'script_sha256': r.sha(r.__file__), 'source_binding_sha256': r.sha(HERE/'source_binding.json'),
                'files': {x.name: r.sha(x) for x in out.iterdir() if x.is_file()}})
            r.write(out/'supervisor.json', {'complete': True, 'exit_code': 0, 'timed_out': False,
                'seed': seed, 'reproduction_gate_passed': gate['gate_0_02pp_passed']})
            return out
        one, two = fixture(1, 'one'), fixture(2, 'two')
        gate_result = r.check_replays([one, two])
        check('two consistent synthetic receipt schemas satisfy only the numeric prerequisite',
              gate_result['seeds'] == [1, 2] and not gate_result['training_started'] and not gate_result['training_authorization_inferred'])
        rejects('one passing seed cannot permit the next stage', lambda: r.check_replays([one]))
        rejects('duplicate seed cannot replace both seeds', lambda: r.check_replays([one, one]))
        drifted = fixture(2, 'drifted', drift=.00021)
        rejects('one completed but numerically failed replay blocks the next stage', lambda: r.check_replays([one, drifted]))
        synthetic = fixture(2, 'explicit_synthetic', mode='synthetic_engineering')
        rejects('explicit engineering mode cannot count as native GPU results', lambda: r.check_replays([one, synthetic]))
        (two/'native_metrics.json').write_text('{}', encoding='utf-8')
        rejects('changed result file fails its manifest before admission', lambda: r.check_replays([one, two]))
    check('no torch or evaluator imported during all engineering checks', 'torch' not in sys.modules
          and not any(k == 'pycocoeval' or k.startswith('pycocoeval.') for k in sys.modules))
    report = {'status': 'PASS_STDLIB_REPLAY_GUARDS_AND_SOURCE_REVIEW', 'checks': checks, 'check_count': len(checks),
              'bound_native_files_verified': len(source_binding['files']),
              'model_forwards': 0, 'checkpoint_loads': 0, 'evaluator_calls': 0, 'optimizer_or_parameter_updates': 0,
              'CUDA_initializations': 0, 'remote_accesses': 0, 'real_images_or_prediction_arrays_read': 0,
              'actual_gpu_replay_verified': False,
              'positive_gate_case_uses_temporary_synthetic_schema_fixtures_not_GPU_evidence': True,
              'files_sha256': {p.name: r.sha(p) for p in (HERE/'replay_original_ema.py', HERE/'source_binding.json', Path(__file__))}}
    r.write(HERE/'engineering_static.json', report)
    print(json.dumps({'status': report['status'], 'checks': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
