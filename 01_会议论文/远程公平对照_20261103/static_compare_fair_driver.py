"""Read-only comparison of the interrupted driver with its frozen source recipe."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
CPU_SOURCE = PAPER / '本地CPU核验_20261103/source_snapshot'
OLD_REC = PAPER / '修订记录_20261103'
MANIFEST = ROOT / 'remote_evidence/source_manifest.json'
PROTOCOL = ROOT / 'execution_protocol.json'
DRIVER = ROOT / 'fair_runner.py'
LOADER = ROOT / 'remote_evidence/source_loader.py'
BASE = ROOT / 'remote_evidence/interrupted_BASE_20261105/execution_configs/BASE.yml'
FULL = ROOT / 'remote_evidence/interrupted_BASE_20261105/execution_configs/FULL.yml'
N0 = OLD_REC / 'source_snapshot/N0/args.json'
REPORT = ROOT / 'fair_driver_static_comparison_v2_20261105.json'
EXPECTED_PROTOCOL = '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def node_named(tree, name, kind=ast.FunctionDef):
    hits = [n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name]
    assert len(hits) == 1, name
    return hits[0]

def call_position(node, name):
    hits = [(n.lineno, n.col_offset) for n in ast.walk(node)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    assert len(hits) == 1, (name, hits)
    return hits[0]

def call_line(node, name):
    return call_position(node, name)[0]

assert not REPORT.exists() and sha(PROTOCOL) == EXPECTED_PROTOCOL
frozen = {x['path']: x['sha256'] for x in load(PROTOCOL)['frozen_files']}
manifest = {x['snapshot']: x['sha256'] for x in load(MANIFEST)}
selected = ['engine/solver/det_engine.py', 'engine/solver/det_solver.py', 'engine/solver/_solver.py',
            'engine/optim/lr_scheduler.py', 'engine/optim/ema.py', 'engine/core/yaml_config.py',
            'engine/data/dataloader.py', 'engine/data/dataset/coco_eval.py',
            'engine/misc/dist_utils.py', 'engine/misc/profiler_utils.py']
sources = {}
for relative in selected:
    key = 'source_snapshot/' + relative
    path = CPU_SOURCE / relative
    actual = sha(path)
    assert actual == frozen[key] == manifest[key]
    sources[relative] = {'path': str(path), 'sha256': actual}
assert sha(DRIVER) == frozen['fair_runner.py'] and sha(LOADER) == frozen['source_loader.py']
assert sha(BASE) == frozen['execution_configs/BASE.yml']
assert sha(FULL) == frozen['execution_configs/FULL.yml']
historical_manifest = {str(Path(x['snapshot']).resolve()): x['sha256'] for x in load(OLD_REC / 'source_manifest.json')}
assert sha(N0) == historical_manifest[str(N0.resolve())]

engine = ast.parse((CPU_SOURCE / 'engine/solver/det_engine.py').read_text(encoding='utf-8'))
driver = ast.parse(DRIVER.read_text(encoding='utf-8'))
orig_train = node_named(engine, 'train_one_epoch')
new_train = node_named(driver, 'train_run')
scaler_condition = [n for n in ast.walk(orig_train) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'scaler is not None']
assert len(scaler_condition) == 1 and scaler_condition[0].orelse
fp32 = ast.Module(body=scaler_condition[0].orelse, type_ignores=[])
original_names = ['model', 'criterion', 'optimizer.zero_grad', 'loss.backward',
                  'torch.nn.utils.clip_grad_norm_', 'optimizer.step']
new_names = original_names + ['ema.update', 'lrfn']
original_positions = {name: call_position(fp32, name) for name in original_names}
original_positions.update({'ema.update': call_position(orig_train, 'ema.update'),
                           'lr_scheduler.step': call_position(orig_train, 'lr_scheduler.step')})
new_positions = {name: call_position(new_train, name) for name in new_names}
assert list(original_positions.values()) == sorted(original_positions.values())
assert list(new_positions.values()) == sorted(new_positions.values())
original_clip = next(n for n in ast.walk(fp32) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == 'torch.nn.utils.clip_grad_norm_')
new_clip = next(n for n in ast.walk(new_train) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'torch.nn.utils.clip_grad_norm_')
assert not any(k.arg == 'error_if_nonfinite' for k in original_clip.keywords)
assert any(k.arg == 'error_if_nonfinite' and isinstance(k.value, ast.Constant) and k.value.value is True
           for k in new_clip.keywords)
orig_eval = node_named(engine, 'evaluate')
assert call_line(orig_eval, 'criterion.eval')
assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'criterion' for n in ast.walk(orig_eval))
new_eval = node_named(driver, 'evaluate')
assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'criterion' for n in ast.walk(new_eval))
prep = node_named(driver, 'prepare_epoch')
assert call_line(prep, 'loader.set_epoch') < call_line(prep, 'loader.generator.manual_seed')
assert call_line(prep, 'loader.sampler.generator.manual_seed')
controls = node_named(driver, 'controls')
det = next(n for n in ast.walk(controls) if isinstance(n, ast.Call)
           and ast.unparse(n.func) == 'torch.use_deterministic_algorithms')
assert any(k.arg == 'warn_only' and isinstance(k.value, ast.Constant) and k.value.value is True
           for k in det.keywords)

base, full, historical = load(BASE), load(FULL), load(N0)['yaml_cfg']
keys = ['train_dataloader', 'val_dataloader', 'optimizer', 'ema', 'epoches', 'lr_gamma', 'seed',
        'use_amp', 'clip_max_norm', 'lrsheduler', 'warmup_iter', 'flat_epoch', 'no_aug_epoch']
equality = {key: base[key] == full[key] for key in keys}
assert all(equality.values())
assert base['train_dataloader']['total_batch_size'] == 4
assert base['val_dataloader']['total_batch_size'] == 4
assert historical['train_dataloader']['total_batch_size'] == 16
assert historical['val_dataloader']['total_batch_size'] == 16
assert base['use_amp'] is False and base['epoches'] == 320 and base['seed'] == 0
assert base['lr_gamma'] == 1.0 and base['clip_max_norm'] == 0.1
assert base['optimizer'] == historical['optimizer']

first = {a: ROOT / ('parallel_evidence/' + a + '_actual_first100.jsonl') for a in ['BASE', 'FULL']}
base_rows, full_rows = [[json.loads(s) for s in first[a].read_text(encoding='utf-8').splitlines()]
                        for a in ['BASE', 'FULL']]
expected = load(ROOT / 'remote_evidence/interrupted_BASE_20261105/preflight/BASE.first100.json')
assert len(base_rows) == len(full_rows) == len(expected) == 100
assert base_rows == full_rows == expected
assert [x['step'] for x in base_rows] == list(range(100))

report = {'status': 'SELECTED_FAIR_DRIVER_PATHS_STATICALLY_COMPARED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(PROTOCOL),
          'source_manifest_sha256': sha(MANIFEST), 'driver_sha256': sha(DRIVER), 'source_loader_sha256': sha(LOADER),
          'selected_frozen_sources': sources, 'base_config_sha256': sha(BASE), 'full_config_sha256': sha(FULL),
          'historical_n0_config_sha256': sha(N0), 'selected_control_config_equal': equality,
          'batch_sizes': {'new_train': 4, 'new_validation': 4, 'historical_N0_train': 16, 'historical_N0_validation': 16},
          'selected_static_fp32_call_positions': {'frozen_original': original_positions, 'fair_driver': new_positions},
          'position_format': '[one-based line, zero-based column]; columns distinguish calls sharing a line',
          'prior_line_only_report_sha256': sha(ROOT / 'fair_driver_static_comparison_20261105.json'),
          'observations': [
              'The inspected original FP32 branch and fair driver both order forward, criterion, zero_grad, backward, gradient clipping, optimizer step, EMA update and their post-update LR operation as recorded.',
              'The new driver requests error_if_nonfinite=True for gradient clipping; the inspected original call does not pass that keyword.',
              'The inspected original validation sets criterion.eval(), but neither inspected validation function calls criterion; the fair driver calls criterion.train() at the start of each training epoch.',
              'The new driver reseeds both loader and sampler generators by epoch; the inspected solver calls set_epoch and only calls sampler.set_epoch in distributed mode.',
              'Both actual first 100 batch records are byte-for-byte equal and match saved BASE preflight records; this evidence does not cover later batches or epochs.',
              'sync_bn=true appears in config, but the fair driver does not invoke the original distributed model wrapper or SyncBatchNorm conversion. The frozen wrapper applies conversion only if distributed is initialized; individual model layers were not exhaustively inspected here.',
              'The driver requests deterministic algorithms with warn_only=True; this is not a guarantee of bitwise deterministic execution.',
              'The inspected original solver also invokes a model-statistics routine and training-batch visualization; the fair driver omits these. Their full effects on random-state consumption were not measured.' ],
          'actual_first100': {a: {'path': str(p), 'sha256': sha(p)} for a, p in first.items()},
          'limitations': ['Only selected code paths and configuration fields were checked, not full runtime equivalence.',
                          'The current frozen original-source snapshot is not proof of the exact code used for historical N0.',
                          'Configuration equality and 100 matched batches do not establish complete matched training or explain the validation collapse.',
                          'No model forward, evaluator, training, or optimizer update was executed.'],
          'model_executions': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
with REPORT.open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'config_fields_equal': len(equality),
                  'matched_first_batches': len(base_rows), 'source_files': len(sources),
                  'original_positions': original_positions, 'driver_positions': new_positions}))
