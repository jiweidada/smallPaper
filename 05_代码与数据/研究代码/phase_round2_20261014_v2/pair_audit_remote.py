"""Read-only resolved-recipe audit. No GPU construction or training."""
import ast
import copy
import hashlib
import json
import importlib.metadata
import os
import pathlib
import sys
import yaml

storage = pathlib.Path('/25085400120' if pathlib.Path('/25085400120/DEIM').is_dir() else '/24085403014')
phase = next(p for p in [storage/'DEIM/experiments/phase_mechanism_20261012', storage/'experiments/phase_mechanism_20261012'] if p.is_dir())
code = phase/'code'
os.chdir(code)
sys.path.insert(0, str(code))
loader_source = (code/'engine/core/yaml_utils.py').read_text()
functions = [node for node in ast.parse(loader_source).body if isinstance(node, ast.FunctionDef) and node.name in ['load_config', 'merge_dict']]
namespace = {'os': os, 'copy': copy, 'yaml': yaml, 'Dict': dict, 'INCLUDE_KEY': '__include__'}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<original-yaml-loader-functions>', 'exec'), namespace)
resolved = namespace['load_config']('configs/experiments/phase/proto96.yml')
roots = [storage/'DEIM/experiments', storage/'experiments']
candidates = []
for root in roots:
    if not root.is_dir():
        continue
    direct = root/'visdrone_light_20260928_090716/structural_candidates/outputs/p2_skip_main_seed0/args.json'
    if direct.is_file():
        candidates.append(direct)
    for directory in root.glob('*c_mainline*'):
        candidates.extend(directory.rglob('args.json'))

fields = ['epoches', 'flat_epoch', 'no_aug_epoch', 'warmup_iter', 'lr_gamma', 'lrsheduler',
          'lr_warmup_scheduler', 'optimizer', 'ema', 'scaler', 'use_ema', 'use_amp',
          'clip_max_norm', 'sync_bn', 'find_unused_parameters', 'HybridEncoder', 'DFINETransformer', 'DEIMCriterion',
          'train_dataloader.dataset.transforms', 'train_dataloader.total_batch_size',
          'train_dataloader.collate_fn', 'train_dataloader.shuffle', 'train_dataloader.num_workers',
          'train_dataloader.drop_last', 'val_dataloader.dataset.transforms', 'val_dataloader.total_batch_size', 'evaluator']
def get(d, key):
    for part in key.split('.'):
        if not isinstance(d, dict) or part not in d:
            return {'MISSING': True}
        d = d[part]
    return d

geometry_paths = list(pathlib.Path(sys.prefix).glob('lib/python*/site-packages/torchvision/transforms/v2/_geometry.py'))
defaults = {}
default_evidence = []
for path in geometry_paths:
    source = path.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == 'RandomZoomOut':
            ctor = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            names = [arg.arg for arg in ctor.args.args][-len(ctor.args.defaults):]
            defaults = {name: ast.literal_eval(value) for name, value in zip(names, ctor.args.defaults) if name == 'p'}
            default_evidence.append({'path': str(path), 'sha256': hashlib.sha256(source.encode()).hexdigest(), 'defaults': defaults})
def effective(value, key):
    value = copy.deepcopy(value)
    if key == 'train_dataloader.dataset.transforms' and isinstance(value, dict):
        for operation in value.get('ops', []):
            if operation.get('type') == 'RandomZoomOut' and 'p' not in operation and 'p' in defaults:
                operation['p'] = defaults['p']
    return value

result = {'phase_root': str(phase), 'resolved': resolved, 'comparisons': [], 'default_evidence': default_evidence,
          'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision']},
          'yaml_loader_sha256': hashlib.sha256(loader_source.encode()).hexdigest(),
          'init_sha256': hashlib.sha256((phase/'artifacts/common_init_seed0.pth').read_bytes()).hexdigest(),
          'solver_selection_lines': []}
for path in candidates:
    old = json.loads(path.read_text())
    cfg = old.get('yaml_cfg', old)
    checks = {key: {'equal': get(cfg, key) == get(resolved, key),
                    'effective_equal': effective(get(cfg, key), key) == effective(get(resolved, key), key),
                    'old': get(cfg, key), 'new': get(resolved, key)} for key in fields}
    result['comparisons'].append({'path': str(path), 'seed': old.get('seed', cfg.get('seed')),
                                  'resume': old.get('resume'), 'checks': checks,
                                  'all_listed_fields_equal': all(v['equal'] for v in checks.values()),
                                  'all_effective_fields_equal': all(v['effective_equal'] for v in checks.values()), 'args': old})
solver = code/'engine/solver/det_solver.py'
for number, line in enumerate(solver.read_text().splitlines(), 1):
    if any(token in line for token in ['best_stg2', 'best_stat', 'test_stats', 'coco_eval_bbox']):
        result['solver_selection_lines'].append({'line': number, 'text': line.strip()})
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
