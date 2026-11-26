"""Inspect archived optimizer settings and saved groups; do not instantiate or run a model."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json
import os
import re
import sys

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
CPU = PAPER / '本地CPU核验_20261103'
SNAP = REC / 'source_snapshot'
SITE = CPU / '.runtime/site-packages'
handles = []
for folder in (Path(sys.base_prefix), Path(sys.base_prefix) / 'DLLs', SITE / 'torch/lib'):
    if folder.is_dir() and hasattr(os, 'add_dll_directory'):
        handles.append(os.add_dll_directory(str(folder)))
sys.dont_write_bytecode = True
sys.path.insert(0, str(SITE))
import torch

def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def ref(path):
    return {'path': str(path), 'sha256': digest(path)}

def group_indices(name, groups):
    return [i for i, item in enumerate(groups) if re.findall(item['params'], name)]

def main():
    output = HERE / 'optimizer_groups_audit.json'
    assert not output.exists()
    source = CPU / 'source_snapshot/engine/core/yaml_config.py'
    stem = SNAP / 'current_code/engine/backbone/hgnetv2_MSCA_EnhancedStem.py'
    msca = SNAP / 'current_code/engine/extre_module/custom_nn/attention/MSCA.py'
    scheduler = CPU / 'source_snapshot/engine/optim/lr_scheduler.py'
    warmup = CPU / 'source_snapshot/engine/optim/warmup.py'
    solver = CPU / 'source_snapshot/engine/solver/det_solver.py'
    engine = CPU / 'source_snapshot/engine/solver/det_engine.py'
    text = ast.unparse(ast.parse(source.read_text(encoding='utf-8')))
    assert 're.findall(pattern, k)' in text
    stem_text = ast.unparse(ast.parse(stem.read_text(encoding='utf-8')))
    assert 'self.high_freq_enhance = nn.Sequential' in stem_text and 'self.spatial_attention = nn.Sequential' in stem_text
    msca_text = ast.unparse(ast.parse(msca.read_text(encoding='utf-8')))
    assert 'class MSCA_Dilated' in msca_text and 'nn.BatchNorm2d(channels // reduction)' in msca_text
    prepared = {entry['run_id']: entry for entry in load(CPU / 'preparation_manifest.json')['prepared_runs']}
    runs = []
    for run in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']:
        args_path = SNAP / run / 'args.json'
        meta_path = SNAP / run / 'checkpoint_metadata.json'
        args = load(args_path)
        cfg = args['yaml_cfg']
        opt = cfg['optimizer']
        groups = opt['params']
        names = load(meta_path)['model_shapes']
        assert opt['type'] == 'AdamW' and opt['weight_decay'] == 0.0001 and opt['betas'] == [0.9, 0.999]
        backbone = cfg[cfg['DEIM']['backbone']]
        assert backbone['freeze_at'] == -1 and backbone['freeze_norm'] is False
        assert args['lrsheduler'] == 'flatcosine' and args['lr_gamma'] == 1.0 and args['warmup_iter'] == 2000
        assert cfg['lr_warmup_scheduler']['warmup_duration'] == 500
        named_bn = sorted(k for k in names if k.startswith('backbone.') and '.bn.' in k and k.endswith(('.weight', '.bias')))
        assert named_bn
        example = named_bn[0]
        selected = group_indices(example, groups)
        assert selected == ([1] if run == 'S1' else [0])
        direct = {'parameter_name': example, 'matched_group_indices': selected}
        hidden = []
        if run in ['N1', 'N4', 'S1']:
            for key in ['backbone.stem.high_freq_enhance.1.weight',
                        'backbone.stem.high_freq_enhance.4.weight',
                        'backbone.stem.spatial_attention.1.weight']:
                assert key in names and group_indices(key, groups) == [0]
                hidden.append({'parameter_name': key, 'matched_group_indices': [0],
                               'source_layer_type': 'BatchNorm2d'})
        if run in ['N2', 'N4', 'S1']:
            prefix = 'backbone.stages.3.blocks.0.aggregation.1.multi_scale_convs.0.'
            for suffix in ['1.weight', '4.weight']:
                key = prefix + suffix
                assert key in names and group_indices(key, groups) == [0]
                hidden.append({'parameter_name': key, 'matched_group_indices': [0],
                               'source_layer_type': 'BatchNorm2d'})
        saved = None
        if run in prepared:
            checkpoint = Path(prepared[run]['checkpoint'])
            assert digest(checkpoint) == prepared[run]['checkpoint_sha256']
            data = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert 'optimizer' in data and 'param_groups' in data['optimizer']
            saved = {'checkpoint': ref(checkpoint), 'last_epoch_metadata': data['last_epoch'],
                     'groups': [{k: group.get(k) for k in ['lr', 'initial_lr', 'weight_decay', 'betas']}
                                | {'parameter_id_count': len(group['params'])}
                                for group in data['optimizer']['param_groups']],
                     'parameter_id_name_mapping_certified': False}
            del data
        runs.append({'run': run, 'args': ref(args_path), 'checkpoint_metadata': ref(meta_path),
                     'optimizer_default_lr': opt['lr'], 'default_weight_decay': opt['weight_decay'],
                     'configured_patterns': groups, 'named_bn_affine_state_keys': len(named_bn),
                     'named_bn_example': direct, 'other_bn_examples_without_bn_in_name': hidden,
                     'saved_optimizer_summary': saved})
    for run in runs[:5]:
        assert run['configured_patterns'][0]['params'] == '^(?=.*backbone)(?!.*norm|bn).*$'
        assert run['configured_patterns'][1]['params'] == '^(?=.*backbone)(?=.*norm|bn).*$'
        assert run['configured_patterns'][0]['lr'] == 0.0004 and run['configured_patterns'][1]['weight_decay'] == 0
    assert runs[-1]['configured_patterns'][0]['params'] == '^(?=.*backbone)(?!.*bn).*$'
    assert runs[-1]['configured_patterns'][0]['lr'] == 0.0002
    assert runs[-1]['configured_patterns'][1]['params'] == '^(?=.*(?:norm|bn)).*$'
    # Parenthesized alternatives behave differently for a representative backbone BN name.
    sample = 'backbone.stem.stem1.bn.weight'
    assert re.search('^(?=.*backbone)(?!.*norm|bn).*$', sample)
    assert not re.search('^(?=.*backbone)(?=.*norm|bn).*$', sample)
    assert re.search('^(?=.*backbone)(?=.*(?:norm|bn)).*$', sample)
    report = {'status': 'OPTIMIZER_NAME_PATTERNS_AND_CHECKPOINT_GROUPS_INSPECTED',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'sources': [ref(p) for p in [source, stem, msca, scheduler, warmup, solver, engine]],
              'source_semantics': 'The inspected YAMLConfig uses re.findall on each named trainable parameter; param groups are defined by names, not module types. An optimizer checkpoint stores group hyperparameters and parameter IDs, but those IDs alone do not authenticate the corresponding names.',
              'n_pattern_issue': 'In (?=.*norm|bn), the unparenthesized second alternative bn is checked at the beginning of the name. For backbone.*.bn.* names without norm, the positive lookahead does not match and the negative lookahead does. The default-decay first group therefore matches these names in the inspected implementation.',
              's_pattern_issue': 'The s backbone group excludes names containing bn, not BatchNorm layers as a type. Sequential BatchNorm parameters can have numeric names without bn or norm and match the first group, with the configured lower rate and default decay.',
              'retained_scheduler': 'All six configs select flatcosine, gamma=1.0, warmup_iter=2000 and construct LinearWarmup with duration 500. In the inspected implementation the latter initializes group LR to base/500, the optimizer update occurs before the flatcosine scheduler step, and subsequent scheduler outputs use a quadratic 2000-iteration warmup with no later reduction because min_lr=base_lr. Exact historical source identity is not certified.',
              'runs': runs, 'historical_source_identity_certified': False,
              'model_instantiations': 0, 'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0,
              'limits': 'Static configuration/name matching is conditional on the inspected source and identified parameter names. Saved optimizer group IDs do not by themselves prove a historical ID-to-name mapping or identical historical source bytes.'}
    with output.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'runs': [x['run'] for x in runs],
                      'saved_group_summaries': {x['run']: x['saved_optimizer_summary']['groups']
                                                for x in runs if x['saved_optimizer_summary']}}, ensure_ascii=False))

if __name__ == '__main__':
    main()
