"""Independently inspect the layer indices and saved regex matches without executing a model."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json
import re

HERE = Path(__file__).resolve().parent
REC, PAPER = HERE.parent, HERE.parent.parent
CPU = PAPER / '本地CPU核验_20261103'
SNAP = REC / 'source_snapshot'
AUDIT = HERE / 'optimizer_groups_audit.json'
OUTPUT = HERE / 'optimizer_semantic_verification.json'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def class_init(tree, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')

def attr_call(node, suffix):
    return isinstance(node, ast.Call) and ast.unparse(node.func).endswith(suffix)

def main():
    assert not OUTPUT.exists()
    audit = load(AUDIT)
    assert audit['status'] == 'OPTIMIZER_NAME_PATTERNS_AND_CHECKPOINT_GROUPS_INSPECTED'
    stem = SNAP / 'current_code/engine/backbone/hgnetv2_MSCA_EnhancedStem.py'
    attention = SNAP / 'current_code/engine/extre_module/custom_nn/attention/MSCA.py'
    init = class_init(ast.parse(stem.read_text(encoding='utf-8')), 'EnhancedStemBlock')
    assignments = {ast.unparse(n.targets[0]): n.value for n in init.body if isinstance(n, ast.Assign) and len(n.targets) == 1}
    high = assignments['self.high_freq_enhance']
    spatial = assignments['self.spatial_attention']
    assert attr_call(high, 'Sequential') and attr_call(high.args[1], 'BatchNorm2d') and attr_call(high.args[4], 'BatchNorm2d')
    assert attr_call(spatial, 'Sequential') and attr_call(spatial.args[1], 'BatchNorm2d')
    attention_init = class_init(ast.parse(attention.read_text(encoding='utf-8')), 'MSCA_Dilated')
    seqs = [n for n in ast.walk(attention_init) if attr_call(n, 'Sequential') and len(n.args) == 5
            and attr_call(n.args[1], 'BatchNorm2d') and attr_call(n.args[4], 'BatchNorm2d')]
    assert len(seqs) == 1
    for run in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']:
        cfg = load(SNAP / run / 'args.json')['yaml_cfg']
        groups = cfg['optimizer']['params']
        meta = load(SNAP / run / 'checkpoint_metadata.json')['model_shapes']
        named = [name for name in meta if name.startswith('backbone.') and '.bn.' in name and name.endswith(('.weight', '.bias'))]
        assert named
        matches = [[j for j, group in enumerate(groups) if re.search(group['params'], name)] for name in named]
        assert all(x == ([1] if run == 'S1' else [0]) for x in matches)
        if run == 'S1':
            # This also rules out regex overlap in the archived backbone state keys.
            assert not [name for name in meta if name.startswith('backbone.') and 'norm' in name and 'bn' not in name]
            for name in ['backbone.stem.high_freq_enhance.1.weight', 'backbone.stem.high_freq_enhance.4.weight',
                         'backbone.stem.spatial_attention.1.weight',
                         'backbone.stages.3.blocks.0.aggregation.1.multi_scale_convs.0.1.weight',
                         'backbone.stages.3.blocks.0.aggregation.1.multi_scale_convs.0.4.weight']:
                assert name in meta and [j for j, g in enumerate(groups) if re.search(g['params'], name)] == [0]
    summaries = {x['run']: x['saved_optimizer_summary'] for x in audit['runs'] if x['saved_optimizer_summary']}
    assert summaries['N0']['groups'][1]['parameter_id_count'] == summaries['N4']['groups'][1]['parameter_id_count'] == 0
    assert summaries['N0']['groups'][0]['weight_decay'] == summaries['N4']['groups'][0]['weight_decay'] == 0.0001
    assert summaries['S1']['groups'][0]['initial_lr'] == 0.0002 and summaries['S1']['groups'][0]['weight_decay'] == 0.0001
    assert summaries['S1']['groups'][1]['initial_lr'] == 0.0004 and summaries['S1']['groups'][1]['weight_decay'] == 0
    report = {'status': 'BN_LAYER_INDICES_REGEX_AND_SAVED_GROUP_VALUES_VERIFIED',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'audit_sha256': sha(AUDIT),
              'source_sha256': {'enhanced_stem': sha(stem), 'dilated_attention': sha(attention)},
              'checked_layer_indices': {'stem.high_freq_enhance': [1, 4], 'stem.spatial_attention': [1],
                                        'dilated.multi_scale_convs': [1, 4]},
              'six_config_bn_name_patterns_checked': True,
              'saved_n0_n4_bn_pattern_group_empty': True,
              'saved_s1_group_initial_lr_and_decay_checked': True,
              'historical_checkpoint_id_name_mapping_proven': False,
              'historical_source_identity_certified': False,
              'model_instantiations': 0, 'model_forwards': 0, 'evaluator_executions': 0, 'optimizer_updates': 0}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'status': report['status'], 'checked_runs': 6, 'no_model_run': True}))

if __name__ == '__main__':
    main()
