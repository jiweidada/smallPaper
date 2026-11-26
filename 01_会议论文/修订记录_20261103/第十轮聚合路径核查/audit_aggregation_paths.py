"""Trace aggregation projections in saved settings and source, without model execution."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import ast
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
SOURCE = REC / 'source_snapshot/current_code/engine/backbone'
REMOTE = PAPER / '远程公平对照_20261103/remote_evidence'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

manifest = {v['snapshot']: v for v in read(REMOTE / 'source_manifest.json')}
sources = {}
for filename in ['hgnetv2.py', 'hgnetv2_MSCA.py', 'hgnetv2_MSCA_EnhancedStem.py', 'hgnetv2_silu.py']:
    path = SOURCE / filename
    digest = sha(path)
    frozen_name = 'source_snapshot/engine/backbone/' + filename
    frozen_entry = manifest.get(frozen_name)
    if filename in ['hgnetv2.py', 'hgnetv2_MSCA_EnhancedStem.py']:
        assert frozen_entry is not None
    if frozen_entry:
        assert frozen_entry['sha256'] == digest
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    block = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == 'HG_Block')
    branches = {}
    for node in ast.walk(block):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == 'agg' and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)):
            continue
        name = test.comparators[0].value
        if name not in ['se', 'ese', 'simam', 'msca_lite', 'msca_dilated']:
            continue
        convs = []
        gate = None
        sequence = None
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == 'ConvBNAct':
                convs.append({'args': [ast.unparse(a) for a in call.args],
                              'kwargs': {k.arg: ast.unparse(k.value) for k in call.keywords}})
            if any(isinstance(t, ast.Name) and t.id == 'att' for t in statement.targets):
                gate = ast.unparse(call)
            if any(isinstance(t, ast.Attribute) and t.attr == 'aggregation' for t in statement.targets):
                sequence = ast.unparse(call)
        if name == 'se':
            assert [c['args'] for c in convs] == [['total_chs', 'out_chs // 2'], ['out_chs // 2', 'out_chs']]
            assert gate is None
        else:
            assert [c['args'] for c in convs] == [['total_chs', 'out_chs']]
            assert gate is not None
        assert all(c['kwargs']['kernel_size'] == '1' and c['kwargs']['stride'] == '1' for c in convs)
        branches[name] = {'line_start_1based': node.lineno,
                          'line_end_1based': node.body[-1].end_lineno,
                          'projection_convs': convs, 'gate': gate, 'sequence': sequence}
    assert {'se', 'ese', 'simam'} <= branches.keys()
    if 'MSCA' in filename:
        assert {'msca_lite', 'msca_dilated'} <= branches.keys()
    model_class = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name.startswith('HGNetv2'))
    constructor = next(v for v in model_class.body if isinstance(v, ast.FunctionDef) and v.name == '__init__')
    default_nodes = dict(zip([a.arg for a in constructor.args.args][-len(constructor.args.defaults):], constructor.args.defaults))
    defaults = {k: ast.literal_eval(v) for k, v in default_nodes.items() if k in ['agg', 'hierarchical_attention']}
    sources[filename] = {'path': str(path), 'sha256': digest,
                         'frozen_manifest_path': frozen_name if frozen_entry else None,
                         'frozen_manifest_match': True if frozen_entry else None,
                         'constructor_defaults': defaults, 'branches': branches}

classes = {'N0': ('HGNetv2', 'hgnetv2.py'), 'BASE': ('HGNetv2', 'hgnetv2.py'),
           'N1': ('HGNetv2_MSCA_EnhancedStem', 'hgnetv2_MSCA_EnhancedStem.py'),
           'N2': ('HGNetv2_MSCA', 'hgnetv2_MSCA.py'), 'N3': ('HGNetv2_Silu', 'hgnetv2_silu.py'),
           'N4': ('HGNetv2_MSCA_EnhancedStem', 'hgnetv2_MSCA_EnhancedStem.py'),
           'S1': ('HGNetv2_MSCA_EnhancedStem', 'hgnetv2_MSCA_EnhancedStem.py'),
           'FULL': ('HGNetv2_MSCA_EnhancedStem', 'hgnetv2_MSCA_EnhancedStem.py')}
settings = {}
for run, (model_class, filename) in classes.items():
    path = REMOTE / 'execution_configs' / (run + '.yml') if run in ['BASE', 'FULL'] else REC / 'source_snapshot' / run / 'args.json'
    raw = read(path)
    cfg = raw.get('yaml_cfg', raw)
    assert cfg['DEIM']['backbone'] == model_class
    params = cfg[model_class]
    defaults = sources[filename]['constructor_defaults']
    assert not params.get('hierarchical_attention', defaults.get('hierarchical_attention', False))
    agg = params.get('agg', defaults.get('agg'))
    selected = agg if isinstance(agg, list) else [agg] * 4
    assert len(selected) == 4
    if run in ['N0', 'BASE']:
        assert selected == ['se'] * 4
    elif run in ['N1', 'N3']:
        assert selected == ['ese'] * 4
    else:
        assert selected == ['ese', 'simam', 'msca_lite', 'msca_dilated']
    settings[run] = {'config_path': str(path), 'config_sha256': sha(path),
                     'selected_backbone_class': model_class, 'source_filename': filename,
                     'saved_parameters': params, 'selected_aggregation_by_stage': selected}

report = {'status': 'AGGREGATION_PROJECTION_PATHS_STATICALLY_TRACED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'source_manifest_path': str(REMOTE / 'source_manifest.json'),
          'source_manifest_sha256': sha(REMOTE / 'source_manifest.json'),
          'sources': sources, 'settings': settings,
          'baseline_path': 'Concatenated HG features -> ConvBNAct 1x1 to C/2 -> ConvBNAct 1x1 to C.',
          'complete_path': 'Concatenated HG features -> ConvBNAct 1x1 directly to C -> stage-specific gate.',
          'scope': 'Eight saved configurations and the inspected source; source SHA matches the frozen manifest for the live pair. No certification of historical training-source bytes.',
          'static_audit_correction': 'The first static pass stopped because hgnetv2_silu.py is absent from the live frozen manifest. It is recorded as an inspected current-only source for N3; frozen matches remain mandatory for the live BASE/FULL sources.',
          'model_executions': 0, 'evaluator_reruns': 0, 'training_source_modified': False}
(HERE / 'aggregation_path_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'sources': len(sources), 'settings': len(settings),
                  'model_executions': 0}, ensure_ascii=False))
