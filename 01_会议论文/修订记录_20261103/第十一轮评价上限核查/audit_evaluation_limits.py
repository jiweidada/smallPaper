"""Trace class-aware evaluator limits and the separate postprocessor output setting."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import ast
import json

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

captures = []
for host, name in [('BASE', 'primary_eval_limit_sources_v2.json'), ('FULL', 'secondary_eval_limit_sources_v2.json')]:
    path = REMOTE / name
    wrapper = read(path)
    assert wrapper['exit_code'] == 0
    capture = json.loads(wrapper['stdout'])
    assert capture['version'] == '1.8.0'
    assert all(capture[k] == 0 for k in ['model_imports', 'model_executions', 'evaluator_imports', 'evaluator_executions'])
    captures.append(capture)
assert [(s['path'], s['sha256']) for s in captures[0]['sources']] == [(s['path'], s['sha256']) for s in captures[1]['sources']]
assert len(captures[0]['sources']) == 4
manifest_path = REMOTE / 'remote_evidence/source_manifest.json'
manifest = {s['snapshot']: s['sha256'] for s in read(manifest_path)}
archive = HERE / 'sources'
archive.mkdir(exist_ok=True)
archived = []
for item in captures[0]['sources']:
    data = item['text'].encode('utf-8')
    if sha256(data).hexdigest() != item['sha256']:
        data = b'\xef\xbb\xbf' + data
    assert sha256(data).hexdigest() == item['sha256']
    name = Path(item['path']).name
    path = archive / name
    assert not path.exists()
    path.write_bytes(data)
    if '/source_snapshot/' in item['path']:
        frozen_name = 'source_snapshot/' + item['path'].split('/source_snapshot/', 1)[1]
        assert manifest[frozen_name] == item['sha256']
    tree = ast.parse(item['text'])
    functions = [{'name': n.name, 'line_start_1based': n.lineno, 'line_end_1based': n.end_lineno}
                 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name in ['computeIoU', '_prepare', 'evaluate', 'forward', '_summarize']]
    archived.append({'path': str(path), 'sha256': sha(path), 'remote_path': item['path'],
                     'functions': functions})

base = next(s['text'] for s in captures[0]['sources'] if s['path'].endswith('/core/cocoeval.py'))
fast = next(s['text'] for s in captures[0]['sources'] if s['path'].endswith('/core/faster_eval_api.py'))
wrapper = next(s['text'] for s in captures[0]['sources'] if s['path'].endswith('/dataset/coco_eval.py'))
post = next(s['text'] for s in captures[0]['sources'] if s['path'].endswith('/deim/postprocessor.py'))
assert 'self.useCats = 1' in base
assert 'get_instances([imgId], [catId] if p.useCats else p.catIds' in base
assert 'dt = dt[0 : p.maxDets[-1]]' in base
assert 'all_iou_pairs = list(itertools.product(p.imgIds, catIds))' in fast
assert '.useCats =' not in wrapper and 'COCOeval_faster(' in wrapper
assert 'scores.flatten(1), self.num_top_queries' in post

settings = {}
for name in ['N0', 'N1', 'N2', 'N3', 'N4', 'S1', 'BASE', 'FULL']:
    path = REMOTE / 'remote_evidence/execution_configs' / (name + '.yml') if name in ['BASE', 'FULL'] else REC / 'source_snapshot' / name / 'args.json'
    value = read(path)
    cfg = value.get('yaml_cfg', value)
    assert cfg['postprocessor'] == 'PostProcessor'
    assert cfg.get('num_top_queries', cfg['PostProcessor'].get('num_top_queries', 300)) == 300
    assert cfg.get('use_focal_loss', True) is True
    settings[name] = {'path': str(path), 'sha256': sha(path), 'postprocessor': cfg['postprocessor'],
                      'num_top_queries': 300, 'use_focal_loss': True}

local_base = PAPER / '本地CPU核验_20261103/.runtime/site-packages/faster_coco_eval/core/cocoeval.py'
local_text = local_base.read_text(encoding='utf-8-sig')
assert 'get_instances([imgId], [catId] if p.useCats else p.catIds' in local_text
assert 'dt = dt[0 : p.maxDets[-1]]' in local_text
saved_params = REC / 'dataset_eval_audit/saved_eval_metadata.json'
assert '"useCats": 1' in saved_params.read_text(encoding='utf-8')
report = {'status': 'CLASS_AWARE_MAXDETS_AND_GLOBAL_POSTPROCESS_OUTPUT_DISTINGUISHED',
          'checked_local_utc': datetime.now(timezone.utc).isoformat(),
          'remote_sources_identical_on_both_hosts': True, 'remote_package_version': '1.8.0',
          'remote_source_capture_files': [{'path': str(REMOTE / n), 'sha256': sha(REMOTE / n)}
              for n in ['primary_eval_limit_sources_v2.json', 'secondary_eval_limit_sources_v2.json']],
          'archived_sources': archived, 'settings': settings,
          'cpu_reevaluation_package_version': '1.6.5',
          'cpu_source': {'path': str(local_base), 'sha256': sha(local_base)},
          'saved_historical_evaluator_parameters': {'path': str(saved_params), 'sha256': sha(saved_params)},
          'postprocessor_output': 'Configured global top-300 predictions per image across categories.',
          'evaluator_limit': 'With useCats=1, detections are grouped by image and category, ranked by score, and truncated at maxDets for that group. AR@100 uses the maxDets=100 setting.',
          'source_reading_scope': 'Python preparation, category grouping, IoU truncation, C++ call arguments, and frozen project wrapper/postprocessor. Native C++ implementation was not separately inspected or executed.',
          'historical_runtime_version_certified': False,
          'model_executions': 0, 'evaluator_executions': 0, 'training_settings_changed': False,
          'scope': 'Static protocol clarification. Keeps current server evaluation (1.8.0), prior CPU reevaluation (1.6.5), and uncertified historical runtime versions distinct. Does not change any metric or validate unfinished training.'}
(HERE / 'evaluation_limit_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'server_version': '1.8.0', 'cpu_version': '1.6.5',
                  'source_files': len(archived), 'settings': len(settings), 'model_executions': 0}))
