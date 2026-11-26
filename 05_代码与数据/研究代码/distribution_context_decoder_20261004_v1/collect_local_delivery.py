"""CPU parameter inventory, immutable local delivery manifest and interface readiness."""
import ast
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch
import bootstrap
from bootstrap import ROOT
from factory import build_model, load_config, verify_snapshot
from run_candidate import implementation_hash

torch.set_num_threads(4)
counts = {}
for path in sorted((ROOT / 'configs').glob('*.yml')):
    model = build_model(load_config(path))
    counts[path.stem] = {'total_parameters': sum(p.numel() for p in model.parameters()),
                         'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                         'state_tensors': len(model.state_dict())}
with (ROOT / 'parameter_counts_actual_v2.json').open('x', encoding='utf8') as f:
    json.dump(counts, f, indent=2)
verify_snapshot()
for p in ROOT.glob('*.py'): ast.parse(p.read_text(encoding='utf8'))
tests = [ROOT / 'cpu_verification_v1.json', ROOT / 'cpu_integration_v4.json']
receipts = [json.loads(p.read_text(encoding='utf8')) for p in tests]
if not all(all(c['passed'] for c in r['checks']) for r in receipts):
    raise RuntimeError('CPU checks not passed')
if receipts[1]['implementation_sha256'] != implementation_hash():
    raise RuntimeError('Implementation changed after the final integration test')
rows = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc' or 'local_deps' in p.parts:
        continue
    rows.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
result = {'created_local': datetime.datetime.now().astimezone().isoformat(),
          'status': 'two_isolated_local_candidates_prepared_and_CPU_verified',
          'implementation_sha256': implementation_hash(), 'files': rows,
          'native_source_count': len(verify_snapshot()['files']), 'parameter_counts': counts,
          'local_environment': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'pycocotools', 'numpy', 'scipy', 'Pillow', 'PyYAML']},
          'CPU_checks_passed': sum(len(r['checks']) for r in receipts),
          'real_data_APs_result': None, 'real_GPU_capacity_result': None,
          'training_capacity_and96_loop_executed': False,
          'existing_C4_or_C8_O8_jobs_changed': False, 'shared_source_changed': False,
          'remaining': ['real batch8 ordinary/dense capacity on each arm, natural allocated peak >=10GiB',
                        'confirm active/source/optimizer parameter counts and native kernel compatibility on actual runtime',
                        'formal paired preparation/source lock and all-batch input/exposure auditing before launch',
                        'complete548val/all12/perclass/raw predictions/independent evaluator and archival acceptance',
                        'limited nearest-method novelty review before any paper claim']}
with (ROOT / 'local_delivery_manifest_v2.json').open('x', encoding='utf8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps({'CPU_checks': result['CPU_checks_passed'], 'parameter_counts': counts}))
