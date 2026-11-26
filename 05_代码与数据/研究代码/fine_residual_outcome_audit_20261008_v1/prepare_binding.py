"""Pin the existing stage analysis; no experiment is run."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
stage = root.parent/'fine_residual_stage_analysis_20261008_v1'
expected = {
    '01_交付回执与SHA.json': '7515f1bfce3182d44aea5fa2c0dace6cbe3018298fccd04d90286abf8325b392',
    'analyze_stages.py': '817bbccb0661ae5ecdcb765cf24f2f425e08d71ed1507d6e941701cb419ab27d',
}
for name, digest in expected.items():
    assert hashlib.sha256((stage/name).read_bytes()).hexdigest() == digest
expected['experiment_binding.json'] = hashlib.sha256((stage/'experiment_binding.json').read_bytes()).hexdigest()
with (root/'stage_binding.json').open('x', encoding='utf-8') as stream:
    json.dump({'directory': stage.name, 'files_sha256': expected}, stream, ensure_ascii=False, indent=2)
print('Bound existing stage analysis and its v5 implementation binding')
