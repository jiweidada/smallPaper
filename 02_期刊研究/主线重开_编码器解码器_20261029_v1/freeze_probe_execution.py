from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent
pairs = [('tinyformer_probe_loader.py', 'tinyformer_probe_loader.py'),
         ('tinyformer_stage_probe.py', 'tinyformer_stage_probe.py'),
         ('diagnostic100_asset_manifest.json', 'diagnostic100_asset_manifest.json'),
         ('06_TinyFormer分阶段最小诊断协议.json', 'parent_protocol.json'),
         ('08_A0实施细则_执行前冻结.json', 'implementation_details.json'),
         ('official_assets_v1/probe_package_manifest.json', 'assets/probe_package_manifest.json')]
rows = [{'local_name': a, 'remote_name': b, 'bytes': (ROOT / a).stat().st_size,
         'sha256': hashlib.sha256((ROOT / a).read_bytes()).hexdigest()} for a, b in pairs]
out = ROOT / 'execution_freeze_manifest.json'
assert not out.exists(), 'Keep freeze immutable; do not replace after model execution'
out.write_text(json.dumps({'recorded_utc': datetime.now(timezone.utc).isoformat(),
                           'state': 'FROZEN_BEFORE_ANY_FULL_MODEL_FORWARD',
                           'planned_forwards': 129, 'hard_cap': 200, 'training_updates': 0,
                           'files': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'frozen_files': len(rows), 'manifest_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
