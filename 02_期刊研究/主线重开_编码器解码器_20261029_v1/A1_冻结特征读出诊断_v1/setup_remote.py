"""Prepare isolated read-only asset links and verify frozen A1 bytes;no model import."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/readout_probe_v1')
assert ROOT == EXPECTED
BASE = ROOT.parent
assert not (ROOT/'receipts/extraction_started.json').exists()
manifest = json.loads((ROOT/'freeze_manifest.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest() == item['sha256']
assert (ROOT/'tinyformer_probe_loader.py').read_bytes() == (BASE/'tinyformer_probe_loader.py').read_bytes()
for name in ('source','assets'):
    destination = ROOT/name
    target = BASE/name
    assert target.resolve().is_relative_to(Path('/root/c10_region_restore_20261029_v1'))
    if destination.exists():
        assert destination.is_symlink() and destination.resolve() == target.resolve()
    else:
        destination.symlink_to(target,target_is_directory=True)
(ROOT/'receipts').mkdir(exist_ok=True)
(ROOT/'receipts/asset_checkpoint_check.json').write_bytes((BASE/'receipts/asset_checkpoint_check.json').read_bytes())
protocol = json.loads((ROOT/'protocol.json').read_text())
native = json.loads((BASE/'receipts/frozen_input_contract.json').read_text())
assert [r['file_name'] for r in protocol['images']] == native['file_names_in_existing_order'][:16]
budget = json.loads((BASE/'receipts/cumulative_forward_budget.json').read_text())
assert budget['attempted_full_model_forwards'] == 169 and budget['cap'] == 200
receipt = dict(status='A1_SETUP_FROZEN_FILES_VERIFIED',files=len(manifest['files']),
               loader_identical_to_A0=True,asset_links_remain_within_authorized_root=True,
               model_forwards=0,training_updates=0,
               freeze_manifest_sha256=hashlib.sha256((ROOT/'freeze_manifest.json').read_bytes()).hexdigest())
(ROOT/'receipts/setup_complete.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
