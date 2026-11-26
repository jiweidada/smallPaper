"""Verify restored accepted164 bytes and pin the immutable next-epoch anchor."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

ROOT = Path('/25085400120/mpsma_fair_20261103_v1/module_ablation_20261107_v1/no_msca_20261107_v1')
PREP = ROOT / 'recovery30715_preparation'


def digest(path):
    value = sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            value.update(block)
    return value.hexdigest()


assert (PREP / 'prior_segment_archive_manifest.json').is_file()
assert all(not (ROOT / name).exists() for name in ['launch.lock', 'supervisor.json', 'supervisor.log', 'training.log'])
manifest_path = ROOT / 'backup_gate/epoch0163/manifest.json'
assert digest(manifest_path) == '8bf322a46106ffc2113e65ddc4710ef5e280f43b58466be18866ebf2dae9e8a0'
manifest = json.loads(manifest_path.read_text())
files = []
for item in manifest['files']:
    path = (ROOT / item['source']).resolve(strict=True)
    assert path.is_relative_to(ROOT)
    assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], item['source']
    files.append({'name': item['name'], 'source': item['source'], 'path': item['source'],
        'sha256': item['sha256'], 'bytes': item['bytes']})
assert len(files) == 12
anchor = ROOT / 'recovery30715_anchor'
anchor.mkdir(exist_ok=False)
os.link(ROOT / 'run/next_epoch.pt', anchor / 'next_epoch.pt')
assert digest(anchor / 'next_epoch.pt') == '512dfe64c5b99479ffc2235054ee1bc1b2b5378ac776e7bead4d57395b3e545d'
result = {'status': 'ALL_12_ACCEPTED164_SOURCE_BYTES_RESTORED_AND_ANCHOR_PINNED',
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
    'accepted_epoch_index': 163, 'retained_rounds': 164, 'next_epoch_index': 164,
    'optimizer_step_attempts': 265188, 'manifest_sha256': digest(manifest_path),
    'anchor_sha256': digest(anchor / 'next_epoch.pt'),
    'prior_archive_manifest_sha256': digest(PREP / 'prior_segment_archive_manifest.json'),
    'restoration_verifier_source_sha256': digest(Path(__file__)),
    'training_started': False, 'model_forwards': 0, 'evaluator_runs': 0}
with (PREP / 'restored_source_records.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(result))
