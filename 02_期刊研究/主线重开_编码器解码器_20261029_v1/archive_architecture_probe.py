"""Archive the bounded diagnostic, excluding already-local model/source assets."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
assert json.loads((ROOT / 'receipts/study_complete.json').read_text())['status'] == 'STUDY100_TRACE_COMPLETE'
assert json.loads((ROOT / 'sampling_observation_v1/completed.json').read_text())['status'] == 'SAMPLING32_OBSERVATION_AND_EQUIVALENCE_PASS'
assert json.loads((ROOT / 'receipts/cumulative_forward_budget.json').read_text())['attempted_full_model_forwards'] == 169
backup = ROOT / 'backup'
backup.mkdir(exist_ok=True)
files = [p for folder in ['receipts', 'traces', 'analysis', 'sampling_observation_v1']
         for p in (ROOT / folder).rglob('*') if p.is_file()]
files += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py', '.json']]
files = sorted(set(files))
rows = []
for p in files:
    assert not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve())
    rows.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
manifest = backup / 'file_manifest.json'
assert not manifest.exists()
manifest.write_text(json.dumps(rows, indent=2))
archive = backup / 'A0_A0b_bounded_diagnostics.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    tf.add(manifest, arcname='backup/file_manifest.json', recursive=False)
    for p in files:
        tf.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
result = {'status': 'ARCHIVED', 'files': len(rows), 'archive': str(archive),
          'archive_bytes': archive.stat().st_size, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'source_and_weight': 'Separate verified local official assets; intentionally not duplicated in this archive',
          'full_model_forwards': 169, 'training_updates': 0}
(backup / 'archive_receipt.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
