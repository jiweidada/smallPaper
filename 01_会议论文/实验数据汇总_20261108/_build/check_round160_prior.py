import json
from pathlib import Path, PurePosixPath
from update_round30 import OUT, read, replace_json, sha

snapshot = OUT / 'verified_snapshots/NO_MSCA_round160_verified_20261109'
manifest_path = snapshot / 'snapshot_manifest.json'
fixed = read(manifest_path)
assert fixed['NO_MSCA_independently_verified_rounds'] == 160 and fixed['per_epoch_csv_rows'] == 1120
assert len(fixed['files']) == 19
for entry in fixed['files']:
    relative = PurePosixPath(entry['file'].replace('\\', '/'))
    assert not relative.is_absolute() and '..' not in relative.parts
    current, old = OUT.joinpath(*relative.parts), snapshot.joinpath(*relative.parts)
    assert current.resolve().is_relative_to(OUT.resolve()) and old.resolve().is_relative_to(snapshot.resolve())
    assert current.stat().st_size == old.stat().st_size == entry['bytes']
    assert sha(current) == sha(old) == entry['sha256']
    assert current.read_bytes() == old.read_bytes()
record = {'status': 'CURRENT_DELIVERY_EQUALS_VERIFIED_ROUND160_SNAPSHOT',
          'verified_files_including_delivery_manifest': 19, 'snapshot': str(snapshot),
          'snapshot_manifest': str(manifest_path), 'snapshot_manifest_sha256': sha(manifest_path),
          'delivery_manifest_sha256': sha(OUT / '交付清单.json')}
replace_json(OUT / '_build/prior_round160_verification.json', record)
print(json.dumps(record, ensure_ascii=True))
