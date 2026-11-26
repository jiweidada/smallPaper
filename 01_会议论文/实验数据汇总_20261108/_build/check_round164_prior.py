import json
from pathlib import PurePosixPath
from update_round30 import OUT, read, replace_json, sha

snapshot = OUT / 'verified_snapshots/NO_MSCA_round164_verified_20261109'
manifest_path = snapshot / 'snapshot_manifest.json'
fixed = read(manifest_path)
assert fixed['NO_MSCA_independently_verified_rounds'] == 164 and fixed['per_epoch_csv_rows'] == 1124
assert len(fixed['files']) == 20
for entry in fixed['files']:
    relative = PurePosixPath(entry['file'].replace('\\', '/'))
    assert not relative.is_absolute() and '..' not in relative.parts
    current, old = OUT.joinpath(*relative.parts), snapshot.joinpath(*relative.parts)
    assert current.resolve().is_relative_to(OUT.resolve()) and old.resolve().is_relative_to(snapshot.resolve())
    assert current.stat().st_size == old.stat().st_size == entry['bytes']
    assert sha(current) == sha(old) == entry['sha256']
    assert current.read_bytes() == old.read_bytes()
assert sha(OUT / '交付清单.json') == 'a3420c420b0a185403765120cfeb129a1ea095b7cd49bf4741a88d78dedca7cf'
record = {'status':'CURRENT_DELIVERY_EQUALS_VERIFIED_ROUND164_SNAPSHOT',
          'verified_files_including_delivery_manifest':20, 'snapshot':str(snapshot),
          'snapshot_manifest':str(manifest_path), 'snapshot_manifest_sha256':sha(manifest_path),
          'delivery_manifest_sha256':sha(OUT / '交付清单.json')}
replace_json(OUT / '_build/prior_round164_verification.json', record)
print(json.dumps(record, ensure_ascii=True))
