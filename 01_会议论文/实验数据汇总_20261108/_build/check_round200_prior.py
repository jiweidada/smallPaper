import json
from pathlib import PurePosixPath
from update_round30 import OUT, read, replace_json, sha

snapshot = OUT / 'verified_snapshots/NO_MSCA_round200_verified_20261110'
manifest_path = snapshot / 'snapshot_manifest.json'
fixed = read(manifest_path)
assert fixed['NO_MSCA_independently_verified_rounds'] == 200 and fixed['per_epoch_csv_rows'] == 1160
assert len(fixed['files']) == 26
for entry in fixed['files']:
    relative = PurePosixPath(entry['file'].replace('\\', '/'))
    assert not relative.is_absolute() and '..' not in relative.parts
    current, old = OUT.joinpath(*relative.parts), snapshot.joinpath(*relative.parts)
    assert current.resolve().is_relative_to(OUT.resolve()) and old.resolve().is_relative_to(snapshot.resolve())
    assert current.stat().st_size == old.stat().st_size == entry['bytes']
    assert sha(current) == sha(old) == entry['sha256']
    assert current.read_bytes() == old.read_bytes()
assert sha(OUT / '交付清单.json') == '99fa19019d0d989da06cea9606d31f6a2413665dec0e32848151c660149d9b74'
assert sha(OUT / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx') == 'f8a07aae1245e6f6708f5b8090be8f34aa0e41eeb91caf98ceb82c3de3eb1d1a'
record = {'status':'CURRENT_DELIVERY_EQUALS_VERIFIED_ROUND200_SNAPSHOT',
          'verified_files_including_delivery_manifest':26, 'snapshot':str(snapshot),
          'snapshot_manifest':str(manifest_path), 'snapshot_manifest_sha256':sha(manifest_path),
          'delivery_manifest_sha256':sha(OUT / '交付清单.json')}
replace_json(OUT / '_build/prior_round200_verification.json', record)
print(json.dumps(record, ensure_ascii=True))
