import json
from pathlib import PurePosixPath
from update_round30 import OUT, read, replace_json, sha

snapshot = OUT / 'verified_snapshots/NO_MSCA_round240_verified_20261110'
manifest_path = snapshot / 'snapshot_manifest.json'
fixed = read(manifest_path)
assert sha(manifest_path) == 'a2c9dd2a4c7f47d7c134e4ad594e354fd841d04014982fc3d19c175d46a6e0f6'
assert fixed['NO_MSCA_independently_verified_rounds'] == 240 and fixed['per_epoch_csv_rows'] == 1200
assert len(fixed['files']) == 31
for entry in fixed['files']:
    relative = PurePosixPath(entry['file'].replace('\\', '/'))
    assert not relative.is_absolute() and '..' not in relative.parts
    current, old = OUT.joinpath(*relative.parts), snapshot.joinpath(*relative.parts)
    assert current.resolve().is_relative_to(OUT.resolve()) and old.resolve().is_relative_to(snapshot.resolve())
    assert current.stat().st_size == old.stat().st_size == entry['bytes']
    assert sha(current) == sha(old) == entry['sha256']
    assert current.read_bytes() == old.read_bytes()
assert sha(OUT / '交付清单.json') == '5d3bb00d1ed234372dd8acb5a394c4092f15afce8031fae67151b61dd1e60b1d'
assert sha(OUT / 'outputs/实验数据汇总_20261108/实验数据汇总.xlsx') == '2e736a20920492c2550648a97b26ad6a3094575aa8929129377478c4109a528c'
record = {'status':'CURRENT_DELIVERY_EQUALS_VERIFIED_ROUND240_SNAPSHOT',
          'verified_files_including_delivery_manifest':31, 'snapshot':str(snapshot),
          'snapshot_manifest':str(manifest_path), 'snapshot_manifest_sha256':sha(manifest_path),
          'delivery_manifest_sha256':sha(OUT / '交付清单.json')}
replace_json(OUT / '_build/prior_round240_verification.json', record)
print(json.dumps(record, ensure_ascii=True))
