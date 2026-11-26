from pathlib import Path,PurePosixPath
import hashlib
import json
import tarfile
root=Path(__file__).resolve().parent
asset=json.loads((root/'asset_selection.json').read_text(encoding='utf-8'))
archive=root/'prediction_blind_val_sources.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==asset['archive_sha256']
expected={}
for c in asset['cases']:
    expected['images/'+c['file']]=c['image_sha256']
    expected['annotations/'+Path(c['file']).stem+'.txt']=c['annotation_sha256']
inputs=root/'inputs'
assert not inputs.exists()
inputs.mkdir()
with tarfile.open(archive) as tar:
    assert {m.name for m in tar.getmembers()}==set(expected)
    for m in tar.getmembers():
        rel=PurePosixPath(m.name)
        assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
        data=tar.extractfile(m).read()
        assert hashlib.sha256(data).hexdigest()==expected[m.name]
        dest=inputs.joinpath(*rel.parts);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(data)
print(json.dumps({'status':'PASS_ALL24_SOURCE_ASSET_HASHES','files':len(expected),'model_forwards':0}))
