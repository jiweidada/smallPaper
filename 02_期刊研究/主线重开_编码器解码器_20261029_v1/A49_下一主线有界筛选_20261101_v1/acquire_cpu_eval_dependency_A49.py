"""Acquire the matching official PyPI wheel into an isolated workspace directory."""
import hashlib,json,stat,zipfile
from pathlib import Path,PurePosixPath
import requests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'postprocess_ignore_strata'
receipt=OUT/'dependency_acquisition_v1.json'
assert not receipt.exists()
metadata_url='https://pypi.org/pypi/pycocotools/2.0.11/json'
response=requests.get(metadata_url,timeout=(8,22));response.raise_for_status();metadata=response.json()
with (OUT/'pycocotools2_0_11_PyPI_metadata.json').open('x',encoding='utf-8') as f:json.dump(metadata,f,ensure_ascii=False,indent=2)
name='pycocotools-2.0.11-cp312-abi3-win_amd64.whl'
row=next(x for x in metadata['urls'] if x['filename']==name)
dest=OUT/'verified_wheels';dest.mkdir(exist_ok=False)
wheel=dest/name
res=requests.get(row['url'],timeout=(8,22));res.raise_for_status()
assert hashlib.sha256(res.content).hexdigest()==row['digests']['sha256']
assert len(res.content)==row['size']
with wheel.open('xb') as f:f.write(res.content)
deps=OUT/'deps';deps.mkdir(exist_ok=False)
with zipfile.ZipFile(wheel) as z:
    for info in z.infolist():
        rel=PurePosixPath(info.filename)
        assert not rel.is_absolute() and '..' not in rel.parts and ':' not in info.filename and '\\' not in info.filename
        assert rel.parts[0] in ['pycocotools','pycocotools-2.0.11.dist-info']
        assert not stat.S_ISLNK(info.external_attr>>16)
        assert (deps/Path(*rel.parts)).resolve().is_relative_to(deps.resolve())
    z.extractall(deps)
record={'status':'OFFICIAL_WHEEL_HASH_VERIFIED_ISOLATED_EXTRACTION','version':'2.0.11',
        'metadata_url':metadata_url,'url':row['url'],'wheel':str(wheel),'sha256':row['digests']['sha256'],
        'bytes':row['size'],'deps_path':str(deps),'global_packages_modified':False,
        'model_forwards':0,'optimizer_updates':0}
with receipt.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps(record,ensure_ascii=False))
