"""Download exactly three Linux/Python3.9 wheels locally, verify official PyPI SHA256."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,ssl,json,hashlib,time
ROOT=Path(__file__).resolve().parent
JOBS=[('scikit-learn','1.6.1'),('joblib','1.4.2'),('threadpoolctl','3.6.0')]
def get(job):
    package,version=job;context=ssl.create_default_context()
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',context=context,timeout=35) as response:raw=response.read()
    metadata=json.loads(raw);(ROOT/(package+'_pypi.json')).write_bytes(raw)
    files=[r for r in metadata['urls'] if r['filename'].endswith('.whl')]
    if package=='scikit-learn':files=[r for r in files if 'cp39-cp39' in r['filename'] and 'manylinux_2_17_x86_64' in r['filename']]
    else:files=[r for r in files if 'py3-none-any' in r['filename']]
    assert len(files)==1,(package,[r['filename'] for r in files]);entry=files[0]
    path=ROOT/'wheels'/entry['filename'];path.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(entry['url'],context=context,timeout=60) as response:data=response.read(40000000)
    digest=hashlib.sha256(data).hexdigest();assert len(data)==entry['size'] and digest==entry['digests']['sha256']
    path.write_bytes(data)
    return dict(package=package,version=version,filename=entry['filename'],url=entry['url'],bytes=len(data),sha256=digest,
                requires_python=metadata['info']['requires_python'],requires_dist=metadata['info'].get('requires_dist'))
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(get,JOBS))
(ROOT/'wheel_manifest.json').write_text(json.dumps(dict(source='official PyPI metadata and distribution URLs',packages=rows),indent=2))
print(json.dumps([{k:r[k] for k in ['package','version','filename','bytes','sha256','requires_python']} for r in rows]))
