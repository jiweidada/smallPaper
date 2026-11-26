"""Mirror transport, with exact hashes independently read from official PyPI release pages."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,hashlib,json
ROOT=Path(__file__).resolve().parent
JOBS=[
 dict(package='scikit-learn',version='1.6.1',filename='scikit_learn-1.6.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',sha256='b8b7a3b86e411e4bce21186e1c180d792f3d99223dcfa3b4f597ecc92fa1a422',blake='7935b1d2188967c3204c78fa79c9263668cf1b98060e8e58d1a730fe5b2317bb'),
 dict(package='joblib',version='1.4.2',filename='joblib-1.4.2-py3-none-any.whl',sha256='06d478d5674cbc267e7496a410ee875abd68e4340feff4490bcb7afb88060ae6',blake='9129df4b9b42f2be0b623cbd5e2140cafcaa2bef0759a00b7b70104dcfe2fb51'),
 dict(package='threadpoolctl',version='3.6.0',filename='threadpoolctl-3.6.0-py3-none-any.whl',sha256='43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb',blake='32d5f9a850d79b0851d1d4ef6456097579a9005b31fea68726a4ae5f2d82ddd9')]
def get(row):
    row=row.copy();d=row['blake'];path='/packages/'+d[:2]+'/'+d[2:4]+'/'+d[4:]+'/'+row['filename']
    row['official_url']='https://files.pythonhosted.org'+path;row['metadata_source']='https://pypi.org/project/'+row['package']+'/'+row['version']+'/'
    target=ROOT/'wheels'/row['filename'];target.parent.mkdir(exist_ok=True);attempts=[]
    for prefix in ['https://mirrors.aliyun.com/pypi','https://pypi.tuna.tsinghua.edu.cn','https://files.pythonhosted.org']:
        url=prefix+path;temp=target.with_suffix('.download')
        p=subprocess.run([r'C:\Windows\System32\curl.exe','-L','--fail','--silent','--show-error','--max-time','90',url,'--output',str(temp)],capture_output=True,timeout=100)
        digest=hashlib.sha256(temp.read_bytes()).hexdigest() if temp.exists() else None
        attempts.append(dict(url=url,returncode=p.returncode,sha256=digest,error=p.stderr.decode(errors='replace')[-300:]))
        if p.returncode==0 and digest==row['sha256']:
            temp.replace(target);row.update(status='VERIFIED_OFFICIAL_PYPI_SHA256',bytes=target.stat().st_size,transport_url=url,attempts=attempts);return row
    raise RuntimeError(attempts)
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(get,JOBS))
(ROOT/'wheel_manifest.json').write_text(json.dumps(dict(source='official PyPI release HTML file hashes; mirror only transports bytes',packages=rows),indent=2))
print(json.dumps([{k:r[k] for k in ['package','version','status','bytes']} for r in rows]))
