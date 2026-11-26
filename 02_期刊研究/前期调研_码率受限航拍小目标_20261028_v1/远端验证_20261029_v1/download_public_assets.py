from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parent
assets=json.loads((ROOT/'public_assets_to_download.json').read_bytes())
folder=ROOT/'public_assets';folder.mkdir(exist_ok=True)
def fetch(a):
    dest=folder/a['name']; started=time.monotonic()
    p=subprocess.run(['curl.exe','-L','--fail','--retry','0','--connect-timeout','20','--max-time','1200','--silent','--show-error','--output',str(dest),a['url']],capture_output=True,timeout=1210)
    r=dict(a,exit_code=p.returncode,seconds=round(time.monotonic()-started,3),local=str(dest))
    if p.returncode: r['error']=p.stderr.decode(errors='replace')[-500:];return r
    with dest.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    r['sha256']=digest;r['bytes']=dest.stat().st_size
    assert r['bytes']==a['size']
    if a.get('expected_digest'):assert a['expected_digest']=='sha256:'+digest
    r['status']='VERIFIED';return r
results=[]
with ThreadPoolExecutor(max_workers=3) as executor:
    for f in as_completed([executor.submit(fetch,a) for a in assets]):
        r=f.result();results.append(r);print(json.dumps(r),flush=True)
(ROOT/'public_asset_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
assert all(r.get('status')=='VERIFIED' for r in results)
