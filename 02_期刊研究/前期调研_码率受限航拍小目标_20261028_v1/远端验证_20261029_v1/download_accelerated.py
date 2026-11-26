"""Public mirror transport, official PyPI SHA256. At most 8 bounded connections."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urlparse,unquote
import hashlib,json,math,subprocess,time,sys
ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'full_linux_download_plan.json').read_bytes())
folder=ROOT/'accelerated_wheels';parts_dir=ROOT/'download_parts'
folder.mkdir(exist_ok=True);parts_dir.mkdir(exist_ok=True)
items=[]
for package in plan['install']:
    source=package['download_info'];url=source['url'];parsed=urlparse(url)
    assert parsed.hostname=='files.pythonhosted.org' and parsed.scheme=='https'
    archive=source['archive_info'];digest=archive.get('hashes',{}).get('sha256')
    if not digest and archive.get('hash','').startswith('sha256='):digest=archive['hash'].split('=',1)[1]
    assert digest
    filename=unquote(Path(parsed.path).name);assert filename.endswith('.whl')
    item={'name':package['metadata']['name'],'version':package['metadata']['version'],'filename':filename,
          'official_url':url,'sha256':digest,'mirror_url':'https://mirrors.aliyun.com/pypi'+parsed.path}
    items.append(item)

def head(item):
    # Downloaded official metadata supplies the integrity; mirror HEAD only supplies the transfer length.
    import re
    for url in [item['mirror_url'],item['official_url']]:
        p=subprocess.run(['curl.exe','-I','-L','--fail','--retry','1','--connect-timeout','10','--max-time','40','--silent','--show-error',url],capture_output=True,timeout=50)
        sizes=re.findall(rb'(?im)^content-length:\s*(\d+)\s*$',p.stdout)
        if p.returncode==0 and sizes:
            item['bytes']=int(sizes[-1]);item['download_url']=url
            assert 0<item['bytes']<2*1024**3
            return item
    raise RuntimeError('No accessible transfer length '+item['filename'])

with ThreadPoolExecutor(max_workers=8) as pool:
    items=list(pool.map(head,items))
jobs=[]
for i,item in enumerate(items):
    output=folder/item['filename']
    if output.exists() and hashlib.file_digest(output.open('rb'),'sha256').hexdigest()==item['sha256']:
        item['status']='VERIFIED';continue
    # ~32MiB ranges keep each transfer bounded and prevent one slow connection dominating a multi-GB wheel.
    chunk=32*1024*1024
    for start in range(0,item['bytes'],chunk):
        end=min(start+chunk,item['bytes'])-1
        jobs.append((i,start,end,parts_dir/(item['filename']+'.'+str(start)+'.part')))
started=time.monotonic();completed_bytes=0;failures=[]
def fetch(job):
    i,start,end,path=job;expected=end-start+1
    if path.exists() and path.stat().st_size==expected:return job
    url=items[i]['download_url']
    if path.exists() and 0<path.stat().st_size<expected:
        url='https://pypi.tuna.tsinghua.edu.cn'+urlparse(items[i]['official_url']).path
    header=path.with_suffix('.headers')
    p=subprocess.run(['curl.exe','-L','--fail','--retry','1','--connect-timeout','15','--max-time','600',
                     '--silent','--show-error','--dump-header',str(header),'--range',str(start)+'-'+str(end),'--output',str(path),url],capture_output=True,timeout=620)
    if p.returncode or not path.exists() or path.stat().st_size!=expected:
        raise RuntimeError(items[i]['filename']+': range '+str(start)+' actual='+str(path.stat().st_size if path.exists() else -1)+' returncode='+str(p.returncode)+' '+p.stderr.decode(errors='replace')[-300:])
    return job
with ThreadPoolExecutor(max_workers=8) as pool:
    pending={pool.submit(fetch,j):j for j in jobs}
    for future in as_completed(pending):
        try:
            i,start,end,path=future.result();completed_bytes+=end-start+1
            progress={'completed_ranges_bytes':completed_bytes,'scheduled_ranges_bytes':sum(j[2]-j[1]+1 for j in jobs),
                      'elapsed_seconds':round(time.monotonic()-started,2),'last_file':items[i]['filename']}
            (ROOT/'accelerated_download_progress.json').write_text(json.dumps(progress),encoding='utf-8')
            print(json.dumps(progress),flush=True)
        except Exception as e:failures.append(str(e));print(json.dumps({'error':str(e)}),flush=True)
assert not failures,failures
for i,item in enumerate(items):
    if item.get('status')=='VERIFIED':continue
    dest=folder/item['filename'];temp=dest.with_suffix('.assembling')
    h=hashlib.sha256()
    with temp.open('wb') as out:
        for _,start,end,part in sorted((j for j in jobs if j[0]==i),key=lambda j:j[1]):
            with part.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):out.write(block);h.update(block)
    assert h.hexdigest()==item['sha256'],('Official SHA256 mismatch',item['filename'])
    assert temp.stat().st_size==item['bytes']
    temp.replace(dest);item['status']='VERIFIED'
manifest={'status':'VERIFIED_ALL','packages':items,'official_integrity_source':'PyPI report hashes',
          'transport':'Aliyun public PyPI mirror, range downloads; official SHA256 required','bytes':sum(x['bytes'] for x in items)}
(ROOT/'accelerated_download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'status':'VERIFIED_ALL','packages':len(items),'bytes':manifest['bytes']}),flush=True)
