"""Local-first download of official source archives and the official COCO warm start."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
import urllib.request

P=Path(__file__).resolve().parent
OUT=P/'assets_local_v1'

def request(url):
    return urllib.request.Request(url,headers={'User-Agent':'Academic-experiment-asset-download/1.0'})

def metadata(url):
    with urllib.request.urlopen(request(url),timeout=25) as r:return r.read()

def download(url,name,expected=None):
    path=OUT/name;partial=OUT/(name+'.partial')
    assert not path.exists() and not partial.exists()
    start=time.monotonic();n=0;h=hashlib.sha256()
    try:
        with urllib.request.urlopen(request(url),timeout=30) as r,partial.open('xb') as f:
            actual=r.url
            while True:
                raw=r.read(1<<20)
                if not raw:break
                f.write(raw);h.update(raw);n+=len(raw)
                assert n<=768*1024*1024
                if time.monotonic()-start>240:raise TimeoutError('Bounded download time cap')
        if expected is not None:assert n==expected,(n,expected)
        assert path.resolve().is_relative_to(OUT.resolve()) and partial.resolve().is_relative_to(OUT.resolve())
        partial.rename(path)
        return dict(status='DOWNLOADED',source_url=url,final_url=actual,path=str(path),bytes=n,sha256=h.hexdigest())
    except Exception as e:
        return dict(status='FAILED',source_url=url,partial_path=str(partial) if partial.exists() else None,
                    bytes=n,error=type(e).__name__+': '+str(e))

def main():
    OUT.mkdir(exist_ok=False)
    release_raw=metadata('https://api.github.com/repos/ShoufaChen/DiffusionDet/releases/tags/v0.1')
    with (OUT/'DiffusionDet_release_v0.1.json').open('xb') as f:f.write(release_raw)
    release=json.loads(release_raw)
    asset=next(a for a in release['assets'] if a['name']=='diffdet_coco_res50_300boxes.pth')
    repo_raw=metadata('https://api.github.com/repos/facebookresearch/detectron2')
    with (OUT/'detectron2_repo.json').open('xb') as f:f.write(repo_raw)
    repo=json.loads(repo_raw)
    commit_raw=metadata('https://api.github.com/repos/facebookresearch/detectron2/commits/'+repo['default_branch'])
    with (OUT/'detectron2_commit.json').open('xb') as f:f.write(commit_raw)
    pin=json.loads(commit_raw)['sha']
    items=[(asset['browser_download_url'],'official_diffdet_coco_res50_300boxes.pth',asset['size']),
           ('https://codeload.github.com/facebookresearch/detectron2/tar.gz/'+pin,'detectron2_'+pin+'.tar.gz',None),
           ('https://codeload.github.com/ShoufaChen/DiffusionDet/tar.gz/1efb36d4b0b721f2d509bc72656235a94d73ab02','DiffusionDet_1efb36d4.tar.gz',None)]
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(lambda a:download(*a),items))
    value=dict(detectron2_pin=pin,diffusiondet_pin='1efb36d4b0b721f2d509bc72656235a94d73ab02',
        sources=rows,model_code_executed=False,checkpoints_loaded=0,model_forwards=0)
    with (P/'framework_asset_download_v1.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
