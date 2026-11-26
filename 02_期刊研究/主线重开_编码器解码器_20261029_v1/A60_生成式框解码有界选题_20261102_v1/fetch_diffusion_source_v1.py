"""Read selected original DiffusionDet source and primary papers, without execution."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from fetch_primary_v1 import P, ARCHIVE, fetch, save

def main():
    dest=P/'DiffusionDet_primary_source_v1';dest.mkdir(exist_ok=False)
    api='https://api.github.com/repos/ShoufaChen/DiffusionDet'
    raw,_=fetch(api);repo=json.loads(raw);save(dest/'repo.json',raw)
    raw,_=fetch(api+'/commits/'+repo['default_branch']);commit=json.loads(raw);save(dest/'commit.json',raw)
    pin=commit['sha']
    raw,_=fetch(api+'/git/trees/'+pin+'?recursive=1');tree=json.loads(raw);save(dest/'tree.json',raw)
    assert not tree.get('truncated',False)
    blobs={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    wanted=['README.md','GETTING_STARTED.md','diffusiondet/detector.py','diffusiondet/head.py',
            'diffusiondet/loss.py','diffusiondet/config.py','diffusiondet/util/box_ops.py',
            'configs/diffdet.coco.res50.yaml','configs/Base-DiffusionDet.yaml']
    def one(name):
        assert name in blobs,name
        raw,url=fetch('https://raw.githubusercontent.com/ShoufaChen/DiffusionDet/'+pin+'/'+name)
        git=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert git==blobs[name]['sha']
        path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
        r=save(path,raw);r.update(repo_path=name,url=url,git_blob_sha1=git)
        return r
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,wanted))
    save(dest/'selected_source_receipt_v1.json',json.dumps(dict(pin=pin,records=records,
        code_imported_or_executed=False),indent=2).encode())
    papers=[('DiffusionDet_ICCV2023','https://arxiv.org/pdf/2211.09788'),
            ('SiamDiff_RemoteSensing2026','https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-00018/article_deploy/remotesensing-18-00018.pdf')]
    results=[]
    for name,url in papers:
        try:
            raw,actual=fetch(url);assert raw.startswith(b'%PDF-')
            r=save(ARCHIVE/('A60_'+name+'.pdf'),raw)
            r.update(title_locator=name,source_url=url,final_url=actual,read_scope='Downloaded only; not yet read')
        except Exception as e:r=dict(title_locator=name,source_url=url,status='FAILED',error=type(e).__name__+': '+str(e))
        save(ARCHIVE/('A60_'+name+'.source.json'),json.dumps(r,indent=2).encode())
        results.append(r)
    save(P/'diffusion_source_fetch_receipt_v1.json',json.dumps(dict(pin=pin,selected_source_files=len(records),papers=results),indent=2).encode())
    print(json.dumps(dict(pin=pin,source_files=len(records),papers=results)))

if __name__=='__main__':main()
