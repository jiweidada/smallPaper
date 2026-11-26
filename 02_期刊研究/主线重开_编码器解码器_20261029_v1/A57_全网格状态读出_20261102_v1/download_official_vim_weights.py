"""Download the author-linked ImageNet initialization locally before upload."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request

P=Path(__file__).resolve().parent


def main():
    receipt=json.loads((P/'Vim_source_receipt.json').read_text('utf-8'))
    meta=next(r for r in receipt['official_weight_files'] if r['path']=='vim_t_midclstok_76p1acc.pth')
    folder=P/'assets'
    folder.mkdir(exist_ok=True)
    target=folder/meta['path']
    assert not target.exists()
    url='https://huggingface.co/hustvl/Vim-tiny-midclstok/resolve/main/'+meta['path']
    h=hashlib.sha256()
    total=0
    start=time.monotonic()
    req=urllib.request.Request(url,headers={'User-Agent':'academic-local-asset-download'})
    with urllib.request.urlopen(req,timeout=30) as src,target.open('xb') as dst:
        while True:
            block=src.read(1<<20)
            if not block:break
            dst.write(block);h.update(block);total+=len(block)
    assert total==meta['size']==meta['lfs']['size']
    assert h.hexdigest()==meta['lfs']['oid']
    out={'status':'VERIFIED_OFFICIAL_LFS_BYTES','url':url,'local_path':str(target),
         'bytes':total,'sha256':h.hexdigest(),'seconds':time.monotonic()-start,
         'model_executed':False,'new_detector_forwards':0,'new_optimizer_updates':0}
    with (P/'Vim_weights_download_receipt.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
