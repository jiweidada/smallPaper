"""Fetch eight pinned author blobs for static operator comparison only."""
from pathlib import Path
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
SELECTION={'fcjian/TOOD':['README.md','configs/tood/tood_r50_fpn_1x_coco.py','mmdet/models/dense_heads/tood_head.py','mmdet/core/bbox/assigners/task_aligned_assigner.py'],'Megvii-BaseDetection/DeFCN':['README.md','playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf.aux/fcos.py','playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf.aux/net.py','playground/detection/coco/poto.res50.fpn.coco.800size.3x_ms.3dmf.aux/config.py']}
def main():
    rows=[]
    for repo,files in SELECTION.items():
        root=R/'repo_metadata'/repo.replace('/','__');pin=json.loads((root/'receipt.json').read_text());tree={x['path']:x for x in json.loads((root/'tree.json').read_text())['tree'] if x['type']=='blob'}
        for path in files:
            assert path in tree,(repo,path);url='https://raw.githubusercontent.com/'+repo+'/'+pin['commit']+'/'+path;req=urllib.request.Request(url,headers={'User-Agent':'UAV-nearest-method-audit'})
            with urllib.request.urlopen(req,timeout=25) as f:data=f.read()
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==tree[path]['sha']
            dest=R/'source'/repo.replace('/','__')/path;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(data)
            row={'repo':repo,'commit':pin['commit'],'path':path,'url':url,'local':str(dest),'bytes':len(data),'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest(),'imported_or_executed':False};rows.append(row);print(json.dumps({'repo':repo,'path':path,'blob_verified':True}),flush=True)
    with (R/'selected_source_manifest.json').open('x',encoding='utf-8') as f:json.dump({'rows':rows},f,ensure_ascii=False,indent=2);f.write('\n')
if __name__=='__main__':main()
