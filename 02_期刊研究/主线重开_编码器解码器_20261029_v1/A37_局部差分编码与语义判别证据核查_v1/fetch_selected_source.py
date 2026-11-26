from pathlib import Path
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
pin=json.loads((D/'repo_metadata/PiDiViT/receipt.json').read_text(encoding='utf-8'))
tree=json.loads((D/'repo_metadata/PiDiViT/tree.json').read_text(encoding='utf-8'))
blobs={x['path']:x for x in tree['tree'] if x['type']=='blob'}
paths=['README.md','configs/few-shot/vitl_shot10.yaml','configs/few-shot/vitl_shot30.yaml','lib/config.py',
       'detectron2/modeling/meta_arch/__init__.py','detectron2/modeling/meta_arch/devit.py','detectron2/modeling/meta_arch/devit_update.py',
       'lib/regionprop.py','lib/regionprop_update.py','lib/ops.py','lib/ops_theta.py','lib/convert_pidinet.py',
       'lib/prototype_learner.py','detectron2/modeling/backbone/vit.py']
rows=[]
for path in paths:
    url='https://raw.githubusercontent.com/Seaz9/PiDiViT/'+pin['commit']+'/'+path
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review'})
    with urllib.request.urlopen(req,timeout=30) as f:raw=f.read()
    h=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();assert h==blobs[path]['sha']
    target=D/'source/PiDiViT'/path;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(raw)
    rows.append(dict(path=path,local=str(target),git_blob_sha=h,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
with (D/'selected_source_verification.json').open('x',encoding='utf-8') as f:
    json.dump(dict(status='ALL14_SELECTED_AUTHOR_GIT_BLOBS_VERIFIED_NOT_EXECUTED',repo='Seaz9/PiDiViT',commit=pin['commit'],files=rows),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(commit=pin['commit'],verified_blobs=len(rows))))
