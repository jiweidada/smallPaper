from pathlib import Path
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
pin=json.loads((D/'repo_metadata/BUTR/receipt.json').read_text(encoding='utf-8'))
tree=json.loads((D/'repo_metadata/BUTR/tree.json').read_text(encoding='utf-8'))
blobs={x['path']:x for x in tree['tree'] if x['type']=='blob'}
paths=['README.md','src/zoo/models/budget.py','src/zoo/models/butr_decoder.py','src/zoo/models/butr.py',
       'configs/models/BUTR-S-VisDrone.yml','configs/models/include/butr_hgnetv2.yml',
       'tools/coco/budget_adapter.py','tests/test_butr.py']
rows=[]
for path in paths:
    url='https://raw.githubusercontent.com/zhangyi-magic/BUTR/'+pin['commit']+'/'+path
    req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-primary-source-review'})
    with urllib.request.urlopen(req,timeout=30) as f:data=f.read()
    git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert git==blobs[path]['sha']
    target=D/'source/BUTR'/path;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(data)
    rows.append(dict(path=path,local=str(target),git_blob_sha=git,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
with (D/'selected_butr_source_verified.json').open('x',encoding='utf-8') as f:
    json.dump(dict(repo='zhangyi-magic/BUTR',commit=pin['commit'],status='ALL8_SELECTED_GIT_BLOBS_VERIFIED_NOT_EXECUTED',files=rows),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(commit=pin['commit'],selected_verified_blobs=len(rows))))
