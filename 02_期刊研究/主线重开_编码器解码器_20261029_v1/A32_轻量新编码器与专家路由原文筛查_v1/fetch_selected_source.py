from pathlib import Path
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
FILES={
 'kwonseungchan/SPAR-Det':['README.md','mmdet/models/backbones/cahf.py','mmdet/models/roi_heads/bbox_heads/__init__.py','mmdet/models/roi_heads/bbox_heads/convfc_bbox_head_custom.py','mmdet/models/roi_heads/bbox_heads/convfc_bbox_head_custom_static9.py','mmdet/models/roi_heads/standard_roi_head.py'],
 'Intellindust-AI-Lab/DEIMv2':['README.md','engine/backbone/dinov3_adapter.py','engine/backbone/vit_tiny.py','configs/deimv2/deimv2_dinov3_s_coco.yml'],
 'Intellindust-AI-Lab/EdgeCrafter':['README.md','ecdetseg/engine/edgecrafter/ecvit.py','ecdetseg/engine/edgecrafter/hybrid_encoder.py','ecdetseg/engine/edgecrafter/decoder.py','ecdetseg/configs/ecdet/ecdet_s.yml'],
 'RT-DETRs/RT-DETRv4':['README.md','engine/rtv4/hybrid_encoder.py','engine/rtv4/dinov3_teacher.py']}
rows=[]
for repo,paths in FILES.items():
 m=D/'repo_metadata'/repo.replace('/','__')
 pin=json.loads((m/'receipt.json').read_text(encoding='utf-8'))['commit']
 tree={x['path']:x for x in json.loads((m/'tree.json').read_text(encoding='utf-8'))['tree'] if x['type']=='blob'}
 for path in paths:
  url=f'https://raw.githubusercontent.com/{repo}/{pin}/{path}'
  req=urllib.request.Request(url,headers={'User-Agent':'UAV-architecture-static-review'})
  with urllib.request.urlopen(req,timeout=30) as f:data=f.read()
  blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  assert blob==tree[path]['sha'],(repo,path,'Git blob mismatch')
  target=D/'source'/repo.replace('/','__')/path;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(data)
  rows.append(dict(repo=repo,commit=pin,path=path,url=url,local=str(target),bytes=len(data),git_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest(),imported_or_executed=False))
 print(json.dumps(dict(repo=repo,selected_files=len(paths),commit=pin)),flush=True)
(D/'selected_source_manifest.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='VERIFIED',blobs=len(rows),author_code_executed=False)),flush=True)
