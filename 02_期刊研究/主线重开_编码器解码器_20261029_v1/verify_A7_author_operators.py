"""Decode public Git blobs and verify their exact pinned author tree entries."""
from pathlib import Path
import base64
import hashlib
import json
from datetime import datetime, timezone

root=Path(__file__).resolve().parent
metadata=root/'source_audit_v1'/'sparse_gate_A7'
sources={
 'SViT':{
  'repo':'https://github.com/uzh-rpg/svit',
  'files':{
   'selective_vit':'mmdet_custom/models/backbones/base/selective_vit.py',
   'transformer_layer':'mmdet_custom/models/backbones/base/TransformerEncoderLayer.py',
   'adapter':'mmdet_custom/models/backbones/selective_vit_adapter.py',
   'adapter_modules':'mmdet_custom/models/backbones/adapter_modules.py',
   'detector':'mmdet_custom/models/detectors/gumbel_two_stage.py',
   'config_tiny':'configs/mask_rcnn/svit-adapter-t-0.5x-ftune.py'}},
 'Salience':{
  'repo':'https://github.com/xiuqhou/Salience-DETR',
  'files':{
   'transformer':'models/bricks/salience_transformer.py',
   'detector':'models/detectors/salience_detr.py',
   'config_r50':'configs/salience_detr/salience_detr_resnet50_800_1333.py'}}}
receipts=[]
for name,spec in sources.items():
 commit=json.loads((metadata/(name+'_author_commit.json')).read_text(encoding='utf-8'))
 tree=json.loads((metadata/(name+'_author_tree.json')).read_text(encoding='utf-8'))
 assert not tree.get('truncated')
 assert tree['sha']==commit['commit']['tree']['sha']
 indexed={x['path']:x for x in tree['tree'] if x['type']=='blob'}
 directory=root/'comparison_sources_v1'/(name+'_operators_A7_v1')
 out=[]
 for label,path in spec['files'].items():
  raw=metadata/(name+'_'+label+'_blob.json')
  blob=json.loads(raw.read_text(encoding='utf-8'))
  assert blob['encoding']=='base64'
  content=base64.b64decode(blob['content'])
  gitsha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
  assert gitsha==blob['sha']==indexed[path]['sha'],path
  assert len(content)==blob['size']==indexed[path]['size'],path
  dest=directory/path
  dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(content)
  out.append({'author_path':path,'local_path':str(dest),'git_blob_sha':gitsha,
   'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),
   'url':spec['repo']+'/blob/'+commit['sha']+'/'+path})
 receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'ALL_PINNED_OPERATOR_BLOBS_VERIFIED',
  'repo':spec['repo'],'commit':commit['sha'],'files':out,
  'scope':'Selected source/config blobs only; no author imports, model construction, weights, training or forward execution.'}
 (directory/'source_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 receipts.append({'name':name,'commit':commit['sha'],'files':len(out),'directory':str(directory)})
print(json.dumps(receipts,ensure_ascii=False,indent=2))
