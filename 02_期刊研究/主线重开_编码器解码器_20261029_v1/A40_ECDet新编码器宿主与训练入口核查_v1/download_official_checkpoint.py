from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,urllib.request
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;B=D.parent
source=B/'A39_语义诊断覆盖调整与分组验证_v1'/'next_compact_ViT_host_DRAFT.json'
plan=json.loads(source.read_text(encoding='utf-8'));asset=plan['asset']
assert asset['status']=='PRIMARY_RELEASE_ASSET_METADATA_VERIFIED_NOT_DOWNLOADED' and asset['bytes']==39938420
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),phase='A40_asset_only',source_draft=str(source),
    source_draft_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_commit=plan['source_commit'],
    original_public_url=asset['url'],release_id=asset['release_id'],asset_id=asset['asset_id'],api_digest=asset.get('api_digest'),
    model_instantiated=False,checkpoint_unpickled=False,new_model_calls=0,new_optimizer_updates=0)
try:
    request=urllib.request.Request(asset['url'],headers={'User-Agent':'UAV-new-encoder-host-asset-audit'})
    with urllib.request.urlopen(request,timeout=30) as response:raw=response.read(asset['bytes']+1)
    assert len(raw)==asset['bytes'],len(raw)
    digest=hashlib.sha256(raw).hexdigest()
    if asset.get('api_digest'):assert asset['api_digest']=='sha256:'+digest
    target=D/'ecdet_s_coco_official.pth'
    with target.open('xb') as output:output.write(raw)
    receipt.update(status='OFFICIAL_COCO_CHECKPOINT_DOWNLOADED_LOCALLY_NOT_LOADED',file=str(target),bytes=len(raw),sha256=digest)
except Exception as error:receipt.update(status='DOWNLOAD_UNAVAILABLE_NO_MODEL_CALLS',error=repr(error))
with (D/'checkpoint_download_receipt.json').open('x',encoding='utf-8') as output:json.dump(receipt,output,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k in ['status','bytes','sha256','error','new_model_calls']},ensure_ascii=False))
