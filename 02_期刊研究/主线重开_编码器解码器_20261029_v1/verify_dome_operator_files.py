"""Verify four fixed public Dome-DETR files without executing the project."""
from pathlib import Path
from datetime import datetime,timezone
import base64
import hashlib
import json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'comparison_sources_v1/Dome_DETR_operators_v1'

def main():
    meta=json.loads((ROOT/'source_audit_v1/Dome_author_commit.json').read_text(encoding='utf-8'))
    tree=json.loads((ROOT/'source_audit_v1/Dome_author_tree.json').read_text(encoding='utf-8'))
    expected={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    paths=['src/zoo/dome/defe.py','src/zoo/dome/dome_criterion.py','src/zoo/dome/hybrid_encoder.py','configs/dome/Dome-S-VisDrone.yml']
    rows=[]
    for rel in paths:
        item=json.loads((OUT/'_retrieval_metadata'/(rel.replace('/','__')+'.json')).read_text(encoding='utf-8'))
        assert item['encoding']=='base64' and item['sha']==expected[rel]['sha'] and item['path']==rel
        data=base64.b64decode(item['content'])
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert blob==item['sha']
        target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert target.read_bytes()==data
        else:
            target.write_bytes(data)
        rows.append(dict(path=rel,git_blob_sha=blob,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),url=item['html_url']))
    receipt=dict(status='ALL4_OPERATOR_CONFIG_BLOBS_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),repo='https://github.com/RicePasteM/Dome-DETR',
                 commit=meta['sha'],scope='Four operator/config files only;no imports,models,weights or fits',files=rows)
    (OUT/'source_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}|{'verified_files':len(rows)},ensure_ascii=False))

if __name__=='__main__':
    main()
