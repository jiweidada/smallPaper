"""Verify/decode7 public Focus-DETR operator blobs;never import fetched project."""
from pathlib import Path
from datetime import datetime,timezone
import base64
import hashlib
import json

ROOT=Path(__file__).resolve().parent
AUDIT=ROOT/'source_audit_v1'
OUT=ROOT/'comparison_sources_v1/Focus_DETR_operators_v1'

def main():
    commits=json.loads((AUDIT/'Focus_DETR_last_path_commit.json').read_text(encoding='utf-8'))
    commit=commits[0]['sha']
    tree=json.loads((AUDIT/'Focus_DETR_projects_tree.json').read_text(encoding='utf-8'))
    expected={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    paths=['focus_detr/modeling/foreground_supervision.py','focus_detr/modeling/focus_detr.py',
           'focus_detr/modeling/focus_detr_transformer.py','focus_detr/configs/models/focus_detr_r50.py',
           'focus_detr/modeling/__init__.py','focus_detr/modeling/dn_criterion.py',
           'focus_detr/modeling/two_stage_criterion.py']
    rows=[]
    for rel in paths:
        metadata=OUT/'_retrieval_metadata'/(rel.replace('/','__')+'.json')
        item=json.loads(metadata.read_text(encoding='utf-8'))
        assert item['encoding']=='base64' and item['sha']==expected[rel]['sha']
        assert 'path' not in item or item['path']=='Focus-DETR/projects/'+rel
        data=base64.b64decode(item['content'])
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert blob==item['sha']
        path=OUT/'projects'/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            assert path.read_bytes()==data
        else:
            path.write_bytes(data)
        rows.append(dict(path=path.relative_to(OUT).as_posix(),git_blob_sha=blob,
                         sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
                         url=item.get('html_url','https://github.com/huawei-noah/noah-research/blob/'+commit+'/Focus-DETR/projects/'+rel)))
    result=dict(status='ALL7_OPERATOR_BLOBS_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),
                repo='https://github.com/huawei-noah/noah-research',commit=commit,
                source_binding='Paper-linked official repository;latest commit affecting Focus-DETR directory plus captured projects subtree Git blob IDs.',
                projects_tree=tree['sha'],scope='Seven listed operator/config/export files only;no imports,builds,models or fits',files=rows)
    (OUT/'source_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}|{'verified_files':len(rows)},ensure_ascii=False))

if __name__=='__main__':
    main()
