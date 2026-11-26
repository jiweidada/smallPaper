"""Finish cache packing after a local np.savez 'file' keyword collision; no model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from a16_spatial_operator import pack
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'A16_实际特征空间变化审计_v1'
A15=ROOT/'A15_固定query的采样中心干预_v1'
assert not (OUT/'old_patch_packing_receipt.json').exists()
assert not list((OUT/'old_patches').glob('*.npz'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=OUT/'asset_selection.json';before=sha(selection)
asset=json.loads(selection.read_text(encoding='utf-8'))
assert len(asset['cases'])==25 and sum(c['tiny_GT'] for c in asset['cases'])==590
assert sha(OUT/asset['archive'])==asset['archive_sha256']
old=json.loads((ROOT/'A12_原生边界分布诊断_v1'/'asset_selection.json').read_text(encoding='utf-8'))
with np.load(OUT/'final_sampler_state.npz',allow_pickle=False) as f:state={k:f[k] for k in f.files}
rows=[]
for index,case in enumerate(old['cases']):
    capture=A15/'captures'/('case%02d.npz'%index);pred=A15/'predictions'/('case%02d.npz'%index)
    with np.load(capture,allow_pickle=False) as c,np.load(pred,allow_pickle=False) as p:
        mq=p['match_q'];mg=p['match_gt'];use=p['gt_sizes800'][mg]<8;qs=mq[use];gs=mg[use]
        data=pack(c['ca_query'][qs],c['reference'][qs],tuple(c['value_%d'%l] for l in range(4)),c['spatial_shapes'],state)
        data.update(query_indices=qs,gt_indices=gs,raw_boxes=p['raw_boxes'],raw_logits=p['raw_logits'],
            gt_boxes800=p['gt_boxes800'],gt_labels=p['gt_labels'],gt_sizes800=p['gt_sizes800'],
            gt_uids=p['gt_uids'],prefix=np.asarray(case['prefix']),image_filename=np.asarray(case['file']))
    output=OUT/'old_patches'/('case%02d.npz'%index);np.savez_compressed(output,**data)
    rows.append({'case':index,'capture_sha256':sha(capture),'prediction_sha256':sha(pred),
                 'patch_file':str(output.relative_to(OUT)),'patch_sha256':sha(output),'tiny':len(qs)})
assert sha(selection)==before
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PACKED_EXISTING_NATIVE_VALUES_NO_RESPONSE_METRICS',
    'local_preparation_error':'First preparer wrote annotation selection and archive,then failed on np.savez reserved file keyword before any patch was written. Corrected to image_filename. Selection and archive preserved.',
    'selection_sha256_before_and_after':before,'new_full_model_forwards':0,'source_operator_calls':0,
    'main_model_updates':0,'files':rows}
(OUT/'old_patch_packing_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'packed_cases':len(rows),'tiny':sum(r['tiny'] for r in rows),'new_model_forwards':0}))
