from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

root=Path(__file__).resolve().parent
receipt=json.loads((root/'annotation_geometry_audit_v1'/'completed.json').read_text(encoding='utf-8'))
rows=receipt['results']
assert receipt['status']=='PASS' and receipt['valid_positives']==343204
for geometry in ('square stretch','YOLO-style centered letterbox'):
    for size in (640,960):
        selected=[r for r in rows if r['geometry']==geometry and r['input_size']==size]
        assert len(selected)==5
        allrow=next(r for r in selected if r['subset']=='all_train6471')
        components=[r for r in selected if r['subset']!='all_train6471']
        for key in ('images','positives','size_lt8','size_8_16','size_16_32','size_ge32'):
            assert sum(r[key] for r in components)==allrow[key], (geometry,size,key)
        for item in selected:
            assert sum(item[key] for key in ('size_lt8','size_8_16','size_16_32','size_ge32'))==item['positives']
            for stride in (4,8,16):
                assert 0<=item[f'small_center_shared_small_s{stride}']<=item[f'center_shared_any_s{stride}']<=item['positives']
diagnostic=next(r for r in rows if r['subset']=='diagnostic100' and r['geometry']=='YOLO-style centered letterbox' and r['input_size']==640)
assert diagnostic['positives']==4491
assert diagnostic['positives']-diagnostic['size_ge32']==4129
with (root/'annotation_geometry_audit_v1'/'per_image.csv').open(encoding='utf-8',newline='') as stream:
    csv_rows=sum(1 for _ in csv.DictReader(stream))
assert csv_rows==6471*4
now=datetime.now(timezone.utc).isoformat()
board_path=root/'01_候选判别表.json'
board=json.loads(board_path.read_text(encoding='utf-8'))
board['latest_update_utc']=now
board['status']='STAGE_SPECIFIC_PROBLEM_PRIORITIZED_NOT_YET_VALIDATED'
board['model_candidates']=['TinyFormer-S-PBM author VisDrone release','ECDet-S','DEIMv2-S']
board['current_priority_question']='After existing SSA/PBM detail compensation, where do crowded tiny-object failures remain: encoder candidate formation, Top-K selection, or decoder refinement?'
board['new_empirical_evidence']='6471 train annotation geometry audit completed with all annotation hashes verified; no detector inference or training'
board['mainline_validated']=False
board['candidates'][0].update(status='PRIORITY_FOR_BOUNDED_STAGE_TRACE',nearest=['TinyFormer SSA/PBM','EdgeCrafter/ECDet','DEIMv2 STA','BUTR for query budget controls'],next='Acquire official author VisDrone weights and verified full source, then only the capped eval-mode stage diagnostic in protocol06')
board['candidates'][1].update(status='DEPRIORITIZED_GENERIC_REUSE_ALREADY_PRESENT',nearest=['DEIMv2 and ECDet shared value implementation','ShareDAB-DETR','Skip-Attention','SparseViT','ESOD'],next='Require a separately documented residual operation and mechanism before reopening; no generic attention-plus-convolution novelty claim')
board['asset_gate']={'full_source_verified':False,'author_weight_downloaded':False,'tinyformer_commit':'9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1','browser_text_reconstruction':'FAILED_GIT_BLOB_HASH; not used as executable source'}
board['latest_report']=str(root/'05_最近方法淘汰与标注证据.txt')
board['bounded_diagnostic_protocol']=str(root/'06_TinyFormer分阶段最小诊断协议.json')
board_path.write_text(json.dumps(board,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

log_path=root/'02_检索记录.json'
log=json.loads(log_path.read_text(encoding='utf-8'))
log['continuation_amendment_20261029']={
  'queries':['"tiny object detection" "sampling" "deformable" 2025 2026','"small object" "ViT" "multi-scale" "adapter" detection 2026','"tiny object" "token" "separability"','"TinyFormer" "Spatial Semantic Adapter"','"Slim-DETR: Real-Time Tiny Object Detection"','"small object" "sub-pixel" "DETR"','"tiny object detection" "aliasing"','"DETR" "query-wise" "decoder"','"DETR" "early exit"','"SkipAt" vision transformers paper','"Pay Less Attention" "Vision Transformers"','"SparseViT" "activation" "object detection"','"ESOD" "Efficient Small Object Detection" paper github'],
  'primary_sources_added':[
    {'title':'TinyFormer','url':'https://arxiv.org/html/2605.25046v1','read_scope':'Sections3.1,3.2,4.3,4.4, limitations; official README and named source modules'},
    {'title':'Skip-Attention','url':'https://arxiv.org/html/2301.02240v2','read_scope':'Sections3.1-3.3; selected appendix implementation passages; final ICLR abstract metadata'},
    {'title':'SparseViT','url':'https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SparseViT_Revisiting_Activation_Sparsity_for_Efficient_High-Resolution_Vision_Transformer_CVPR_2023_paper.html','read_scope':'Abstract and search-visible ablation passages only'},
    {'title':'ESOD','url':'https://arxiv.org/abs/2407.16424','read_scope':'Abstract and official repository documentation only'}],
  'unverified_lead_excluded':'Slim-DETR third-party notes; corresponding primary method not verified',
  'technical_access_notes':'Some GitHub/other direct HTTPS requests failed TLS. One official TinyFormer git tree retrieved and commit verified. Browser-reconstructed raw text failed blob hashes and was not executed. Standard web backend later failed; connected search-service tool continued retrieving primary sources.'}
log_path.write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

previous=root.parent/'前期调研_码率受限航拍小目标_20261028_v1'
ledger_path=previous/'远端验证_20261029_v1'/'execution_ledger.json'
ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
ledger['current_architecture_report']=str(root/'05_最近方法淘汰与标注证据.txt')
ledger['annotation_geometry_audit']={'status':'PASS','images':6471,'valid_positives':343204,'files_hashed':6471,'diagnostic_positives_match_old':4491,'diagnostic_input_small_match_old':4129}
ledger['current_priority']='bounded TinyFormer stage-trace preparation; algorithm mechanism not validated'
ledger['new_experiments_this_architecture_turn']={'detector_forwards':0,'detector_updates':0,'remote_jobs_started':0,'cpu_annotation_audits':1}
ledger['mainline_validated']=False
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

files=['00_方向纠偏与架构证据.txt','01_候选判别表.json','02_检索记录.json','04_标注几何审计预定方案.json','05_最近方法淘汰与标注证据.txt','06_TinyFormer分阶段最小诊断协议.json','annotation_geometry_audit_v1/completed.json','annotation_geometry_audit_v1/per_image.csv']
check={'recorded_utc':now,'geometry_invariants':'PASS','old_diagnostic_GT_contract':'4491 positives,4129 input-small reproduced','csv_rows':csv_rows,'model_runs':0,'files':[]}
for name in files:
    path=root/name
    if path.suffix=='.json':json.loads(path.read_text(encoding='utf-8'))
    check['files'].append({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'07_本轮校验与状态.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','csv_rows':csv_rows,'old_diagnostic_contract_matches':True,'updated_artifacts':len(files),'mainline_validated':False,'new_model_experiments':0},ensure_ascii=False))
