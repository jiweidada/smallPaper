"""Freeze a bounded source-faithful diagnostic before any Dome full forward."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

R=Path(__file__).resolve().parent
O=R/'A8_固定query的窗口更新诊断_v1'
O.mkdir(exist_ok=True)
assert not (O/'started.json').exists(),'Do not modify a started protocol.'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=read(R/'15_当前状态与结果校验.json')
assert state['latest_authoritative_phase']['full_model_forwards']==185
assert state['latest_authoritative_phase']['main_model_training_updates']==0
old=R/'A7_源码去重与资产准备_v1'
p=read(old/'A8_minimal_memory_path_draft.json')
strict=read(R/'official_assets_v1/Dome_DETR_2dde3bc_v1/strict_model_load.json')
assert strict['status']=='STRICT_AUTHOR_DOME_S_VISDRONE_MODEL_LOADED_CPU'
assert strict['all_state_tensor_values_equal_checkpoint'] and strict['state_keys']==1033
assert strict['new_full_detector_forwards']==0 and strict['num_classes']==12
for n in ['a8_window_memory_probe.py','a8_analysis.py','dome_probe_loader.py']:
 shutil.copy2(R/n,O/n)
shutil.copy2(old/'A8_draft_cases.json',O/'cases.json')
p.update(status='FROZEN_BEFORE_FIRST_DOME_FORWARD',frozen_utc=datetime.now(timezone.utc).isoformat(),
 strict_CPU_model_digest=strict['state_digest'],case_file='cases.json',case_sha256=sha(O/'cases.json'),
 input=strict['preprocessing'],training_memory_verified=False,executed=False)
p['source_contract_corrections_before_any_Dome_prediction']=[
 {'change':'GT analysis labels0..9 are mapped to native category IDs1..10; classification output has12 channels.',
  'evidence':'config dataset num_classes12,converter category IDs1..10,and strict checkpoint score-head(12,256).'},
 {'change':'Replace draft fixedTop300 language with the actual native flatTopQ, Q=raw dynamic query count.',
  'evidence':'postprocessor.py46–59 overrides num_top_queries with logits.shape1. No fixedTop300 experimental arm is introduced.'},
 {'change':'A selected window participates in within-window and cross-selected-window attention.',
  'evidence':'get_roi_features.py146–158 applies the same layer before and after axis permutation. No per-window independence assumption is made.'}]
p['label_contract']={'output_classes':12,'valid_native_category_ids':list(range(1,11)),
 'analysis_label_mapping':'native_id=prior_10class_id+1',
 'unused_predicted_channels':'0/11 remain in native scores/topQ; no post-hoc class filtering before selection.'}
p['comparison_baseline']='REPLAY_NATIVE. All replay arms bypass query initialization using the same cached native tuple; NATIVE is the numerical identity reference, not a latency baseline.'
p['endpoints']=['Actual selected10x10P3 window count; GT center and normalized-area coverage at800.',
 'Native flatTopQ class-aware one-to-one physical-query IoU50/75 coverage for all,<8,and8–16 cohorts; each cohort matched separately, not a joint partition or officialAP.',
 'Exact postprocess/query identity for native replay and maxabs normalized-box/raw-logit tolerance1e-6.',
 'State/input/positional hashes; instrumented inference memory/time only, on a shared GPU.']
p['runtime_free_memory_gate']='At least8GiB free before model-to-CUDA; do not stop or modify existing GPU processes.'
p['helper_selfchecks']='Physical query reuse across classes, non-overlap/area conservation, and same-count window masks; no detector calls.'
save(O/'protocol.json',p)
inventory=[]
for name in ['protocol.json','cases.json','a8_window_memory_probe.py','a8_analysis.py','dome_probe_loader.py']:
 inventory.append({'file':name,'sha256':sha(O/name),'bytes':(O/name).stat().st_size})
save(O/'freeze_manifest.json',{'utc':p['frozen_utc'],'files':inventory,
 'author_weight_sha256':p['weight_sha256'],'author_commit':p['source_commit'],
 'strict_CPU_receipt_sha256':sha(R/'official_assets_v1/Dome_DETR_2dde3bc_v1/strict_model_load.json'),
 'phase_budget':{'prior':185,'max_new':15,'max_global':200,'new_fits':0,'updates':0}})
print(json.dumps({'status':p['status'],'output':str(O),'files':len(inventory),
 'cases':3,'tiny_gt':69,'max_new_full_forwards':15,'new_forwards_executed_by_prepare':0},ensure_ascii=False))
