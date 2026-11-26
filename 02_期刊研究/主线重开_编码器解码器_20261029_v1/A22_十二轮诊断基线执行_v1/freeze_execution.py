"""Allocate only fixed12 baseline after CPU validation; no endpoint assay allocation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys,zipfile
sys.stdout.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent;R=O.parent
REMOTE='/root/c10_region_restore_20261029_v1/architecture_probe_v1'
C=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):return json.loads(p.read_text(encoding='utf-8'))
def wr(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (O/'execution_manifest_FROZEN.json').exists(),'Already allocated; inspect rather than repeat'
    previous=rd(R/'53_A21共享状态路径与研究设计核验回执.json')
    for row in previous['files']:assert sha(R/row['path'])==row['sha256'],row['path']
    pre=rd(O/'cpu_preflight_result.json')
    assert pre['status']=='CPU_DATA_METRIC_AND_CONSTRUCTION_CHECKS_PASS' and not pre['CUDA_initialized']
    assert pre['detector_complete_forwards']==pre['optimizer_steps']==0
    tested=rd(O/'preflight_source_manifest.json')['files']
    for row in tested:assert sha(O/row['path'])==row['sha256'],row['path']
    seal=rd(O/'scientific_design_seal.json');assert sha(O/'scientific_design_PRESPECIFIED.json')==seal['scientific_design_sha256']
    assert not (O/'evaluation_allocation_FROZEN.json').exists()
    state=rd(R/'15_当前状态与结果校验.json')
    assert state['model_forwards']==state['cap']==260 and state['training_updates']==2 and state['formal_detector_training_updates']==0
    snap=R/'history_before_A22_freeze_v1';snap.mkdir(exist_ok=False)
    for n in C+['53_A21共享状态路径与研究设计核验回执.json']:shutil.copy2(R/n,snap/n)
    dependencies=[]
    for local,remote in [
      ('A20_VSSD宿主严格装载与容量预检_v1/a20_host.py','a20_vssd_host_capacity_20261030_v1/a20_host.py'),
      ('A20_VSSD宿主严格装载与容量预检_v1/bundle_manifest.json','a20_vssd_host_capacity_20261030_v1/bundle_manifest.json'),
      ('A21_共享状态路径隔离与基线协议_v1/state_path_bridge.py','a21_shared_state_path_20261030_v1/state_path_bridge.py'),
      ('official_assets_v1/Dome_DETR_2dde3bc_v1/source/src/zoo/dome/postprocessor.py','dome_assets_v1/source/src/zoo/dome/postprocessor.py'),
      ('official_assets_v1/Dome_DETR_2dde3bc_v1/source/src/data/dataset/coco_dataset.py','dome_assets_v1/source/src/data/dataset/coco_dataset.py')]:
        dependencies.append({'path':REMOTE+'/'+remote,'sha256':sha(R/local),'local_source':str(R/local)})
    files=tested+[{'path':'cpu_preflight_result.json','sha256':sha(O/'cpu_preflight_result.json'),'bytes':(O/'cpu_preflight_result.json').stat().st_size}]
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_TRAINING_EXECUTION_ONLY',
       'purpose':'One seed fixed12 raw diagnostic baseline; not candidate-module training or innovation validation',
       'source_commit_VSSD':'c10de2f8b717a04459b09cbd651de2e768f302c8','source_commit_Dome':'2dde3bc1946a3e9fad9abd0612b59fc39bd6b861',
       'initial_state_digest':'8e27a3a06df1f52d91c4190c1f903e4a27566e9d8aaa7b4ce95d30e5f3c76229',
       'scientific_design_sha256':sha(O/'scientific_design_PRESPECIFIED.json'),
       'new_training_complete_forward_cap':38832,'new_optimizer_attempt_cap':9708,
       'new_actual_optimizer_update_cap':9708,'formal_epochs_cap':12,'training_seed':42301,
       'evaluation_complete_forwards_allocated':0,'no_validation_during_training':True,
       'query_mode_train':[250,500],'query_mode_later_eval_if_allocated':[500,500],
       'physical_batch':2,'accumulation_groups':4,'input_size':[800,800],
       'data_loader':{'workers':4,'prefetch_factor':2,'pin_memory':True,'persistent_workers':False,'context':'spawn','per_epoch_order_seed':'42301+epoch'},
       'augmentation':'Only deterministic per-image/epoch horizontal flip0.5 after exact author resize/255; normalized cx->1-cx',
       'orig_size_semantics':'[W,H], matching author coco_dataset.py and postprocessor.py; input size square800',
       'warmup':'Update#1 factor0.001; update#1000 factor1; use completed-update count so skips do not advance warmup',
       'checkpoint':'Raw fixed12 endpoint; preserve every completed epoch; no resume, EMA selection or val-based choice',
       'wall_cap_hours':48,'supervision':'One isolated child; hard wall-cap termination with identity-checked process-group signals and30second cleanup grace; no restart or followup eval',
       'AMP_skip_policy':'Normal GradScaler overflow skips counted; stop at3consecutive or32cumulative; no make-up batches/epochs',
       'native_source_preservation':'A20 strict loader verifies original source/dependency blobs/EMA; no native detector arithmetic edits in baseline',
       'state_norm_control_fidelity':'Future conditional assay requires <=1% relative energy mismatch after nativeFP32 quantization in every primary case; pre-outcome engineering guard, no case removal or rescaling search',
       'files':files,'dependencies':dependencies,
       'libraries':pre['versions'],'COCOeval_source_sha256':pre['installed_COCOeval_source_sha256'],
       'mainline_validated':False,'existing_user_authorization_continues':True}
    wr(O/'execution_manifest_FROZEN.json',manifest)
    with zipfile.ZipFile(O/'execution_allocation_control.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(O/'execution_manifest_FROZEN.json','execution_manifest_FROZEN.json')
    wr(O/'execution_allocation_receipt.json',{'sha256':sha(O/'execution_allocation_control.zip'),
       'bytes':(O/'execution_allocation_control.zip').stat().st_size,'previous_full_calls':260,
       'new_training_cap':38832,'cumulative_full_cap':39092,'previous_actual_updates':2,
       'new_actual_update_cap':9708,'new_eval_allocation':0,'manifest_sha256':sha(O/'execution_manifest_FROZEN.json')})
    state.setdefault('authoritative_phase_history',[]).append(state['latest_authoritative_phase'])
    state.update(recorded_utc=manifest['utc'],cap=39092,current_goal_turn='progress_A22_fixed12_training_execution_frozen_before_launch',
       cap_scope='260closed historical complete calls +38832new fixed12 training calls; total39092capOPEN, no endpoint/assay calls allocated.',
       training_counts_terminal=False)
    state['latest_authoritative_phase']={'utc':manifest['utc'],'phase':'A22','classification':'progress',
       'status':'TRAINING_EXECUTION_FROZEN_NOT_YET_LAUNCHED','full_model_forwards':260,'cumulative_authorized_protocol_cap':39092,
       'actual_cumulative_detector_optimizer_updates':2,'formal_detector_training_updates':0,
       'new_training_full_cap':38832,'new_actual_update_cap':9708,'new_epoch_cap':12,
       'new_eval_calls_allocated':0,'manifest_sha256':sha(O/'execution_manifest_FROZEN.json'),'mainline_validated':False}
    wr(R/'15_当前状态与结果校验.json',state)
    print(json.dumps(rd(O/'execution_allocation_receipt.json'),ensure_ascii=False))
if __name__=='__main__':main()
