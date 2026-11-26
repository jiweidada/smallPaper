"""Verify new saved B8 arrays, preserve prior state, publish an interim result."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
import zipfile
import numpy as np
from preservation_boundary_A58_v1 import P,B,sha,save_new,begin,finish

R54=B/'A54_场景尺度约束候选_20261101_v1'
def load(p):return json.loads(Path(p).read_text('utf-8'))
def replace(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

rpc=load(R54/'A58_B8_interim_export_v1.json');assert rpc['exit']==0
export=json.loads(rpc['stdout']);assert load(R54/'A58_B8_interim_download_v1.json')['status']=='TRANSFERRED'
z=R54/'A57_downloads/A58_B8_interim_small_v1.zip'
assert sha(z)==export['sha256'] and z.stat().st_size==export['bytes']
assert not (P/'B8_interim_small_v1.zip').exists();shutil.copyfile(z,P/'B8_interim_small_v1.zip')
d=P/'B8_interim_remote_v1';d.mkdir(exist_ok=False)
with zipfile.ZipFile(z) as archive:
    assert all(Path(n).name==n for n in archive.namelist())
    for n in archive.namelist():(d/n).write_bytes(archive.read(n))
manifest=load(d/'B8_artifact_manifest.json')
assert manifest['epoch']==8 and manifest['arm']=='B' and manifest['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
for n in ['epoch08_EMA_metrics.json','epoch08_EMA_precision_recall.npz','epoch08_input_flow.json','initialization.json']:
    row=manifest['files'][n];assert (d/n).stat().st_size==row['bytes'] and sha(d/n)==row['sha256'],n
b4manifest=load(P/'B4_interim_remote_v1/B4_artifact_manifest.json')
assert manifest['files']['epoch08_EMA_GT.json']==b4manifest['files']['epoch04_EMA_GT.json']
m=load(d/'epoch08_EMA_metrics.json');assert m==export['metrics']
arrays=np.load(d/'epoch08_EMA_precision_recall.npz',allow_pickle=False)
p,rec,ious=arrays['precision'],arrays['recall'],arrays['iou_thresholds']
assert p.shape==(10,101,10,4,3) and rec.shape==(10,10,4,3)
assert np.isfinite(p).all() and np.isfinite(rec).all()
assert p.min()>=-1 and p.max()<=1 and rec.min()>=-1 and rec.max()<=1
def ap(area=0,iou=None):
    v=p[:,:,:,area,-1]
    if iou is not None:v=v[np.isclose(ious,iou)]
    return float(v[v>=0].mean()*100)
v=rec[:,:,1,-1]
actual=dict(AP=ap(),AP50=ap(iou=.5),AP75=ap(iou=.75),APs=ap(1),APs75=ap(1,.75),APm=ap(2),APl=ap(3),ARs=float(v[v>=0].mean()*100))
error=max(abs(actual[k]-m[k]) for k in actual);assert error<1e-10
flow=load(d/'epoch08_input_flow.json');assert flow['epoch']==8 and flow['optimizer_updates']==6472
assert len(flow['first_eight_actual_input_sha256'])==len(flow['first_eight_pre_forward_rng_sha256'])==8
save_new(P/'B8_saved_arrays_local_verification_v1.json',dict(status='PASS_SAVED_BYTES_AND_ARRAY_AGGREGATION',
    metrics=m,recomputed_metrics=actual,maximum_difference_pp=error,GT_identical_to_B4=True,
    checkpoint_preserved_remote=manifest['files']['epoch08.pth'],GT_and_predictions_preserved_remote=True,
    new_audit_detector_forwards=0,new_optimizer_updates=0,extra_COCO_matching_runs=0,
    limits='Saved array and GT hash checks; no independent rerun of detector or COCO matching; B8 is interim, candidate untrained'))
rpc=load(R54/'A58_formal_observe_017.json');assert rpc['exit']==0
obs=json.loads(rpc['stdout']);assert obs['source_binding_verified'] and obs['protocol_sha256']==sha(P/'protocol_FROZEN_v1.json')
assert obs['supervisor']['status']=='FORMAL_RUNNING' and obs['supervisor']['arm']=='B'
for key in ['supervisor_identity','child_identity']:
    assert obs[key]['exists'] and obs[key]['expected_ticks_match'] and obs[key]['state']!='Z'
assert not obs['arms']['M']['directory_exists'] and not obs['arms']['Q']['directory_exists']
b=obs['arms']['B'];event=next(e for e in b['completed_epochs'] if e['epoch']==8)
assert event['EMA_validation']==m
c=b['journal_counts'];f=c['forward_start'];u=c['optimizer_returned'];completed=c['forward_complete']
assert b['forward_phases']['validation']==274 and f<=10159 and u<=9708
begin('A58_B8_interim_v1','A58_B4_and_geometry_preservation_receipt_v1.json')
for i in range(10,18):
    rpc=load(R54/('A58_formal_observe_%03d.json'%i));assert rpc['exit']==0
    save_new(P/'formal_observations_v1'/('%03d.json'%i),json.loads(rpc['stdout']))
transport=P/'transport_receipts_v4';transport.mkdir(exist_ok=False)
for src in sorted(R54.glob('A58_*.json')):shutil.copyfile(src,transport/src.name)
now=datetime.now(timezone.utc).isoformat();progress=b['progress.json']
report=P/'08_B8中期结果与固定终点边界.txt'
with report.open('x',encoding='utf-8') as dst:
    dst.write(f'''A58原版B8中期结果

B4 EMA：AP22.567752，APs12.661749。
B8 EMA：AP{m['AP']:.6f}，AP50{m['AP50']:.6f}，AP75{m['AP75']:.6f}，APs{m['APs']:.6f}，small AP75{m['APs75']:.6f}，ARs{m['ARs']:.6f}。
均为冻结COCO-style10类/maxDets300口径，非VisDrone官方榜单分数。B8仍覆盖38602个计分GT，GT文件字节与B4完全一致。
保存的precision/recall数组重聚合8项指标，最大差{error:.3g}pp；没有增加检测器前向、训练更新或COCO匹配运行。
B8权重继续保留在远端，SHA256：{manifest['files']['epoch08.pth']['sha256']}；GT/预测也保留并记录哈希。本次仅下载指标、数组和输入流，见B8_interim_remote_v1/B8_artifact_manifest.json。

解释：原版在本配方下继续学习。M/Q尚未启动，没有候选收益或主线成立结论；不以B8提前替代B12学习门槛。
当前已核验原监督器26447/start29306939、B训练26448/start29306942存活，9项冻结源码一致。进度文件记录第{progress['epoch']}轮、第{progress['microbatch']}/809批；事件前缀为前向开始{f}、完成{completed}、优化器返回{u}次，含274次验证前向。观察时间{now}；此为记录前缀，后续需重新读取实际进程。

下一步：继续原B至固定12轮；B12 AP≥8、APs≥3才启动fresh M12、Q12。Q相对B的APs≥+0.5pp，相对M的APs≥+0.3pp，相对B的AP≥−0.2pp，Q/M模型中位前向延迟≤1.20。4/8轮只观察，不选最好轮，不增seed、轮次或超参抢救。
标注少数占比事实及其机制限制见06_标注几何事实与机制边界.txt；最近方法的有限源码核查见04_AdaCluster有界源码核查.txt。几何占比不是特征均值损害证据，候选还需固定对照检验。
目标保持active，数据、代码、权重、文献索引与旧裁决保留。
''')
current=load(P/'CURRENT_A58.json')
current.update(status='A58_B8_INTERIM_VERIFIED_B_CONTINUES_MQ_UNSTARTED',observed_utc=now,
    new_detector_forwards=6+f,new_optimizer_updates=6+u,formal_calls_observed=f,
    formal_completed_calls_observed=completed,formal_updates_observed=u,formal_completed_epochs_observed=len(b['completed_epochs']),
    baseline_B12_gate_passed=None,baseline_gate_status='Pending fixed12; B4/B8 interim only',
    candidate_MQ_trained=False,formal_supervisor_identity=obs['supervisor_identity'],formal_trainer_identity=obs['child_identity'],
    report=str(report),next='Observe original fixedB12; B12 learning gate controls unstarted M/Q; no rescue.',
    limits='Baseline B4/B8 and saved-array checks only; candidate efficacy, feature-mean harm and novelty remain unproven.')
current.setdefault('B_interim_metrics',{})['epoch8_EMA']=m
replace(P/'CURRENT_A58.json',current)
for name in ['01_候选判别表.json','15_当前状态与结果校验.json']:
    x=load(B/name)
    x.update(status=current['status'],A58_execution=current,latest_report=str(report),latest_update_utc=now,
             pilot_validation_status='B8 AP23.896074/APs14.024436 saved arrays verified; B continues, M/Q unstarted')
    if name.startswith('01_'):
        row=next(r for r in x['candidates'] if r['id']==current['candidate'])
        row.update(status=current['status'],report=str(report),interim_B8_metrics=m,
                   formal_observed_calls=f,formal_observed_updates=u,candidate_trained=False)
    else:
        assert x['cap']==276634 and x['model_forwards']<=246157+f and x['training_updates']<=204476+u
        x.update(model_forwards=246157+f,training_updates=204476+u,recorded_utc=now,
            current_native_job_terminal=False,counts_terminal=False,training_counts_terminal=False,
            current_phase='A58_B8_INTERIM_B_LIVE',current_goal_turn='progress: new B8 actual metrics and saved arrays verified; originalB9 live; candidate M/Q unstarted',
            current_user_task_status='Goal active; wait fixedB12 gate and candidate M/Q fixed endpoints')
        x.setdefault('authoritative_phase_history',[]).append(dict(phase='A58',utc=now,classification='progress_and_verified_wait',
            B8_metrics=m,observed_formal_calls=f,observed_formal_updates=u,new_audit_model_calls=0))
    replace(B/name,x)
save_new(P/'goal_progress_A58_B8_v1.json',dict(previous_goal_turn_classification='verified_wait on originalB',
    current_goal_turn_classification='progress: B8 actual result and saved arrays; verifiedwait originalB9',
    goal_status='active',mainline_validated=False,B8_metrics=m,observed_formal_calls=f,observed_formal_updates=u,new_audit_model_calls=0,report=str(report)))
save_new(P/'A58_turn_handoff_B8_v1.json',dict(goal_status='active',current=current,bridge_session=19671,credentials_recorded=False,
    remote_root='/root/uav_a58_20261102_v1/pilot_v1',last_observer_index=17,
    observer='/root/uav_a54_20261101_v1/A58_uploads/observe_remote_A58_v1.py',observer_sha256=sha(P/'observe_remote_A58_v1.py'),
    global_actual_model_calls=246157+f,global_actual_optimizer_updates_including_capacity=204476+u,
    global_model_call_cap=276634,next='Continue observing originalB12. M/Q automatically conditional; do not rerun preparation, capacity, geometry or launchers.'))
finish('A58_B8_interim_preservation_receipt_v1.json','B8_INTERIM_ARRAYS_VERIFIED_B_LIVE',observed_formal_model_calls=f,observed_formal_updates=u,new_audit_model_calls=0)
print(json.dumps(dict(status=current['status'],B8_AP=m['AP'],B8_APs=m['APs'],B_current_epoch=progress['epoch'],
    global_model_calls=246157+f,global_updates=204476+u,goal_status='active'),ensure_ascii=False))
