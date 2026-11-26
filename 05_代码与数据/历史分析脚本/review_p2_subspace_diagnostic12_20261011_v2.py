"""Verify actual CPU artifacts and preserve live queues; no efficacy substitution."""
from pathlib import Path
import base64,datetime,hashlib,json
import numpy as np
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
code=ROOT/'英文小论文/code/p2_subspace_diagnostic12_20261011_v1'
spec=read(code/'protocol.json');inputs=read(ROOT/'outputs/p2_subspace_diagnostic12_20261011_v1_inputs.json')
observed={}
for family,seeds in inputs.items():
    observed[family]={}
    for s,v in seeds.items():
        b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256']
        observed[family][s]=json.loads(b)
assert {n:sha(code/n) for n in spec['remote_sources']}==spec['remote_sources']
assert observed['subspace_deploy']['2']['files']=={p.name:sha(p) for p in code.iterdir() if p.is_file()}
actual=observed['subspace_CPU_actual']['2'];assert actual['exit_code']==0
cpu=actual['reports']['report.json'];assert cpu['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS'
assert cpu['checkpoint_sha256']=='e5ef315e707aabb4f6741d8cfab5f219608ed01a01b7f0fafbe9a79345b6b325'
assert cpu['original_numerical_rank']==64 and cpu['original_weight_unchanged'] and not cpu['AP_or_APs_measured']
assert cpu['protocol_sha256']==sha(code/'protocol.json') and cpu['source_sha256']==sha(code/'prepare.py')
dispatch=observed['subspace_wait_dispatch']['2'];assert dispatch['same_live'] and not dispatch['GPU_started'] and not dispatch['log']
dest=ROOT/'outputs/p2_subspace_diagnostic12_20261011_v1';dest.mkdir(exist_ok=True)
for n,b in dispatch['files'].items():
    decoded=base64.b64decode(b);assert hashlib.sha256(decoded).hexdigest()==dispatch['file_SHA256'][n]
    p=dest/n;p.parent.mkdir(exist_ok=True)
    if p.exists():assert p.read_bytes()==decoded,'Existing byte-exact artifact changed'
    else:
        with p.open('xb') as f:f.write(decoded)
assert read(dest/'seed2_CPU/report.json')==cpu
assert sha(dest/'seed2_CPU/report.json')==read(dest/'seed2_CPU/completion.json')['report_sha256']
assert sha(dest/'seed2_CPU/weights.npz')==cpu['weights_npz_sha256']
with np.load(dest/'seed2_CPU/weights.npz',allow_pickle=False) as z:weights={n:z[n].copy() for n in z.files}
assert set(weights)=={'original',*spec['interventions']}
W=weights['original'].astype(np.float64).reshape(256,64)
assert hashlib.sha256(weights['original'].tobytes()).hexdigest()==cpu['parent_restore_weight_SHA256']
dc=np.repeat(W.reshape(256,16,4).mean(2,keepdims=True),4,axis=2).reshape(256,64)
u,s,vh=np.linalg.svd(W,full_matrices=False);svd=(u[:,:16]*s[:16])@vh[:16]
independent=[]
for case,P in zip(cpu['cases'],[dc,svd]):
    mode=case['mode'];actual_weight=weights[mode];assert actual_weight.shape==(256,64,1,1) and actual_weight.dtype==np.float32
    assert hashlib.sha256(actual_weight.tobytes()).hexdigest()==case['weight_SHA256']
    M=actual_weight.astype(np.float64).reshape(256,64);vals=np.linalg.svd(M,compute_uv=False)
    rank=int((vals>vals[0]*spec['numeric_rank_rtol']).sum());assert rank==case['numerical_rank']==16
    norm_error=float(abs(np.linalg.norm(M)/np.linalg.norm(W)-1));assert norm_error<1e-6
    expected=P*(np.linalg.norm(W)/np.linalg.norm(P));projection_error=float(np.linalg.norm(M-expected)/np.linalg.norm(W));assert projection_error<1e-6
    basis=np.eye(64).reshape(64,16,2,2)
    assert np.array_equal(basis.reshape(64,64)@M.T,np.einsum('bcij,ocij->bo',basis,M.reshape(256,16,2,2)))
    assert abs(np.linalg.norm(M-W)/np.linalg.norm(W)-case['operator_relative_error'])<1e-6
    independent.append(dict(mode=mode,rank=rank,Frobenius_norm_relative_error=norm_error,Torch_Numpy_projection_relative_error=projection_error,basis64_equal=True,operator_relative_error=case['operator_relative_error']))
live=read(dest/'seed2_wait_live.json');assert live['parent_replay_status'] is None and live['GPU_compute_apps']
assert [{k:v for k,v in r.items() if k!='same_live'} for r in live['predecessors']]==spec['predecessor_handles']['2']
assert all(r['same_live'] for r in live['predecessors'])
epochs={};trainers=[];collectors=[];native=[]
for s,q in observed['observations'].items():
    assert q['code_SHA_verified'] and q['queue']['same_live'] and not q['failures']
    c=observed['collector_observation'][s];assert c['remote4_SHA_verified']
    d=observed['native_diagnostic_queue'][s];assert d['source_verified'] and d['queue']['same_live'] and d['failure'] is None
    w=observed['Warm_dispatcher'][s];assert w['source_SHA_verified'] and w['dispatcher']['same_live'] and w['failure'] is None
    assert q['queue']['pid']==spec['predecessor_handles'][s][0]['pid'] and q['queue']['start_ticks']==spec['predecessor_handles'][s][0]['start_ticks']
    assert d['queue']['pid']==spec['predecessor_handles'][s][1]['pid'] and d['queue']['start_ticks']==spec['predecessor_handles'][s][1]['start_ticks']
    assert w['dispatcher']['pid']==spec['predecessor_handles'][s][2]['pid'] and w['dispatcher']['start_ticks']==spec['predecessor_handles'][s][2]['start_ticks']
    epochs[s]={}
    for arm,j in q['jobs'].items():
        epochs[s][arm]=j['completed_epochs'];assert j['failure'] is None and j['first100']['all100_rows_exact']
        collector=c['jobs'][arm];assert collector['failure'] is None and collector['start_receipt_source_exact']
        if j['completed_epochs']<12:
            assert j['trainer']['same_live'] and j['controller']['same_live'] and collector['same_live']
            trainers.append(dict(seed=int(s),arm=arm,pid=j['trainer']['child_pid'],start_ticks=j['trainer']['child_start_ticks']))
            collectors.append(dict(seed=int(s),arm=arm,pid=collector['pid'],start_ticks=collector['start_ticks']))
        else:
            assert s=='2' and arm in ('backbone_silu','p2_skip') and not j['trainer']['same_live'] and collector['completed'] and collector['waiter_completed']
            native.append(dict(seed=int(s),arm=arm,AP=j['latest_stats'][0]*100,APs=j['latest_stats'][3]*100))
for family,seeds in inputs.items():
    for s,v in seeds.items():
        with (dest/f'{family}_seed{s}.json').open('xb') as f:f.write(v['payload'].encode())
write(dest/'transport_proofs.json',{f:{s:v['transport'] for s,v in seeds.items()} for f,seeds in inputs.items()})
report=dict(status='ACTUAL_FIXED12_P2_CPU_SUBSPACE_REVIEW_PASSED_NATIVE_EVALUATION_WAITING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    evidence='seed2_CPU/report.json',checkpoint_sha256=cpu['checkpoint_sha256'],independent_actual_weight_Numpy_checks=independent,negative_cases=cpu['negative_cases'],CPU_worker=dict(actual['receipt'],exit_code=0),GPU_waiter=dispatch['receipt'],GPU_waiter_observed_same_live=True,GPU_started=False,
    wait_phase=live['phase'],wait_predecessors=live['predecessors'],parent_replay_status=None,completed_epochs=epochs,training_observed_UTC={s:q['UTC'] for s,q in observed['observations'].items()},collector_observed_UTC={s:q['UTC'] for s,q in observed['collector_observation'].items()},live_actual_training_handles=trainers,live_primary_collectors=collectors,fixed12_seed2_user_primary_metrics=native,
    new_actual_trained_weight_controls=2,training_allowed=False,new_training_epochs=0,new_optimizer_updates=0,new_full_detector_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,existing_process_signals=0,
    planned_candidate_forward_batches=274,planned_candidate_native_evaluator_runs=2,seed1_P2_fixed12_pending=True,two_seed_training_efficacy_proven=False,root_proven=False,novelty_proven=False,
    scope=spec['inferential_limit'],next_action='Collect original user fixed12 endpoints and formal savedEMA replays. Execute queued rank16 runtime diagnostic only after all three bound owners stop, original P2 replayPASS and GPUidle; compare fresh native matching against cached P2 endpoint. Decide fresh training only from these results.',source_sha256=sha(__file__))
write(dest/'report.json',report)
text='审查与上机状态\n\nP2训练后对照已在B服务器CPU验证通过，使用真实第12轮EMA权重。两个还原矩阵秩均为16、范数均与原权重一致；64个基向量映射一致，5类非法输入被拒绝。下载权重后，独立NumPy计算再次通过。\n\n完整验证程序已进入B队列，但GPU验证尚未启动：原注意力训练、现有诊断队列与Warm收尾仍在运行，P2独立重载校验尚未产生。两个对照各验证548张图，共274个候选模型批次；新增训练轮次为0，基线前向为0。\n\n固定第12轮B服务器结果（AP点）：\n原基线 AP=13.71136，APs=8.63686\nP2 AP=14.22479，APs=9.00438，差值+0.51344/+0.36753\nSiLU AP=13.94253，APs=8.35198，差值+0.23117/-0.28488\n\n当前完成轮次：A SiLU10/P2 10/注意力6；B SiLU12/P2 12/注意力11。尚未完成的配置不能用中间轮代替第12轮判定。\n\n这两个对照只检验训练后模型的信息依赖；特征输出能量、训练容量未匹配，不能证明独特根因或新训练模型有效。P2第二种子完整终点与正式重载收尾仍需核实。\n'
with (dest/'审查与远程测试状态.txt').open('x',encoding='utf8') as f:f.write(text)
main=ROOT/'outputs/stem_component12_20261011_v1';state=read(main/'execution_state_v39.json')
state.update(status='ACTIVE_ORIGINAL_USER_REPLICATION_P2_CPU_SUBSPACE_PASSED_NATIVE_DIAGNOSTIC_QUEUED',supersedes='execution_state_v39.json',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,evidence='../p2_subspace_diagnostic12_20261011_v1/report.json',original_user_current_completed_epochs=epochs,original_user_actual_trainers_verified_live=trainers,
    P2_runtime_subspace_CPU_actual_passed=True,P2_runtime_subspace_GPU_started=False,P2_runtime_subspace_GPU_waiter=dispatch['receipt'],P2_phase_mean_sole_matched_capacity_attribution_allowed=False,phase_mean_training_launched=False,changed_next_action=report['next_action'],root_or_novelty_proven=False,
    remote_result_observer_liveness_scope='Four original-user trainers and four readonly primary collectors live at recordedUTC; both original queues, both native8400 queues and both Warm dispatchers additionally verified. Seed2 newCPU worker32596 completed0; native diagnostic waiter32667/257659599 live; no GPU native intervention started.')
write(main/'execution_state_v40.json',state)
wrapper=read(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v42.json')
wrapper.update(status=state['status'],supersedes='execution_state_v42.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v40.json',goal_complete=False)
write(ROOT/'outputs/original_stem12_20261011_v1/execution_state_v43.json',wrapper)
user=read(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v16.json')
user.update(status='TWO_SEED2_USER_ENDPOINTS_FOUR_ORIGINAL_TRAINERS_LIVE_AND_P2_RUNTIME_DIAGNOSTIC_CPU_READY',supersedes='execution_state_v16.json',evidence='../p2_subspace_diagnostic12_20261011_v1/report.json',completed_epochs=epochs,live_actual_training_handles=trainers,P2_runtime_subspace_GPU_started=False,phase_mean_training_launched=False,goal_complete=False)
write(ROOT/'outputs/user_backbone12_20261011_v1/execution_state_v17.json',user)
write(dest/'execution_state_v1.json',dict(status=report['status'],evidence='report.json',seed2_CPU_passed=True,seed2_GPU_waiter=dispatch['receipt'],seed2_GPU_started=False,seed1_fixed12_P2_pending=True,training_allowed=False,goal_complete=False))
print(json.dumps(dict(status=report['status'],independent=independent,epochs=epochs,Main_state='execution_state_v40.json',goal_complete=False)),flush=True)

write(dest/'manifest.json',dict(files={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}))
