"""Freeze diagnostic preparation checks and actual second-epoch reference numbers."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';MAIN=OUT/'stem_component12_20261011_v1';REF=OUT/'native_stem_scratch12_20261011_v3';DIAG=OUT/'native_selection_diagnostic12_20261011_v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
checks=[];meta=read(DIAG/'preparation_transport_v1.json');build=read(DIAG/'build_manifest.json')
for seed in (1,2):
    p=DIAG/f'preparation_seed{seed}_v1.json';t=meta[str(seed)]
    assert p.stat().st_size==t['bytes'] and sha(p)==t['sha256']
    report=read(p);assert report['status']=='PASS_BOTH_NATIVE_BINDINGS_AND_LIVE_TRAINING_REPLAY_GUARD'
    assert report['detector_forwards']==report['optimizer_updates']==0
    assert {x['arm'] for x in report['cases']}=={'original_stem','native_stem_scratch'}
    for case in report['cases']:
        assert case['returncode']==0 and not case['stderr']; row=json.loads(case['stdout'])
        assert row['seed']==seed and row['immutable_native_bundle_verified'] and row['live_training_guard_rejected'] and not row['torch_imported']
    p=DIAG/f'integration_seed{seed}_v1.json';assert sha(p)=='bb42b981572e4c73f698fcee930f5f8460c8a91578e8dde19919ff91a59e83bf'
    r=read(p);assert r['diagnostic_code_sha256']==build['code_sha256'] and r['actual_detector_forwards']==0
    assert len(r['cases'])==2 and not r['CUDA_initialized']
    for case in r['cases']:
        assert case['exact_fixture_outputs'] and case['RNG_unchanged'] and case['image_ID_and_size_join_checked']
        assert case['source_head_calls']==dict(score=1,bbox=1) and case['additional_head_calls']==0
    checks.append(dict(seed=seed,integration_exact_transfer=True,code_binding_verified=True,live_training_replay_guard_verified=True))
write(DIAG/'preparation_validation_v1.json',dict(status='PASS_BOTH_SERVER_CPU_INTEGRATION_AND_NATIVE_BINDING_CHECKS',checks=checks,
    new_actual_detector_forwards=0,new_training_epochs=0,synthetic_fixture_forwards_total=8,actual_fixed12_GPU_replays=0,
    code_and_protocol_manifest='build_manifest.json',verification_scope='Actual native producer methods inside synthetic CPU fixtures, real StageTrace hooks, immutable model-runtime bindings and live-training replay rejection only.',
    limitation='No real fixed12 checkpoint recording/evaluator replay or detector mechanism efficacy is established by these preparation checks.'))
metadata=read(REF/'compact_transport_v5.json');observations=[]
for seed in (1,2):
    p=REF/f'compact_payload_seed{seed}_v5.json';t=metadata[str(seed)]
    assert p.stat().st_size==t['expected_bytes'] and sha(p)==t['expected_sha256']
    o=read(p);assert o['seed']==seed and o['code_sha256_verified'] and not o['failures']
    assert all(x['same_live'] and x['state']!='T' for x in o['processes'].values())
    assert o['completed_epochs']==2 and o['latest_metrics']['epoch']==1
    observations.append(o)
write(REF/'compact_observation_v5.json',observations)
progress=read(MAIN/'progress_v19.json');snapshot=read(MAIN/'training_observation_v19_transport_verification.json')
assert sha(MAIN/'training_observation_v19.json')==snapshot['verified_snapshot_sha256']==progress['snapshot_sha256']
pairs=[]
for o in observations:
    seed=o['seed'];full=next(r for r in progress['rows'] if r['seed']==seed and r['arm']=='original_stem' and r['epoch']==2)
    v=o['latest_metrics']['test_coco_eval_bbox'];a,s=100*v[0],100*v[3]
    pairs.append(dict(seed=seed,epoch=2,enhanced_AP=full['AP'],random_native_AP=a,delta_AP_pp=full['AP']-a,
        enhanced_APs=full['APs'],random_native_APs=s,delta_APs_pp=full['APs']-s))
write(MAIN/'same_epoch_random_native_comparison_v2.json',dict(status='ANALYZED_DESCRIPTIVE_SAME_EPOCH_EARLY_LOGS',current_S_pairs=pairs,
    full_progress_sha256=sha(MAIN/'progress_v19.json'),random_native_observation_sha256=sha(REF/'compact_observation_v5.json'),
    all_seeds_preserved=True,same_completed_epoch=2,fixed12_efficacy_established=False,root_cause_established=False,
    interpretation='Epoch2 signs differ: enhanced loses on seed1, slightly exceeds random native on seed2. No uniform efficacy or causal mechanism established. Fixed12 remains the endpoint; old n-scale evidence is preserved separately.',
    new_model_forwards=0,new_optimizer_updates=0))
ref=read(REF/'execution_state_v3.json');ref.update(supersedes='execution_state_v3.json',evidence='compact_observation_v5.json',
    completed_epochs={str(o['seed']):o['completed_epochs'] for o in observations},optimizer_updates={str(o['seed']):o['runtime']['training_audit_live.json']['actual_optimizer_steps'] for o in observations},
    latest_live_processes={str(o['seed']):o['processes'] for o in observations},transport_verification='compact_transport_v5.json exact SHA256 and byte length verified')
write(REF/'execution_state_v4.json',ref)
main=read(MAIN/'execution_state_v14.json');live=read(MAIN/'training_observation_v19.json')
main.update(supersedes='execution_state_v14.json',evidence='training_observation_v19.json',baseline_comparison='progress_v19.json',
    reference_state='../native_stem_scratch12_20261011_v3/execution_state_v4.json',previous_goal_turn='progress',current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    independent_native_selection_diagnostic='../native_selection_diagnostic12_20261011_v1/preparation_validation_v1.json',same_epoch_reference_comparison='same_epoch_random_native_comparison_v2.json',
    finding='Joined selection/bbox-offset/stage recording passed actual native-producer CPU fixture checks and live-training rejection on both real bundles/server seeds. Both random-native arms completed2 epochs; same-epoch enhanced deltas have different signs across seeds.',
    next='Collect bounded fixed12 terminals; validate their actual EMA replay and diagnose query selection versus selected-only initial regression. No new mechanism acceptance from CPU preparation or early AP logs.')
for seed in (1,2):
    main['current_live_trainers']['seed'+str(seed)]={a:live['seed'+str(seed)]['jobs'][a]['processes'] for a in ('original_stem','detail_lowres','highfreq_rgb')}
    assert all(p['same_live'] and p['state']!='T' for ps in main['current_live_trainers']['seed'+str(seed)].values() for p in ps)
write(MAIN/'execution_state_v15.json',main)
full=read(OUT/'original_stem12_20261011_v1/execution_state_v17.json');full.update(supersedes='execution_state_v17.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v15.json',reference_state='../native_stem_scratch12_20261011_v3/execution_state_v4.json')
write(OUT/'original_stem12_20261011_v1/execution_state_v18.json',full)
write(DIAG/'execution_state_v1.json',dict(status='PREPARED_CPU_CHECKED_ACTUAL_FIXED12_RECORDING_PENDING',goal_complete=False,actual_fixed12_GPU_replays=0,automatic_retry=False,
    evidence='preparation_validation_v1.json',source='英文小论文/code/native_selection_diagnostic12_20261011_v1',next=main['next']))
with (DIAG/'诊断链路准备核对.txt').open('x',encoding='utf8') as f:
    f.write('候选筛选至解码框记录链路准备核对\n两台服务器：CPU检查通过，未新增真实检测器前向或训练轮数。\n实际原生_get_decoder_input、_select_topk与_generate_anchors方法在合成工程夹具内执行；批大小1/4的记录不改变输出/RNG，分类头和回归头各调用一次。\n新记录：8400位置分类分数、未回归锚点logit、原生实际300索引、原生选中锚点与既有回归头300偏移量；按同一批图像编号连接已有初始框/首层框/最终框。\n未回归锚点不是8400个预测框，位置邻域也不是IoU框覆盖。\n独立第12轮EMA回放流程已准备；真实权重回放、全部548张图像与原生GT标记/评价统计的逐项一致性仍待完成。冻结训练和原验证程序均未改动。\n随机原始Stem当前完成2轮，两种子相同第2轮对比的正负号不同；这些不是第12轮终点或因果证据。\n')
print(json.dumps(dict(status='PROGRESS_INDEPENDENT_DIAGNOSTIC_PREPARATION_VERIFIED',same_epoch_pairs=pairs),ensure_ascii=False))
