"""Record actual deployment, six current process handles and the source-bound endpoint readers."""
from pathlib import Path
import datetime,hashlib,json
ROOT=Path.cwd();read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)
ip=ROOT/'outputs/user_primary_collection_progress12_20261011_v1_inputs.json';inputs=read(ip);obs={}
for family,seeds in inputs.items():
    obs[family]={}
    for seed,v in seeds.items():
        raw=v['payload'].encode();assert len(raw)==v['transport']['bytes'] and hashlib.sha256(raw).hexdigest()==v['transport']['sha256']
        obs[family][seed]=json.loads(raw)
sources=read(ROOT/'outputs/user_primary_collection12_20261011_v1_source_seal.json')
code=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'
assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==sources['files']
cpu=read(ROOT/'outputs/user_primary_collection12_20261011_v1_CPU_review.json')
assert cpu['captured_initialization_positive_cases']==6 and cpu['negative_cases']==150
assert sources['CPU_review_sha256']==sha(ROOT/'outputs/user_primary_collection12_20261011_v1_CPU_review.json')
analysis=read(ROOT/'outputs/user_primary_analysis12_20261011_v1_source_review.json')
for p,h in analysis['sources'].items():assert sha(ROOT/p)==h,p
assert analysis['actual_scientific_endpoints_verified']==0
for s in ('1','2'):
    u,d,a=obs['observations'][s],obs['deployment'][s],obs['actual_reader_field_audit'][s]
    assert u['code_SHA_verified'] and u['queue']['same_live'] and not u['failures']
    assert d['frozen_user28_sources_verified'] and a['user28_sources_SHA_verified'] and a['remote_sources_SHA_verified']
    for arm,j in u['jobs'].items():
        assert j['trainer']['same_live'] and j['controller']['same_live'] and j['failure'] is None
        assert j['first100']['all100_rows_exact'] and j['full_preflight']['identity_matches_receipt']
        assert j['runtime_start']['identity_equals_receipt'] and j['runtime_start']['optimizer_coverage']['all_parameters_exactly_once']
        worker=d['jobs'][arm];audit=a['jobs'][arm]
        assert worker['same_live'] and worker['start_receipt_exists'] and worker['source_SHA_verified'] and not worker['failure_exists']
        assert audit['waiter']['same_live'] and audit['failure'] is None and not audit['collection_complete']
        assert worker['pid']==audit['waiter']['pid'] and worker['start_ticks']==audit['waiter']['start_ticks']
        assert worker['existing_trainer_pid']==j['trainer']['child_pid']
        assert audit['resolved_config_exact'] and audit['initialization_exact'] and audit['runtime_optimizer_exact'] and audit['full_AMP_preflight_exact']
dest=ROOT/'outputs/user_primary_collection_progress12_20261011_v1';dest.mkdir(exist_ok=False)
for f,seeds in inputs.items():
    for s,v in seeds.items():(dest/f'{f}_seed{s}.json').write_bytes(v['payload'].encode())
write(dest/'transport_proofs.json',{f:{s:v['transport'] for s,v in seeds.items()} for f,seeds in inputs.items()})
counts={s:{a:j['completed_epochs'] for a,j in u['jobs'].items()} for s,u in obs['observations'].items()}
jobs={}
for s,u in obs['observations'].items():
    jobs[s]={}
    for arm,j in u['jobs'].items():
        stats=j['latest_stats'];jobs[s][arm]=dict(j,state='training',AP_points=100*stats[0] if stats else None,
            APs_points=100*stats[3] if stats else None,compared_to_cached_BASE12=False,fixed12_efficacy_not_claimed=True)
report=dict(status='USER_SIX_PRIMARY_COLLECTION_WAITERS_DEPLOYED_LIVE_REAL_READER_FIELDS_VALIDATED',
    local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_goal_turn='progress',
    current_goal_turn='progress',no_progress_streak=0,goal_complete=False,
    completed_epochs=counts,jobs=jobs,remote_training_observed_UTC={s:u['UTC'] for s,u in obs['observations'].items()},
    actual_collectors_verified_live_UTC={s:a['UTC'] for s,a in obs['actual_reader_field_audit'].items()},
    primary_collector_processes={s:{a:j['waiter'] for a,j in x['jobs'].items()} for s,x in obs['actual_reader_field_audit'].items()},
    source_seal_sha256=sha(ROOT/'outputs/user_primary_collection12_20261011_v1_source_seal.json'),
    CPU_fixture_review_sha256=sha(ROOT/'outputs/user_primary_collection12_20261011_v1_CPU_review.json'),
    actual_user_current_provenance_field_checks_passed=6,engineering_negative_cases=150,
    analysis_source_review_sha256=sha(ROOT/'outputs/user_primary_analysis12_20261011_v1_source_review.json'),
    user_primary_packages_collected=0,actual_user_scientific_endpoints_analyzed=0,
    independent_saved_EMA_and_formal_terminal_readers_pending=True,
    changed_next_action='Collect each existing user arm immediately after its exact trainer/controller positively finish12; then download and verify actual files and548 stages before candidate-specific GT, class/IoU and exact native all-area PR accounting. Preserve the separate independent EMA gate.',
    attention_causal_scope='Five architecture+initialization replacements, native10,231,997 to14,030,003 parameters; cannot be interpreted as a capacity-matched pure attention effect.',
    intermediate_APs_or_AP_not_a_fixed12_criterion=True,new_training_commands=0,new_BASE_forwards=0,
    new_model_forwards_for_this_pipeline=0,new_evaluator_calls_for_this_pipeline=0,new_optimizer_updates_for_this_pipeline=0,
    existing_training_process_signals=0,configuration_training_retry=False,root_proven=False,novelty_proven=False,
    observed_read_failure='Initial oversized34KiB provenance capture was truncated in transport and rejected by JSON decoding; only a smaller read-only capture was repeated. No experiment/collection/training restart.',
    predeployment_reader_field_fix='Removed a nonexistent added_grad_finite producer field; require actual finite_gradients and all three P2 branch proof fields, validated on all six current remote artifacts.')
write(dest/'report.json',report)
lines=['原始三项实验：终点采集与分析流程审查、部署', '',
    '两台服务器各3个已有配置持续训练。未启动新训练，也未重跑基线。',
    '新增6个CPU采集等待进程，PID及启动时间已实测存活，并绑定实际训练和控制进程。',
    '采集只在两类进程都结束且原生精确12轮正完成后执行一次。保留日志、last.pth、548张阶段数组、原生PR与全面积匹配，独立EMA核验仍是另一项要求。',
    '6组初始化核对、150项拒绝反例通过；另在两个服务器实际核对全部6组配置、预训练状态、优化器和AMP预检查字段。流程检查不代表AP提升。','']
for s,js in jobs.items():
    lines.append('种子%s，远程观察%s'%(s,report['remote_training_observed_UTC'][s]))
    for arm,j in js.items():lines.append('%s：%d/12；当前AP %.4f / APs %.4f。'%(arm,j['completed_epochs'],j['AP_points'],j['APs_points']))
lines+=['','注意力替换同时增加约379.8万个参数和80份原始状态的重新初始化；不能把其结果写成纯注意力机制的收益。P2新增17440参数；SiLU保留794个原状态，参数量不增。',
    '后续依每个配置的真实固定12轮产物分析覆盖、定位与真类别排序。全面积原生PR可精确重建；小GT前缀是全面积匹配的子集，不能冒充原生小面积APs前缀。',
    '本轮无新增终点AP结论、唯一根因或创新成立声明。目标继续。']
(dest/'原始实验采集审查与进度.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
write(dest/'manifest.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()},source_sha256=sha(__file__),input_sha256=sha(ip)))
evidence='../user_primary_collection_progress12_20261011_v1/report.json'
collection=ROOT/'outputs/user_backbone_primary_collection12_20261011_v1';collection.mkdir(exist_ok=False)
write(collection/'execution_state_v1.json',dict(status='SIX_READ_ONLY_PRIMARY_WAITERS_DEPLOYED_CURRENT_HANDLES_LIVE',
    evidence=evidence,CPU_review='../user_primary_collection12_20261011_v1_CPU_review.json',
    source_seal='../user_primary_collection12_20261011_v1_source_seal.json',
    actual_live_collectors=report['primary_collector_processes'],actual_packages_collected=0,
    actual_scientific_endpoints_verified=0,goal_complete=False,independent_saved_EMA_pending=True))
def advance(directory,v,changes):
    folder=ROOT/'outputs'/directory;old=folder/f'execution_state_v{v}.json';value=read(old)
    value.update(changes);value.update(supersedes=old.name,goal_complete=False);write(folder/f'execution_state_v{v+1}.json',value)
advance('user_backbone12_20261011_v1',13,dict(status='ALL_SIX_ORIGINAL_USER_TRAINERS_LIVE_AT6_10_PRIMARY_COLLECTION_READY',evidence=evidence,
    saved_local_UTC=report['local_saved_UTC'],completed_epochs=counts,jobs=jobs,
    last_remote_observation_UTC=report['remote_training_observed_UTC'],queue_processes={s:u['queue'] for s,u in obs['observations'].items()},
    queue_phases={s:u['live']['phase'] for s,u in obs['observations'].items()},
    read_only_primary_collection='../user_backbone_primary_collection12_20261011_v1/execution_state_v1.json',
    primary_stage_and_native_order_readers_source_review='../user_primary_analysis12_20261011_v1_source_review.json',
    fixed12_efficacy_not_yet_claimed=True))
advance('stem_component12_20261011_v1',36,dict(latest_compact_status=evidence,previous_goal_turn='progress',current_goal_turn='progress',
    no_progress_streak=0,standalone_user_execution='../user_backbone12_20261011_v1/execution_state_v14.json',
    standalone_user_completed_epochs=counts,original_user_primary_collection='../user_backbone_primary_collection12_20261011_v1/execution_state_v1.json',
    existing_process_signals_this_turn=0,old_training_process_signals_this_turn=0,
    original_user_primary_collection_workers_verified_live_both_seeds=True,
    interpretation='All six original user training handles observed live: SiLU/P2 at6/10, attention1/5. Six separately source-bound CPU primary collectors deployed and current receipt/config/initialization/optimizer/AMP fields validated. Exact12 primary stage/native-order readers prepared but not run without real endpoint packets. No new training/BASE/model/evaluator calls for this pipeline. Independent EMA remains pending; attention changes architecture+initialization and adds37.12% parameters.'))
advance('original_stem12_20261011_v1',39,dict(authoritative_state='../stem_component12_20261011_v1/execution_state_v37.json'))
print(json.dumps(dict(status=report['status'],counts=counts,live_collectors=6,actual_user_endpoints=0,goal_complete=False),ensure_ascii=False))
