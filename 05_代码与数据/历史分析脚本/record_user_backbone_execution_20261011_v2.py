from pathlib import Path
import datetime, hashlib, json

data = json.loads(Path('outputs/user_backbone_execution_20261011_v2_inputs.json').read_bytes())
root = Path('outputs/user_backbone_execution_20261011_v2')
assert not root.exists()
obs = {}
for family, seeds in data['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf8'); meta = item['transport']
        assert len(raw) == meta.get('bytes', meta.get('expected_bytes'))
        assert hashlib.sha256(raw).hexdigest() == meta.get('sha256', meta.get('expected_sha256'))
        obs[family][seed] = json.loads(raw)
        assert obs[family][seed]['seed'] == int(seed)
for seed in ('1', '2'):
    assert obs['queue'][seed]['code_SHA_verified'] and obs['queue'][seed]['queue']['same_live']
    assert not obs['queue'][seed]['failures']
    monitor = obs['monitor'][seed]
    assert monitor['monitor_same_live'] and monitor['log_tail'] == ''
    assert monitor['monitor_live']['queue_same_live'] and not monitor['monitor_live']['queue_failure']
    assert obs['reference'][seed]['completed_epochs'] == 12 and not obs['reference'][seed]['failures']
    assert obs['warm'][seed]['source_SHA_verified'] and not obs['warm'][seed]['failure']
root.mkdir()
def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
for family, seeds in data['observations'].items():
    for seed, item in seeds.items():
        (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf8'))
write(root / 'transport.json', {f: {s: item['transport'] for s, item in seeds.items()} for f, seeds in data['observations'].items()})
jobs = {s: v['monitor_live']['jobs'] for s, v in obs['monitor'].items()}
summary = dict(status='REMOTE_EXECUTION_QUEUES_AND_RESULT_OBSERVERS_VERIFIED_LIVE',
    saved_local_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    last_remote_observation_UTC={s: v['UTC'] for s, v in obs['monitor'].items()},
    servers={'1': '172.17.120.160:30310', '2': '172.17.120.160:30411'},
    queued_configurations=['backbone_silu', 'p2_skip', 'backbone_attention'], jobs=jobs,
    queue_processes={s: v['queue'] for s, v in obs['queue'].items()},
    observation_processes={s: v['monitor'] for s, v in obs['monitor'].items()},
    completed_epochs={s: {a: v['completed_epochs'] for a, v in items.items()} for s, items in jobs.items()},
    max_epochs_per_configuration=12, cached_BASE_reused=True, new_BASE_epochs=0,
    frozen_model_or_training_code_changed=False, old_training_process_signals=0,
    attention_scope='Net effect of the original user aggregation architecture; not attention-only causal identification.',
    independent_saved_EMA_results_pending=True, automatic_retry=False, goal_complete=False)
write(root / 'execution_summary.json', summary)
lines = ['三项独立远程实验启动状态（2026-10-11）', '',
    '30310 服务器执行 seed1；30411 服务器执行 seed2。每台分别测试 SiLU、原主干聚合/注意力配置和原 P2→P3 接入。',
    '执行队列和只读结果记录进程均已核实存活。三项初始化检查已通过，GPU完整预检和训练需等待容量满足；当前均为0轮，不能声称已经得到精度结果。',
    '每个配置最多12轮，固定第12轮EMA对比同seed已有基线；两seed APs提高且AP不下降才通过。不重跑基线、不延长失败配置、不自动重试。', '',
    '项目 | seed1 | seed2']
for arm, title in [('backbone_silu', '主干SiLU'), ('p2_skip', '原P2旁路'), ('backbone_attention', '原主干聚合/注意力配置')]:
    lines.append(f"{title} | {jobs['1'][arm]['completed_epochs']}/12 等待GPU位置 | {jobs['2'][arm]['completed_epochs']}/12 等待GPU位置")
lines += ['', '现有训练继续：',
    f"Stem空间注意力对照：{obs['component']['1']['jobs']['attention_spatial_mean']['completed_epochs']}/12、{obs['component']['2']['jobs']['attention_spatial_mean']['completed_epochs']}/12。",
    f"Stem下采样时机对照：{obs['component']['1']['jobs']['early_final_downsample']['completed_epochs']}/12、{obs['component']['2']['jobs']['early_final_downsample']['completed_epochs']}/12。",
    f"预训练Stem并联增强候选：{obs['warm']['1']['completed_epochs']}/12、{obs['warm']['2']['completed_epochs']}/12，两端完整GPU预检均通过。",
    '随机原始Stem对照：两seed原生训练均完成12轮，独立保存权重复核仍待GPU空闲。', '',
    '结果记录区分排队、初始化、实际训练和第12轮完成，保留原生AP/APs、与缓存基线的差值和错误日志。训练中数字只表示中间进度；保存EMA复核完成前不冒充最终验证。']
(root / '远程实验状态.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
def update(old, new, changes):
    value = json.loads(Path(old).read_bytes()); value.update(changes)
    value['supersedes'] = Path(old).name; value['goal_complete'] = False
    write(new, value)
evidence = '../user_backbone_execution_20261011_v2'
update('outputs/user_backbone12_20261011_v1/execution_state_v1.json',
       'outputs/user_backbone12_20261011_v1/execution_state_v2.json', summary | {'evidence': evidence + '/execution_summary.json'})
update('outputs/native_stem_scratch12_20261011_v3/execution_state_v11.json',
       'outputs/native_stem_scratch12_20261011_v3/execution_state_v12.json',
       dict(status='BOTH_NATIVE_TRAINING_EXACT12_COMPLETE_SAVED_EMA_CLOSEOUT_PENDING', evidence=evidence + '/reference_seed1/2.json',
       completed_epochs={s: v['completed_epochs'] for s, v in obs['reference'].items()},
       optimizer_updates={s: v['runtime']['training_audit_live.json']['actual_optimizer_steps'] for s, v in obs['reference'].items()},
       processes={s: v['processes'] for s, v in obs['reference'].items()}))
update('outputs/warm_stem12_20261011_v1/execution_state_v4.json',
       'outputs/warm_stem12_20261011_v1/execution_state_v5.json',
       dict(status='BOTH_FULL_DETECTOR_PREFLIGHT_PASS_BOTH_TRAINERS_ACTIVE', evidence=evidence + '/warm_seed1/2.json',
       full_detector_preflight_passed=True, full_detector_preflight_passed_by_seed={'1': True, '2': True},
       actual_new_training_epochs={s: v['completed_epochs'] for s, v in obs['warm'].items()},
       trainers={s: v['trainer'] for s, v in obs['warm'].items()}, phases={s: v['live']['phase'] for s, v in obs['warm'].items()}))
update('outputs/stem_component12_20261011_v1/execution_state_v23.json',
       'outputs/stem_component12_20261011_v1/execution_state_v24.json',
       dict(status='ACTIVE_STEM_ATTRIBUTION_AND_REMOTE_USER_CONFIG_QUEUES', current_goal_turn='progress', no_progress_streak=0,
       reference_state='../native_stem_scratch12_20261011_v3/execution_state_v12.json',
       warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v5.json',
       latest_compact_status=evidence + '/component_seed1/2.json',
       standalone_user_execution=evidence + '/execution_summary.json',
       remote_result_observers_verified_live_both_seeds=True))
update('outputs/original_stem12_20261011_v1/execution_state_v26.json',
       'outputs/original_stem12_20261011_v1/execution_state_v27.json',
       dict(status='ACTIVE_STEM_ATTRIBUTION_AND_REMOTE_USER_CONFIG_QUEUES',
       authoritative_state='../stem_component12_20261011_v1/execution_state_v24.json'))
print(json.dumps(dict(status=summary['status'], queues=2, result_observers=2,
    new_configurations=3, per_seed_epochs=summary['completed_epochs'], BASE_retrained=False)))
