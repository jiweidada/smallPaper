from pathlib import Path
import argparse, base64, datetime, hashlib, json

p = argparse.ArgumentParser()
p.add_argument('--input-base64')
p.add_argument('--input-json')
a = p.parse_args()
assert bool(a.input_base64) != bool(a.input_json)
inputs = json.loads(Path(a.input_json).read_bytes()) if a.input_json else json.loads(base64.b64decode(a.input_base64))
root = Path('outputs/user_change_review_20261011_v1')
assert not root.exists(), 'Preserve previous evidence; use a new version.'
obs = {}
for family, seeds in inputs['observations'].items():
    obs[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf-8')
        meta = item['transport']
        assert len(raw) == meta.get('bytes', meta.get('expected_bytes'))
        assert hashlib.sha256(raw).hexdigest() == meta.get('sha256', meta.get('expected_sha256'))
        value = json.loads(raw)
        assert value['seed'] == int(seed)
        obs[family][seed] = value
for seed in ('1', '2'):
    assert obs['CPU'][seed]['code_SHA_verified'] and not obs['CPU'][seed]['failure']
    assert obs['queue'][seed]['code_SHA_verified'] and not obs['queue'][seed]['failures']
    for arm, job in obs['CPU'][seed]['jobs'].items():
        assert job['cpu']['status'] == 'PASS_STRICT_SHARED_INITIALIZATION_AND_ISOLATED_NEW_RNG'
        assert job['cpu']['optimizer_updates'] == job['cpu']['training_epochs'] == 0
    assert obs['reference'][seed]['code_sha256_verified'] and not obs['reference'][seed]['failures']
    assert obs['component'][seed]['code_SHA_verified']
    assert all(j['failure'] is None for j in obs['component'][seed]['jobs'].values())
    assert obs['warm'][seed]['source_SHA_verified'] and not obs['warm'][seed]['failure']

code = Path('英文小论文/code/user_backbone12_20261011_v1')
actual = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in code.iterdir() if f.is_file()}
assert actual == inputs['manifest'] and len(actual) == 28
native = inputs['native_source']
assert native['source_sha256'] == '5e107a7b387bcdffab8f834b9c9d5df3e4c848079d62eda5bf2a0c22088bb0da'
history_path = Path('英文小论文/data/01_C主线_已完成/C正式配对结果_远端.json')
assert hashlib.sha256(history_path.read_bytes()).hexdigest() == 'bb6a0d6cf2148320e455c5b949e24c926601ddfee6ebb55d0c13be945db2a08e'
history = json.loads(history_path.read_bytes())
base_n = 10231997
rows = []
for arm in ('backbone_silu', 'p2_skip', 'backbone_attention'):
    init = obs['CPU']['1']['jobs'][arm]['cpu']['initialization']
    other = obs['CPU']['2']['jobs'][arm]['cpu']['initialization']
    assert init == other
    rows.append(dict(arm=arm, initialization=init,
        parameter_increase_percent=100 * init['extra_parameters'] / base_n,
        completed_epochs={s: o['jobs'][arm]['completed_epochs'] for s, o in obs['queue'].items()}))

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

root.mkdir()
for family, seeds in inputs['observations'].items():
    for seed, item in seeds.items():
        (root / f'{family}_seed{seed}.json').write_bytes(item['payload'].encode('utf-8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in v.items()} for f, v in inputs['observations'].items()})
write(root / 'review.json', dict(status='DESIGN_REVIEW_FINISHED_CURRENT_CONFIG_EFFECTS_UNPROVEN',
    local_saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    observation_times={f: {s: v.get('UTC', v.get('observed_remote_UTC')) for s, v in vs.items()} for f, vs in obs.items()},
    code_manifest=actual, native_aggregation_source=native, rows=rows,
    attention_scope='Joint change of five aggregation projections, four stage gate types, capacity and aggregation initialization. Net configuration effect is testable; attention-only causality is not identified.',
    P2_scope='Historical width16 P2->P3 bridge before downstream stages; three decoder levels retained. Pixel unshuffle alone is a permutation; compression and nonlinear reduction are not lossless.',
    SiLU_scope='Thirty backbone activations changed, including Stem. All794 common states kept and no extra parameters. Loading unchanged weights does not preserve the ReLU pretrained function.',
    warm_scope='Preserves initial native Stem output but adds a complete EnhancedStem branch and zero channel gains. Initial branch parameter gradients are gated; this is a new candidate rather than a pure initialization control.',
    fixed12_scope='Use the authorized fixed12 gate to reject the current configuration if it fails. A rejection does not negate historical96/320-epoch gains or disprove every implementation of the mechanism.',
    P2_historical_stage=history['stage'], P2_historical_delta_points={split: {k: 100 * values[k] for k in ('AP', 'APs')} for split, values in history['candidate_minus_baseline'].items()},
    review_new_detector_forwards=0, review_new_optimizer_updates=0, BASE_training_added=0,
    frozen_deployed_code_changed=False, existing_process_signals=0, goal_complete=False))

report = '''当前改进的代码与对照设计审查（2026-10-11）

判断：这些改动可以做效果筛选，但目前还不能完整解释“为什么提升”。最明显的归因问题在主干注意力组；P2 的问题假设更具体，SiLU 的对照更干净。

1. 主干注意力组：实际改动超过了注意力公式。
当前 S/B0 基线对应的聚合分支虽然名为 se，实际代码是两层 1×1 ConvBNAct，没有标准 SE 的全局池化与乘法门控。用户历史配置换成单层投影加 ESE / SimAM / MSCA_Lite / MSCA_Dilated，当前移植覆盖五个聚合模块。
总参数从 10,231,997 增至 14,030,003，增加 3,798,006（约 37.12%）。80 个原聚合状态不复用，新增 90 个状态，其余 714 个状态严格保留。因此它同时改变了投影结构、容量、门控方式和聚合初始化。
当前实验回答“用户完整聚合配置在统一12轮条件下是否有效”。无论涨跌，都不能单独归因于注意力公式。
归因设计应先比较同一投影结构的无门控组与门控组，共有投影权重从同一初始化快照复制；如要判断空间分配是否起作用，再保持注意力计算与参数量，将空间权重替换为每通道空间均值广播。上述新增归因组本次未启动。
理论风险：平方偏离均值的权重看的是特征离群程度，亮边、纹理和噪声也可能离群。MSCA_Lite 对池化后的多尺度特征计算权重，可能弱化小目标的局部差异；P5 的大范围上下文也无法凭空恢复早期已丢失的细节。这些是待检验假设，不能当作已确认失败原因。

2. SiLU：适合先做独立筛选，但不能提前写成小目标专用创新。
只替换主干内30处 ReLU（包含 Stem），全部794个原始状态保留，不增加参数。这是目前三组中更干净的功能替换。
SiLU 保留部分负响应并改变梯度，但“弱响应”不等于“小目标响应”。它也可能只改善优化。保持原预训练权重并不等于保持原函数，激活与 BN 后的分布会变化，12轮测试包含适应过程。
若有提升，需区分是否主要发生在小目标、是否改善定位还是分类，以及是否伴随背景误检增加。

3. P2：有具体问题假设，也找到了用户历史正收益。
本地原实现是 P2 64→16通道压缩、pixel_unshuffle×2、零初始化1×1恢复到256通道，再加到P3。加入点在后续P4/P5阶段之前，所以会影响三层特征；Decoder 仍为三个层级，没有新增第四层。
增加17,440参数（约0.17045%），全部794个原始状态保留。零初始化恢复卷积使初始分支输出为零；首步恢复卷积可以得到梯度，上游压缩分支的参数梯度初始被零恢复权重挡住，之后恢复权重学动才打开。
pixel_unshuffle 本身只重排张量，但整个分支含64→16压缩和ReLU，不能称为无损保留原P2信息。细节也可能带来背景纹理。
历史96轮、单seed0：val AP 26.5654→27.2228（+0.6573点），APs 17.6512→18.5968（+0.9457点）；test-dev AP +0.6308点，APs +0.8338点。这个历史结果支持重验，不能混为当前双种子12轮结果，也不能证明收益只由小目标细节造成。
需要检验的路径是：细节旁路是否改善小目标的可用特征→是否增加正确候选而非背景候选→最终是否减少漏检并提升APs。现有初始框是选出的300个回归框，不能直接代表所有8400个候选的覆盖情况。

4. Stem：结构效应和初始化效应必须区分。
直接增强Stem替换丢掉原Stem预训练状态，与保留预训练Stem的基线相比不能只归因于结构。用户授权的随机原始Stem对照用于比较相近初始化条件。
保留原Stem并联增强分支的 Warm 方案初始输出保持原Stem，但它改变了容量、双路结构和门控学习过程，并非纯初始化对照。零增益下增强分支首步梯度被挡住，不能凭增益大小判断有效性。

5. 12轮的结论边界。
遵守已约定的第12轮固定EMA、同seed缓存基线比较：当前配置12轮不提升就淘汰，不延长救结果。这里筛选的是当前配方下的早期收益，不能据此宣布某类机制永远无效，也不能覆盖用户已有96/320轮历史结果。
同样不能把最佳轮次替代第12轮，不能用单seed偶然上涨作为两个seed都通过的结果。

本次审查未改写冻结实验代码，没有新增基线训练或前向，没有发出停止现有训练的信号。新的注意力、SiLU、P2双种子CPU初始化检查全部通过；最近远程快照中三项各为0轮，均在等待GPU容量。CPU通过只证明初始化和结构准备检查通过，完整GPU预检及训练结果仍未产生。
'''
with (root / '改进方案审查.txt').open('x', encoding='utf-8') as f:
    f.write(report)

def updated(old, new, changes):
    v = json.loads(Path(old).read_bytes())
    v.update(changes)
    v['supersedes'] = Path(old).name
    v['goal_complete'] = False
    write(new, v)

evidence = '../user_change_review_20261011_v1'
updated('outputs/native_stem_scratch12_20261011_v3/execution_state_v10.json',
        'outputs/native_stem_scratch12_20261011_v3/execution_state_v11.json',
        dict(status='SEED2_NATIVE_TRAINING_EXACT12_COMPLETE_SEED1_ACTIVE_CLOSEOUT_PENDING',
             evidence=evidence + '/reference_seed1/2.json',
             completed_epochs={s: v['completed_epochs'] for s, v in obs['reference'].items()},
             optimizer_updates={s: v['runtime']['training_audit_live.json']['actual_optimizer_steps'] for s, v in obs['reference'].items()},
             processes={s: v['processes'] for s, v in obs['reference'].items()}))
updated('outputs/warm_stem12_20261011_v1/execution_state_v3.json',
        'outputs/warm_stem12_20261011_v1/execution_state_v4.json',
        dict(status='SEED2_TRAINING_SEED1_RESOURCE_WAIT', evidence=evidence + '/warm_seed1/2.json',
             actual_new_training_epochs={s: v['completed_epochs'] for s, v in obs['warm'].items()},
             full_detector_preflight_passed_by_seed={s: bool(v['detector_preflight']) for s, v in obs['warm'].items()},
             full_detector_preflight_passed=all(bool(v['detector_preflight']) for v in obs['warm'].values())))
updated('outputs/stem_component12_20261011_v1/execution_state_v22.json',
        'outputs/stem_component12_20261011_v1/execution_state_v23.json',
        dict(status='ACTIVE_STEM_ATTRIBUTION_AND_QUEUED_ORIGINAL_USER_BACKBONE_CONFIGS',
             current_goal_turn='progress', no_progress_streak=0,
             reference_state='../native_stem_scratch12_20261011_v3/execution_state_v11.json',
             warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v4.json',
             latest_compact_status=evidence + '/component_seed1/2.json',
             user_backbone_review=evidence + '/review.json',
             standalone_user_cpu_preparation_passed_both_seeds=True,
             standalone_user_completed_epochs={r['arm']: r['completed_epochs'] for r in rows},
             standalone_user_queue_phases={s: v['live']['phase'] for s, v in obs['queue'].items()},
             source28_verified_both_servers=True, attention_only_causality_identified=False))
updated('outputs/original_stem12_20261011_v1/execution_state_v25.json',
        'outputs/original_stem12_20261011_v1/execution_state_v26.json',
        dict(status='ACTIVE_STEM_ATTRIBUTION_AND_QUEUED_ORIGINAL_USER_BACKBONE_CONFIGS',
             authoritative_state='../stem_component12_20261011_v1/execution_state_v23.json'))
write(Path('outputs/user_backbone12_20261011_v1/execution_state_v1.json'),
      dict(status='BOTH_SEEDS_THREE_CPU_BUNDLES_PASS_GPU_CAPACITY_QUEUE', evidence=evidence + '/review.json',
           completed_epochs={r['arm']: r['completed_epochs'] for r in rows},
           full_GPU_preflight_and_training_pending=True, BASE_training_added=0, source_files=28,
           frozen_source_modified_by_review=False, goal_complete=False))
print(json.dumps(dict(status='VERIFIED_REVIEW_AND_STATE_SAVED', report=str(root / '改进方案审查.txt'),
                     source_files=28, attention_added_parameters=3798006,
                     attention_increase_percent=100 * 3798006 / base_n,
                     current_new_epochs={r['arm']: r['completed_epochs'] for r in rows}), ensure_ascii=True))
