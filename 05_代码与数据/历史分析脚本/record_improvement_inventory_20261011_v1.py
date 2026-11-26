from pathlib import Path
import argparse, base64, datetime, hashlib, json

p = argparse.ArgumentParser()
p.add_argument('--input-base64', required=True)
a = p.parse_args()
inputs = json.loads(base64.b64decode(a.input_base64))
root = Path('outputs/improvement_inventory_20261011_v1')
assert not root.exists(), 'Preserve an existing inventory; create a new version instead.'
verified = {}
for family, seeds in inputs.items():
    verified[family] = {}
    for seed, item in seeds.items():
        raw = item['payload'].encode('utf-8')
        meta = item['transport']
        hs = meta.get('sha256', meta.get('expected_sha256'))
        nb = meta.get('bytes', meta.get('expected_bytes'))
        assert len(raw) == nb and hashlib.sha256(raw).hexdigest() == hs
        obs = json.loads(raw)
        assert obs['seed'] == int(seed)
        if family == 'reference':
            assert obs['code_sha256_verified'] and not obs['failures']
        elif family == 'component':
            assert obs['code_SHA_verified']
            assert all(v['failure'] is None for v in obs['jobs'].values())
        else:
            assert obs['source_SHA_verified'] and obs['dispatcher']['same_live']
            assert not obs['failure']
            if obs['detector_preflight']:
                assert obs['detector_preflight']['status'] == 'PASS_CANDIDATE_FULL_DETECTOR_AMP_LOSS_BACKWARD_NO_UPDATES'
                assert obs['detector_preflight']['identity_equals_current_receipt']
        verified[family][seed] = obs

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

progress_path = Path('outputs/stem_component12_20261011_v1/progress_v27.json')
progress = read(progress_path)
full = {str(seed): next(r for r in progress['rows'] if r['seed'] == seed and r['arm'] == 'original_stem' and r['epoch'] == 12) for seed in (1, 2)}
hist_path = Path('outputs/stem_component12_20261011_v1/historical_stem_recipe_audit_v1.json')
history = read(hist_path)
assert all(history['records'][k]['total_logged_epochs'] == 320 for k in ('old_standard_stem_silu', 'old_standard_stem_msca_relu'))
stopped_path = Path('outputs/support_matching12_20261011_v1/execution_state_v40.json')
stopped = read(stopped_path)
assert stopped['status'] == 'STOPPED_USER_REDIRECT_TO_ORIGINAL_STEM'

root.mkdir()
for family, seeds in inputs.items():
    for seed, item in seeds.items():
        with (root / f'{family}_seed{seed}.json').open('xb') as f:
            f.write(item['payload'].encode('utf-8'))
write(root / 'transport.json', {f: {s: i['transport'] for s, i in v.items()} for f, v in inputs.items()})

rows = [
    dict(id='user_full_enhanced_stem', name='你的原始完整增强 Stem', epochs={'1': 12, '2': 12}, status='NATIVE_TRAINING_EXACT12_COMPLETE_INDEPENDENT_RELOAD_PENDING', source=str(progress_path)),
    dict(id='user_backbone_attention', name='你原来主干的注意力替换', epochs=None, status='NO_CURRENT_STANDALONE_MATCHED12_RESULT', historical_epochs=320, historical_record='old_standard_stem_msca_relu', source=str(hist_path)),
    dict(id='user_activation', name='你原来的激活函数替换 SiLU', epochs=None, status='NO_CURRENT_STANDALONE_MATCHED12_RESULT', historical_epochs=320, historical_record='old_standard_stem_silu', source=str(hist_path)),
    dict(id='user_P2', name='你原来单独的 P2 接入', epochs=None, status='NO_CURRENT_STANDALONE_MATCHED12_RESULT', note='Existing fine/degraded encoder-decoder tests also change queries and decoder reads; they do not isolate the original P2 replacement.'),
]
for arm, label in [('detail_lowres','Stem 细节分支分辨率对照'), ('highfreq_rgb','Stem 高频分支 RGB 输入对照'), ('attention_spatial_mean','Stem 内部空间注意力对照'), ('early_final_downsample','Stem 下采样时机对照')]:
    epochs = {s: o['jobs'][arm]['completed_epochs'] for s, o in verified['component'].items()}
    rows.append(dict(id=arm, name=label, epochs=epochs, status='EXACT12_COMPLETE_RELOAD_PENDING' if all(n == 12 for n in epochs.values()) else 'TRAINING_ACTIVE'))
rows.append(dict(id='native_stem_scratch', name='仅原始 Stem 随机初始化对照', epochs={s:o['completed_epochs'] for s,o in verified['reference'].items()}, status='PARTIALLY_COMPLETE_REFERENCE_ACTIVE', note='Only Stem is random; 754 states outside Stem share the common initialization.'))
rows.append(dict(id='warm_stem', name='保留预训练原始 Stem 并联增强分支', epochs={s:o['completed_epochs'] for s,o in verified['warm'].items()}, phases={s:o['live']['phase'] for s,o in verified['warm'].items()}, status='SEED2_TRAINER_STARTED_SEED1_RESOURCE_WAIT'))
rows += [
    dict(id='fine_and_degraded',name='专用 Encoder/Decoder 和 P2 退化对照',epochs={'1':12,'2':12},status='COMPLETE',source='outputs/small_object_encoder_decoder12_20261010_v5/final_comparison/report.json'),
    dict(id='target_and_random_protection',name='目标感知 Mosaic 保护和随机保护对照',epochs={'1':12,'2':12},status='COMPLETE',source='outputs/scale_protection12_20261010_v1/fixed12_policy_decision_v1.json'),
    dict(id='FDR_tail',name='FDR 回归范围端点扩大',epochs={'1':12,'2':12},status='COMPLETE_REJECTED',source='outputs/fdr_tail12_20261011_v2/two_seed_fixed12_decision_v32.json'),
    dict(id='support_matching',name='回归范围感知匹配',epochs={str(x['seed']):x['stop_verification']['completed_epochs'] for x in stopped['seeds']},status='STOPPED_BY_USER_REDIRECTION',source=str(stopped_path)),
    dict(id='P3_feedback',name='P3 最终检测梯度反馈',epochs={'1':5,'2':7},status='STOPPED_EARLY_BY_USER',source='outputs/sod_problem_reasoning_20261010/反馈实验提前淘汰_结果与证据.txt'),
    dict(id='P3_Haar',name='P3 Haar 小波门控',epochs={'1':7,'2':8},status='INTERRUPTED_NO_FIXED12_ENDPOINT',source='outputs/sod_problem_reasoning_20261010/wavelet_two_seed_progress_delivery_v1.json'),
    dict(id='phase_attention',name='相位注意力',epochs={'1':0,'2':0},status='NO_TRAINING_RESULT'),
    dict(id='manifold',name='流形方向',epochs={'1':0,'2':0},status='DESCRIPTIVE_DIAGNOSTICS_ONLY_NO_TRAINED_MODULE'),
    dict(id='ROI',name='原图 ROI 精修',status='COMPLETE_8_REFINER_PLUS4_GATE_SUBSET_PASSES_PER_ARM',note='This is not12 complete detection training epochs; official accepted modifications0.',source='outputs/sod_roi12_20261009/最终实验结论与交付索引.txt'),
    dict(id='P3_spatial_historical',name='旧 P3 空间读出',epochs={'1':24,'2':24},status='HISTORICAL_DIFFERENT_BUDGET',source='outputs/chat_improvement_review_20261011_v1/改进与测试方向复盘.txt'),
]
sources = [progress_path, hist_path, stopped_path, Path('outputs/chat_improvement_review_20261011_v1/改进与测试方向复盘.txt')]
write(root/'inventory.json',dict(status='NOT_ALL_USER_PROPOSED_IMPROVEMENTS_TESTED',saved_local_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_observation_times={f:{s:o.get('UTC',o.get('observed_remote_UTC')) for s,o in v.items()} for f,v in verified.items()},rows=rows,source_sha256={str(p):sha(p) for p in sources},new_model_forwards=0,new_optimizer_updates=0,baseline_retrained=False,goal_complete=False,claim='Current Stem internal attention control does not validate the user earlier backbone attention replacement. Historical N-scale320 runs are separate from current S-scale matched12 tests.'))

lines=['改进点 12 轮实验核对（2026-10-11，本次远程只读快照）','',
       '结论：没有全部跑完。当前重点跑的是你的增强 Stem 及其归因对照；主干注意力、激活函数和你原来单独 P2 接入，还没有当前统一条件下的独立双种子 12 轮结果。',
       '已有你之前的主干注意力和 SiLU 的 320 轮历史日志，不能说它们从未训练过；那些是 N 规模旧配方，不能当作当前 S 规模的独立归因验证。','',
       '配置 | seed1 完成轮次 | seed2 完成轮次 | 状态']
for r in rows:
    e=r.get('epochs');v1=str(e['1']) if e else '—';v2=str(e['2']) if e else '—'
    lines.append(f"{r['name']} | {v1} | {v2} | {r['status']}")
lines += ['', '新并联候选：seed2 完整检测器 AMP/损失/反向预检通过，实际训练进程已启动，尚未完成首轮；seed1 等资源。',
          '训练日志到第12轮与保存 EMA 的独立重载复核是两个状态；本轮 Stem 完成项的独立重载复核仍待完成。',
          'Stem 内部空间注意力消融不是你之前的主干注意力替换；专用 Encoder/Decoder 也不是单独 P2 接入对照。',
          '已按你的指令提前停止的路线保留原轮次，不补造12轮终点。']
(root/'实验进度与未覆盖项.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

refdir=Path('outputs/native_stem_scratch12_20261011_v3')
rstate=read(refdir/'execution_state_v9.json')
rstate.update(supersedes='execution_state_v9.json',evidence='../improvement_inventory_20261011_v1/reference_seed1/2.json',completed_epochs={s:o['completed_epochs'] for s,o in verified['reference'].items()},latest_live_processes={s:o['processes'] for s,o in verified['reference'].items()},current_goal_turn='progress',no_progress_streak=0)
write(refdir/'execution_state_v10.json',rstate)
warmdir=Path('outputs/warm_stem12_20261011_v1')
wstate=read(warmdir/'execution_state_v2.json')
wstate.update(supersedes='execution_state_v2.json',evidence='../improvement_inventory_20261011_v1/warm_seed1/2.json',phases={s:o['live']['phase'] for s,o in verified['warm'].items()},actual_new_training_epochs={s:o['completed_epochs'] for s,o in verified['warm'].items()},full_detector_preflight_passed_by_seed={s:o['detector_preflight'] is not None for s,o in verified['warm'].items()},trainers={s:o['trainer'] for s,o in verified['warm'].items()},full_detector_preflight_passed=all(o['detector_preflight'] is not None for o in verified['warm'].values()))
write(warmdir/'execution_state_v3.json',wstate)
compdir=Path('outputs/stem_component12_20261011_v1')
cstate=read(compdir/'execution_state_v21.json')
cstate.update(supersedes='execution_state_v21.json',current_goal_turn='progress',no_progress_streak=0,latest_compact_status='../improvement_inventory_20261011_v1/component_seed1/2.json',reference_state='../native_stem_scratch12_20261011_v3/execution_state_v10.json',warm_candidate_state='../warm_stem12_20261011_v1/execution_state_v3.json',user_improvement_inventory='../improvement_inventory_20261011_v1/inventory.json',standalone_user_improvements_still_unverified=['backbone_attention','activation_SiLU','original_P2_connection'],warm_training_started_seeds=[int(s) for s,o in verified['warm'].items() if o['trainer'] and o['trainer']['same_live']],goal_complete=False)
write(compdir/'execution_state_v22.json',cstate)
write('outputs/original_stem12_20261011_v1/execution_state_v25.json',dict(status=cstate['status'],supersedes='execution_state_v24.json',authoritative_state='../stem_component12_20261011_v1/execution_state_v22.json',goal_complete=False,frozen_code_changed=False))
print(json.dumps(dict(status='INVENTORY_AND_FRESH_REMOTE_STATES_SAVED',reference_epochs={s:o['completed_epochs'] for s,o in verified['reference'].items()},warm_phases={s:o['live']['phase'] for s,o in verified['warm'].items()},goal_complete=False)))
