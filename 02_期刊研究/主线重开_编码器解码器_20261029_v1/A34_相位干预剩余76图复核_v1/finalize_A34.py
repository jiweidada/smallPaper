from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parent
B = D.parent
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
REPORT = B / '78_A34相位解释复核否定与当前主线状态.txt'
RECEIPT = B / '79_A34剩余76图独立复核回执.json'
CAN = ['01_候选判别表.json', '13_架构检索增补记录.json',
       '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json',
       '16_最近工作源码核查.json']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


assert not REPORT.exists() and not RECEIPT.exists()
prior_path = B / '77_A33原生相位实验与独立核验回执.json'
prior = read(prior_path)
for name, h in prior['files_sha256'].items():
    assert sha(name) == h, name
assert len(prior['files_sha256']) == 243
cfg = read(D / 'phase_protocol_FROZEN.json')
inputs = read(D / 'execution_inputs_manifest.json')
for name, key in [('native_phase_worker.py', 'worker_sha256'),
                  ('phase_protocol_FROZEN.json', 'protocol_sha256'),
                  ('launch_native_once.py', 'launcher_sha256'),
                  ('analyze_native_phase.py', 'analysis_sha256')]:
    assert sha(D / name) == inputs[key], name
r = read(D / 'native_phase_analysis_result.json')
av = read(D / 'archive_verification.json')
iv = read(D / 'independent_primary_verification.json')
terminal = read(D / 'remote_terminal_observation.json')
complete = read(D / 'remote_complete.json')
assert r['images'] == 76 and r['groups']['small_GT'] == 3402
assert av['manifest_files_verified'] == 554
assert iv['max_contrast_error'] == 0 and iv['scalar_GT_hit_checks'] == 190512
assert complete['counts'] == dict(whole_detector_forwards=533, extra_native_O_decodes=532,
                                  all_native_head_decodes=1065, optimizer_updates=0,
                                  standalone_TAL_calls=0)
assert terminal['pid'] == 14842 and terminal['expected_start_ticks'] == '101897935'
assert not terminal['same_process_live'] and terminal['exception'] is None
assert complete['zero_fidelity'] and complete['repeat_fidelity'] and complete['weights_unchanged']
assert not r['mechanism_question_promotion'] and sum(r['gates'].values()) == 2
assert r['contrasts']['O']['paired_reestimated_FP_bootstrap95'][1] < 0
state = read(B / CAN[3])
assert state['model_forwards'] == 2635 and state['training_updates'] == 528
assert state['current_goal_turn']['phase'] == 'A33'
matrix = read(B / CAN[2])
assert len(matrix['rows']) == 106

backup = D / 'canonical_before_A34'
backup.mkdir(exist_ok=False)
for name in CAN:
    shutil.copy2(B / name, backup / name)
remapped = {}
for name, h in prior['files_sha256'].items():
    p = Path(name)
    target = backup / p.name if p.parent == B and p.name in CAN else p
    assert sha(target) == h
    remapped[str(target)] = h

question = ('当前无已确认主线。A34未复现“粗格共占导致邻近小目标额外混合”的方向性解释，'
            '按冻结规则停止延长该相位—邻近实例分支。继续开放编码器、特征交互和检测解码器，'
            '先找同一任务上可复现的具体失效及与最近方法可区分的算子差异；'
            '尚未选定新的具体研究问题或新算法，下一模型调用分配为0。')
phase = dict(
    phase='A34', classification='progress_frozen_remaining_cohort_replication_rejects_directional_hypothesis',
    recorded_utc=NOW, status='REPLICATION_COMPLETE_DIRECTIONAL_HYPOTHESIS_NOT_SUPPORTED',
    goal_status='active', mainline_validated=False, complete_mainline=False,
    report=str(REPORT), receipt=str(RECEIPT),
    new_whole_detector_forwards=533, new_GPU_model_calls=533,
    new_extra_native_O_decodes=532, new_all_native_head_decodes=1065,
    new_optimizer_updates=0, new_standalone_TAL_calls=0, new_complete_decoder_calls=0,
    new_training_jobs=0, images=76, input_small_GT=3402, arms=14, matched_FP_budget=703,
    bootstrap_draws=1000, mechanism_question_promotion=False,
    directional_signature_replicated=False, phase_neighbor_branch_continuation=False,
    general_phase_instability_observed=True, causal_operator_identified=False,
    research_gap_confirmed=False, new_algorithm_approved=None,
    full_model_forwards=3168, actual_cumulative_detector_optimizer_updates=528,
    formal_detector_training_updates=526, historical_capacity_updates=2,
    cumulative_complete_decoder_only_calls=48, historical_last_layer_suffix_calls=72,
    core_matrix_rows=106, new_core_papers=0, new_full_PDFs=0, candidate_count=16,
    prior_A33_files_verified=243, A34_cap=533, A34_cap_closed=True,
    A33_169_cap_remains_closed=True, A30_101_cap_remains_closed=True,
    historical_39092_cap_remains_closed=True, worker_pid=14842,
    worker_start_ticks='101897935', worker_process_observed_absent=True,
    worker_exit_code_observed=None, launcher_exit_code=0,
    automatic_followups=False, next_model_call_allocation=0, A22_restarted=False,
    new_heldout_dataset_confirmation=False, next=question)

lines = [
    'A34：剩余76图相位解释复核与当前主线状态', '记录时间（UTC）：' + NOW, '',
    '结论：尚未找到经验证可支撑小论文的研究主线。当前具体的“粗格共占导致邻近小目标额外混合”解释未通过复核。'
    '按冻结协议停止继续扩展这一分支；不能把一般相位敏感或相反方向的结果改写为新创新点。', '',
    '1. 检索策略与检索记录',
    '本轮是预先固定的诊断复核，不开展新的全面文献检索，不增加核心文献或宣称新颖性已排除。'
    '前期检索、原文范围与源码记录继续见13、14、16号文件及74、76号报告。'
    '另保留4条相位相关发现检索及TIPS、CROWn、Reviving Shift Equivariance线索，'
    '仅摘要/检索片段范围，未读完整方法，不加入核心矩阵，不据此主张创新空白。', '',
    '2. 核心文献与阅读范围',
    '沿用A33已经登记的AeroPinWorld、LPS、等变ViT等阅读范围；本轮无新增全文阅读。'
    'AeroPinWorld阅读方法3.1–3.5及已登记选段；LPS/等变ViT沿用各自已登记方法选段。'
    '不把这些阅读范围表述为全文、作者代码复现或人类已读。', '',
    '3. 与最近工作的比较边界',
    '相位选择、相位索引传递、偏移探测和解码恢复已有直接先例。'
    '本轮只测试现有YOLO26s的输出是否符合一个具体失效解释，未实现新编码器或新解码器。'
    '通用增加细节分支、相位打包或替换编码器尚不能作为本文创新。', '',
    '4. 观察、推断与待验证假设',
    '【预设问题】在保持真实像素、目标尺寸、遮挡及相对布局不变时，'
    '与采样网格的相对位置变化是否对粗格中邻近实例产生额外、方向一致的检测损害？',
    '【设计】原100张开发图中，A33先用固定文件名哈希顺序的24张；本轮仅用余下76张。'
    '它们此前已有开发诊断用途，因此是剩余队列的相位干预复核，不是全新盲测数据。'
    '不合并100图来覆盖本轮失败，不依据模型结果挑图、位移、指标或阈值。',
    '【干预】同一D0任务适配YOLO26s，640输入，已有填充内y方向0、±8、±16、±32整数平移。'
    'RGB仅缩放一次；真实像素及逆变换逐条件严格相等。绝对位置和边界距离仍变化，不能分离纯混叠效应。',
    '【分组】输入面积严格小于1024的小目标3402个；拆开组398个/53图，合并组374个/50图。'
    '粗格16/细格8只是名义几何标签，不是实际感受野或单token容量的证据。',
    '【指标】各原生分支和位移使用同FP预算703，采用全局分数前缀及同分整块处理。'
    '1000次配对图像重抽样，每次重新估计各条件全局阈值。'
    '采用源码对齐忽略区域预处理＋COCO101@500，不称完整官方VisDrone评分复现。',
    '【主要对比】拆开组的检出变化减合并组的检出变化；变化定义为±8相对±32的对称均值。'
    '以下数值是方向性诊断的百分点差，不是新算法的AP提升：'
]
for k, label in [('M', '常规分支＋NMS'), ('O', '单次预测分支')]:
    z = r['contrasts'][k]
    ci = z['paired_reestimated_FP_bootstrap95']
    fi = z['flip8_minus_flip32_bootstrap95']
    lines.append(f"{label}：主要对比{100*z['separation_minus_merge_contrast']:+.4f}个百分点，"
                 f"95%区间[{100*ci[0]:+.4f}, {100*ci[1]:+.4f}]；"
                 f"±8相对±32的额外检出翻转{100*z['flip8_minus_flip32']:.4f}个百分点，"
                 f"区间[{100*fi[0]:.4f}, {100*fi[1]:.4f}]。")
lines += [
    '【复核结果】5项事前条件仅样本支持和一般相位翻转条件通过；'
    '单次预测方向性效应≥2个百分点、其区间下界>0、常规分支同方向这3项均失败。'
    '单次预测方向与原解释相反；不能据此证明全部邻近目标无干扰，也不能反向发明机制。',
    '【与A33分开】A33的24图单次预测对比+7.8273个百分点，区间[-3.4581,+16.2502]；'
    'A34作为复核判定依据。不能改用合并100图或GT预知最佳位移挽救原假设。',
    '【仍成立的观察】网格相对位置变化伴随检出翻转。'
    '【尚未成立的推断】某一编码器算子造成邻近实例混合。'
    '【尚不存在的结果】新算法、经核实的新颖性、检测精度改善及新编码器的24GB训练可行性。', '',
    '5. 候选增量、可行性与风险',
    '本轮不批准算法增量。“小目标实例信息保留与实例级读出”仍是研究范围，尚未收敛为可立项问题。'
    'YOLO26s是现有诊断参照；新视觉编码器/检测解码器尚未选定。'
    '一般相位敏感已有前人研究且算子归因未闭合，因此不足以独立支撑论文。', '',
    '6. 最小验证与执行情况',
    '已完成冻结的76×7＋1次重复＝533次完整前向，零训练。'
    '额外原生O解码532次、全部原生头解码1065次、TAL独立调用0次；533上限封闭。',
    '归档582095382字节，555项，554项内部清单哈希全部通过；'
    '零位移456个原始/解码数组及152个逐图原生预测记录与A30逐项相等；532组像素保真通过。',
    '独立标量实现复算14条件、190512个GT事件及原样本和3个重抽样统计；1000次种子序列一致，主要统计最大误差0。',
    'PID14842/start_ticks101897935已观察到不存在，status为COMPLETED，无exception。'
    '启动器退出码0；worker退出码未单独持久化，不混称worker退出码0。'
    '下一模型调用配额0，无自动续跑；旧A22、旧DEIM和压缩分支均未重启。',
    '累计架构完整前向3168、优化器更新528、完整DETR解码器重放48。'
    '本次仅证明这种推理诊断在现有3090上可执行，不能据此承诺新模型训练资源。', '',
    '7. 参考文献及可核验文件',
    'AeroPinWorld：https://doi.org/10.3390/electronics15071364',
    'LPS：https://arxiv.org/abs/2210.08001v1',
    '等变ViT：https://arxiv.org/abs/2305.16316v2',
    '冻结协议：' + str(D / 'phase_protocol_FROZEN.json'),
    '完整结果：' + str(D / 'native_phase_analysis_result.json'),
    '独立核验：' + str(D / 'independent_primary_verification.json'), '',
    '下一步：' + question
]
REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
board = read(B / CAN[0])
board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A34', mainline_validated=False,
             latest_update_utc=NOW, latest_report=str(REPORT), current_priority_candidate=None,
             current_priority_question=question, next_exploratory_question=question,
             A34_evidence=phase,
             A34_selection_note='剩余76图的冻结复核未支持原方向，停止延长相位—邻近实例分支；尚无替代具体主线。')
write(B / CAN[0], board)
search = read(B / CAN[1])
search.update(supplement_utc=NOW, A34_frozen_replication=dict(
    phase=phase, new_core_literature_reading=False,
    discovery_only_search_log=str(D / 'discovery_only_phase_prior_leads.json'),
    primary_verification=str(D / 'independent_primary_verification.json')))
write(B / CAN[1], search)
matrix.update(updated_utc=NOW, latest_report=str(REPORT),
              latest_empirical_boundary='A34:remaining76cohort didnotreplicate neighborseparationdirection;'
              'Ocontrastnegative,primarygatesfailed;generalphaseinstabilityisnotanoperatorcausalornewalgorithmclaim.')
write(B / CAN[2], matrix)
state.update(recorded_utc=NOW, goal_status='active', mainline_validated=False,
             model_forwards=3168, previous_goal_turn=state['current_goal_turn'],
             current_goal_turn=phase, latest_authoritative_phase=phase,
             latest_report=str(REPORT), current_priority_question=question, A34_evidence=phase,
             current_native_job_terminal=dict(phase='A34', pid=14842, start_ticks='101897935',
             observed_absent=True, status='COMPLETED', worker_exit_code=None, cap_closed=True))
write(B / CAN[3], state)
source = read(B / CAN[4])
source.update(updated_utc=NOW, A34_directional_replication=dict(
    phase=phase, frozen_protocol=str(D / 'phase_protocol_FROZEN.json'),
    archive_verification=str(D / 'archive_verification.json'), novel_algorithm_approved=None))
write(B / CAN[4], source)
files = [p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [REPORT] + [B / name for name in CAN]
receipt = dict(phase)
receipt.update(archive_verification=av, independent_primary_verification=iv, scientific_result=r,
               previous_A33_evidence_remapped_sha256=remapped,
               files_sha256={str(p): sha(p) for p in sorted(set(files))})
write(RECEIPT, receipt)
for name, h in receipt['files_sha256'].items():
    assert sha(name) == h, name
for name, h in remapped.items():
    assert sha(name) == h, name
print(json.dumps(dict(status='A34_PUBLISHED_DIRECTIONAL_HYPOTHESIS_NOT_SUPPORTED',
                      evidence_files=len(receipt['files_sha256']), previous_A33_files_verified=243,
                      model_forwards=3168, training_updates=528, mainline_validated=False,
                      report=str(REPORT)), ensure_ascii=False))
