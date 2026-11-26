import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
B = Path(__file__).resolve().parent
D = B / 'A24_最近方法重合与损失数值核查_v1'
E = B.parent
ARCHIVE = E / '文献_相关原始论文_20261004'
OPLOG = E / '前期调研_码率受限航拍小目标_20261028_v1' / '远端验证_20261029_v1' / 'ssh_operation_receipts.jsonl'
REPORT = B / '58_A24最近方法重合与训练数值边界.txt'
RECEIPT = B / '59_A24文献与CPU数值核验回执.json'
NAMES = ['01_候选判别表.json', '13_架构检索增补记录.json', '14_架构核心文献矩阵.json', '15_当前状态与结果校验.json', '16_最近工作源码核查.json']
assert not REPORT.exists() and not RECEIPT.exists()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

ids = ['A24_create_isolated_CPU_evidence_directory', 'A24_upload_CPU_operator_probe_only', 'A24_CPU_native_zero_branch_synthetic_probe', 'A24_terminal_source_integrity_and_checkpoint_inventory']
operations = {}
for line in OPLOG.read_text(encoding='utf-8').splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get('id') in ids and row.get('action') in ['exec', 'upload']:
        if row.get('status') == 'TRANSFER_PROGRESS':
            continue
        operations[row['id']] = row
assert set(operations) == set(ids)
for key in [ids[0], ids[2], ids[3]]:
    assert operations[key]['exit_code'] == 0
assert operations[ids[1]]['status'] == 'UPLOADED'
cpu = json.loads(operations[ids[2]]['stdout'])
integrity = json.loads(operations[ids[3]]['stdout'])
assert cpu['detector_forwards'] == cpu['optimizer_updates'] == 0
assert not cpu['CUDA_initialized']
assert cpu['status'] == 'FINITE_FP16_INPUT_SYNTHETIC_NONFINITE_ZERO_BRANCH_REPRODUCED'
assert integrity['files_checked'] == 22 and not integrity['mismatches']
assert integrity['checkpoint_files'] == integrity['failure_tensor_dump_files'] == []
native = B / 'official_assets_v1' / 'Dome_DETR_2dde3bc_v1' / 'source' / 'src' / 'zoo' / 'dome' / 'dome_criterion.py'
assert sha(native) == cpu['source_sha256']
write(D / 'CPU_native_zero_branch_result.json', cpu)
write(D / 'terminal_source_integrity_and_checkpoints.json', integrity)
write(D / 'plain_SSH_operations.json', operations)

paper = ARCHIVE / 'A24_Scatterbrain_NeurIPS2021.pdf'
paper_text = ARCHIVE / '_text' / 'A24_Scatterbrain_NeurIPS2021.txt'
assert paper.read_bytes().startswith(b'%PDF-')
assert sha(paper) == '1503ff9a48b07855bbcb8e0cb3a09ff312dcdd3c8c6e1356ca1b0c67f9e97846'
pfe = ARCHIVE / '139_PFE_Det_Sensors2026.pdf'
assert sha(pfe) == sha(B / 'source_audit_v1' / 'A19_PFE_Det_Sensors2026.pdf') == '8bcbaf9fc4281bf77390485cc8c03a694017bcd7f876b9731535cb3740c19647'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
scatter_scope = 'Abstract and complete Sections4.1-4.2; PDF page6 Eq1-2 and local/linear alternatives visually inspected. Only opening of4.3; not its proofs, AppendixC, experiments, or author code.'
paper_meta = {
    'title': 'Scatterbrain: Unifying Sparse and Low-rank Attention',
    'arxiv_title_variant': 'Scatterbrain: Unifying Sparse and Low-rank Attention Approximation',
    'authors': ['Beidi Chen','Tri Dao','Eric Winsor','Zhao Song','Atri Rudra','Christopher Ré'],
    'venue': 'NeurIPS2021, Advances in Neural Information Processing Systems34',
    'publisher_url': 'https://papers.neurips.cc/paper/2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html',
    'download_url': 'https://proceedings.neurips.cc/paper_files/paper/2021/file/9185f3ec501c674c7c788464a36e7fb3-Paper.pdf',
    'version': 'Official conference PDF, not latest arXiv by assumption',
    'download_status': 'HTTP200_PDF_HEADER_AND_VISUAL_METHOD_CHECKED',
    'bytes': paper.stat().st_size, 'sha256': sha(paper),
    'text_path': str(paper_text), 'text_sha256': sha(paper_text),
    'archived_utc': now, 'read_scope': scatter_scope,
}
meta_path = paper.with_suffix('.source.json')
assert not meta_path.exists()
write(meta_path, paper_meta)

evm = read(D / 'github_EVMDet_repositories.json')
pide = read(D / 'github_PIDE_Net_all_repositories.json')
assert evm['total_count'] == 0 and not evm['incomplete_results']
assert pide['total_count'] == len(pide['items']) == 24 and not pide['incomplete_results']
access = {
    'EVMDet': {'scope': 'Publisher abstract, metadata and references only. Subscription preview; no full method or code obtained.', 'github_name_query_results': 0, 'query_url': 'https://api.github.com/search/repositories?q=EVMDet&per_page=10'},
    'PIDE_Net': {'scope': 'Publisher abstract, introduction preview and section fragments only; no full methods/formulas/code.', 'github_name_query_results': 24, 'metadata_screened_results': 24, 'screening': 'Names and descriptions identify no repository as this paper author release; not a claim that no public/private/unindexed code exists.', 'query_url': 'https://api.github.com/search/repositories?q=PIDE-Net&per_page=100'},
    'PFE_Det': {'publisher_page_access_this_turn': 'HTTP429; previously downloaded publisher PDF available and hash matched central archive.', 'scope': 'Complete Method3.1-3.4 across A19/A23 reading; experimental4.1-4.3 and Discussion5.1-5.4, Table1 and Figures8-9. Not complete comparative/cross-dataset experiments, no code/reproduction.', 'visual_this_turn': [18,20], 'code': 'No paper-author code identified in the bounded search; dataset links in data availability do not prove code absence.'},
}
write(D / 'public_access_and_read_scope.json', access)

raw_rows = [
    ('baseline','0.203','0.118','10.18','24.86'),
    ('FAENet','0.211','0.125','10.22','34.21'),
    ('MFAM','0.218','0.129','12.67','28.71'),
    ('MG_SSM','0.210','0.122','10.81','23.64'),
    ('FAENet_MFAM','0.218','0.132','12.71','38.07'),
    ('FAENet_MG_SSM','0.196','0.111','10.84','32.99'),
    ('MFAM_MG_SSM','0.222','0.129','13.30','27.55'),
    ('full','0.225','0.134','13.33','36.91'),
]
table = {row[0]:dict(zip(['AP','APsmall','parameters_M','GFLOPs'],row[1:])) for row in raw_rows}
delta = lambda a,b,key: str((Decimal(table[a][key])-Decimal(table[b][key]))*100)
calculations = {
    'source': str(pfe), 'source_pdf_page': 18, 'table': 1,
    'reported_metrics_only_not_our_experiment': True,
    'author_protocol': '640, random initialization, seed0, batch4, VisDrone300epochs, val Stage2 checkpoint selection, test-dev1610; not comparable to our fixed12/800/val548 diagnostic pilot.',
    'transcribed_visually_checked_rows': table,
    'our_point_difference_calculations_pp': {
        'full_minus_baseline_AP':delta('full','baseline','AP'),
        'full_minus_baseline_APsmall':delta('full','baseline','APsmall'),
        'MG_alone_minus_baseline_AP':delta('MG_SSM','baseline','AP'),
        'FAE_plus_MG_minus_baseline_AP':delta('FAENet_MG_SSM','baseline','AP'),
    },
    'inference_limits': 'Point differences only, not significance, causal S/M effects, our accuracy, or proof of instance-level interference. Authors discuss intermodule dependence and leave multi-seed variability to future work.',
}
write(D / 'PFE_Table1_reported_data_and_arithmetic.json', calculations)

comparison = {
    'PFE_Det': {'original_author_conclusion': 'MFAM plus gated HSM-SSD is motivated by structural overwrite, noise propagation and dense-object merged activation responses; combination ablations and KPCA_CAM support author interpretation.', 'our_inference': 'This removes novelty from generic global-noise/local-gate motivation. Those ablations and projections do not separately manipulate VSSD S and denominator M while holding the original local read fixed.', 'not_proven': 'The observed task changes arise specifically from harmful neighbouring-instance writes; whether an improved native VSSD still has this effect.'},
    'EVMDet': {'original_abstract': 'EfficientViM-based UAV/small detection, training-only dynamic TopK/history HADF and spatial SAFE.', 'our_inference': 'Simply migrating EfficientViM to VisDrone or adding historical/spatial filtering is not enough.', 'not_proven': 'Exact operator/loss overlap without full text/code.'},
    'PIDE_Net': {'original_preview': 'Position-semantic disentanglement, semantic routing and dynamic prompts in SG-SSM, spatial weaving fusion.', 'our_inference': 'Generic semantic separation, guided state selection or density-scene prompts already have direct UAV prior.', 'not_proven': 'Whether a future specific instance interference correction is distinct from full SG-SSM.'},
    'Scatterbrain': {'original_method': 'Section4.1 diagnoses sparse/low-rank double counting;4.2 uses a sparse residual correction and explicitly permits local attention plus Linear attention components.', 'our_inference': 'Generic local/global de-overlap is an existing algorithmic idea; do not package subtraction plus fusion as the core invention.', 'not_proven': 'Equivalence of a complete VSSD block, its nonlinear convolutional local branch, RoPE, normalization and gate to Scatterbrain; no UAV detection effect follows from its approximation results.'},
    'candidate_status': 'No confirmed new algorithm. N remains a narrow mechanism question; broad local/global/gating and generic de-overlap claims are excluded.',
}
write(D / 'closest_method_comparison.json', comparison)
write(D / 'next_validation_constraints_DRAFT_NOT_A_LAUNCH.json', {
    'status': 'RESEARCH_CONSTRAINTS_ONLY_NO_NEW_ALLOCATION_OR_LAUNCH',
    'original_A22': 'Terminated; no checkpoints or failure tensor dumps retained; cannot exactly replay the lost trained failure state.',
    'engineering_before_any_new_run': [
        'Save the offending named raw loss, tensor dtype/range and complete output-branch finite summary before sanitation.',
        'Preserve a failure model/state and exact input/order/RNG evidence if a new isolated run is later selected.',
        'Validate a numerically stable zero-DDF expression as an engineering-only change; synthetic accumulator control alone does not validate the entire criterion.',
        'Do not claim that disabling autocast converts existing FP16 tensors to FP32; the author engine also disables autocast for criterion.',
    ],
    'scientific_protocol': 'A21 S/M endpoints, cohort and gates stay frozen; no block/mask/threshold changes selected from outcomes.',
    'novelty_gate': 'A written specific deployable operator difference from closest available methods is required before candidate-module training; EVMDet/PIDE full-method gaps remain disclosed.',
    'no_restart_or_eval_of_A22': True,
})

REPORT.write_text('''本轮结论：仍未找到可以确认成文的新主线。N“共享状态是否损害邻近微小实例”保留为机制假设；普通局部/全局门控和通用去重均已有明确先例。A22终止不构成该假设的正反证据，本轮没有追加检测器调用或训练。

1. 检索策略与检索记录
本轮为非系统性定向追踪：从最近方法的原文动机、公式及阅读缺口出发，追查“局部与低秩全局重复计数”算子先例。新增9条网页检索式、2个出版社页面访问、3次GitHub API读取（其中PIDE第二次补齐全部24条结果）、1份NeurIPS官方PDF下载。A23已执行但未归档的网页记录单独标记为历史补归档，不算本轮新增。
精确请求、原始返回、失败状态及API JSON均在A24目录。EVMDet公开仓库名检索0条；PIDE检索24条的名称与描述未能识别出作者发布，不能外推代码不存在。EVM仍为订阅预览，PIDE仍仅出版社预览；不声称读过二者完整方法。

2. 核心文献矩阵与阅读范围
矩阵77增至78条，只新增实际阅读的Scatterbrain。PFE-Det阅读扩展为完整Method3.1–3.4、4.1–4.3和5.1–5.4；Table1及Figures8–9用PDF第18/20页核图。其比较实验和跨数据集部分未全部读完，无作者代码复现。中央归档139_PFE与工作副本SHA256一致。
Scatterbrain取得NeurIPS2021官方PDF，完整读4.1–4.2，并核图第6页Eq1/2及可替换组件段；未完整读4.3证明、附录C、全部实验或作者代码。核心矩阵分别注明原文、推断与未证实部分。

3. 与最接近工作的逐项对比
PFE-Det已经将全局状态传播噪声、密集实例响应成片和局部门控/多路径保留联系起来。因此，“SSM全局混合可能伤害小目标，加入局部与门控”不能成为新的核心叙事。其模块组合消融与激活投影没有把当前VSSD的S、均值归一化M及原图局部读取路径分别控制，这只是一个尚可检验的机制差异，尚不是新算法。
EVMDet摘要已含EfficientViM小目标检测、训练期历史动态TopK与空间增强；PIDE预览已含位置/语义分离、语义路由和动态提示。二者完整算子仍缺，不能默认未来设计与它们不同。
Scatterbrain明确处理稀疏与低秩分支重复计数，并允许局部注意力和Linear attention作为组件。因而“全局减局部再融合”的宽泛增量被排除；这不等于证明整个VSSD和Scatterbrain同构，局部卷积、RoPE、M归一化和非线性门控都须另核。

4. 有文献依据但未证实的窄问题
在已有位置编码、局部通路、门控和残差的非因果视觉SSM中，邻近实例的共享状态贡献是否仍造成有害交互？已查原文不足以回答该精确问题；它们也不足以保证问题存在。当前机制路径应分别区分S写入、M归一化、原图局部读取以及背景干预代价，不能把CAM亮斑、低秩或模型调用成功当作答案。
尚缺：训练充分的固定模型、预先约定的因果路径对照结果、独立重复及可部署算法。只有方法论文没有报告某项诊断，并不能自动形成算法创新空白。

5. 候选增量、可行性与风险
现有原始数据、严格宿主装载与单3090容量已可承担受控小规模验证；这不证明训练稳定或12轮足以学习。通用局部/全局门控与去重不进入模块训练。更具体的实例写入/读取干预目前仍缺机制证据与可核验算子差异，暂不强行命名或拼模块。主要风险包括直接先例、反事实分布偏移、有限27组主样本的统计效能以及宿主学习不足。
PFE Table1作者报告：baseline AP0.203/APs0.118，完整模型0.225/0.134；本轮计算为+2.2/+1.6绝对百分点。FAENet+MG-SSM缺MFAM时AP0.196，较baseline为-0.7百分点。这些是作者表格的点差，不能证明S/M因果或统计显著。其640、scratch、seed0、300轮、test-dev口径与当前800/固定12/val548不一致，不作跨论文排名；作者已承认模块依赖与重复seed分析缺口。

6. 最小验证与训练故障边界
A22已按原保护终止：2105次前向、526次实际更新、0完整轮、0验证调用。冻结执行清单22个源码/依赖文件逐项复核0差异；checkpoints与失败张量文件均为空。第526次更新后的模型状态与原始失败loss没有保存，不能精确重放第2105批来确认故障根因。
CPU读取未改动的原生loss_local第253行表达式pred_corners.sum()*0，在全1有限FP16合成张量[2,500,132]和[2,1804,132]上复现sum=inf、零项=nan。小张量与FP32对照有限；FP32累加的零项及其梯度为有限零。这证明存在该数值风险，不证明它就是A22失败分量。关闭autocast不会自动提升既有FP16张量；作者训练引擎对criterion也关闭autocast，不能把这一点误说成自定义配方独有错误。
本轮只运行该CPU算子例子，CUDA未初始化、完整检测器前向0、优化器更新0。未修改A22冻结源码、未续跑、未启动候选或评估。以后若选择新的隔离验证，须先记录具名raw loss和各输出分支有限性、保存异常状态，并验证数值修复；新协议不得靠补跑来重解释旧终止结果。A21的S/M对照、样本与门槛保持原记录，不随本轮文献或故障改阈值。

7. 可核验参考与原始证据
PFE-Det: Progressive Feature Evolution for Small Object Detection in UAV Aerial Images. Sensors2026,26(15),5003. https://doi.org/10.3390/s26155003
EVMDet: EfficientViM for Small Object Detection. PRCV2025, pp414–427; first online2026-02-07. https://doi.org/10.1007/978-981-95-5758-5_30 （只读摘要/书目）
PIDE-Net: A heterogeneous processing paradigm for UAV object detection. ESWA309(2026),131194. https://doi.org/10.1016/j.eswa.2026.131194 （只读出版社预览）
Scatterbrain: Unifying Sparse and Low-rank Attention. NeurIPS2021. https://papers.neurips.cc/paper/2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html
原生criterion源码固定RicePasteM/Dome-DETR commit2dde3bc1946a3e9fad9abd0612b59fc39bd6b861；CPU记录校验其SHA256=357e8997241ca12ca146ccaab8d51df28ed563ae5bac53797825ab697b3474bc。新论文已归档到中央文献目录并记录下载来源、版本、SHA和阅读范围。59回执及A24目录保存检索、核图、CPU与逐项对比证据。

目标保持ACTIVE。当前推进的是排除已有想法、修正证据边界；没有确认新的论文主线，也没有新的训练许可等待用户处理。
''', encoding='utf-8')

backup = D / 'canonical_before_A24'
backup.mkdir(exist_ok=False)
for name in NAMES:
    shutil.copy2(B / name, backup / name)
phase = {
    'phase':'A24', 'classification':'progress', 'recorded_utc':now,
    'status':'CLOSEST_METHOD_NOVELTY_NARROWED_CPU_NUMERIC_RISK_VERIFIED',
    'goal_status':'active','mainline_validated':False,'complete_mainline':False,
    'report':str(REPORT),'receipt':str(RECEIPT),
    'new_detector_forwards':0,'new_optimizer_updates':0,
    'full_model_forwards':2365,'actual_cumulative_detector_optimizer_updates':528,
    'formal_detector_training_updates':526,'historical_capacity_updates':2,
    'cumulative_authorized_protocol_cap':39092,'full_forward_cap_closed':True,
    'counts_terminal':True,'new_training_jobs':0,'new_real_val_calls':0,
    'A22_status':'STOPPED_WITH_EVIDENCE_NO_AUTOMATIC_RESUME',
    'execution_source_and_dependencies_checked':22,'execution_source_mismatches':0,
    'A22_checkpoint_files':0,'exact_failed_loss_or_trained_state_retained':False,
    'synthetic_CPU_operator_probe':'Finite FP16 input can produce NaN in native sum-times-zero DDF branch; FP32 accumulator control finite.',
    'actual_A22_failure_root_cause_proven':False,
    'PFE_full_method_and_key_ablations_read':True,
    'new_core_papers':1,'core_matrix_rows':78,
    'broad_novelty_excluded':['generic local/global gating','generic local/global de-overlap as a first algorithm'],
    'trained_mechanism_effect_estimated':False,'algorithm_increment_trained':False,
    'next':'N只保留窄机制假设；先形成相对最近方法的具体算子差异并完善具名loss/异常状态记录。没有明确差异前不训练新模块；数值修复不作为论文创新，A22不自动重启。总体编码器与解码器选题继续开放。'
}
board = read(B / NAMES[0])
board.update(status='NO_CONFIRMED_MAINLINE_BROAD_LOCAL_GLOBAL_IDEAS_ALREADY_PRIOR',mainline_validated=False,latest_update_utc=now,latest_report=str(REPORT),current_priority_question=phase['next'])
board['A24_evidence'] = phase
for candidate in board['candidates']:
    if candidate['id']=='N_NONCAUSAL_STATE_INSTANCE_INTERFERENCE':
        candidate.update(status='BROAD_NOVELTY_EXCLUDED_NARROW_MECHANISM_UNTESTED_BASELINE_INCOMPLETE',A24_evidence=phase)
board.setdefault('rejected_generic_novelty_claims',[]).append('A24: Generic local/global de-overlap has explicit Scatterbrain2021 precedent; not a full-block equivalence or a UAV interference result.')
write(B / NAMES[0],board)

search=read(B/NAMES[1])
search['A24_closest_method_and_operator_followup']={'recorded_utc':now,'new_web_search_query_count':9,'publisher_open_count':2,'GitHub_API_requests':3,'new_official_PDFs':1,'historical_A23_web_records_archived_separately':True,'raw_record_files':[str(D/'prior_and_current_web_search_records.json'),str(D/'additional_web_search_records.json')],'access_limits':access,'report':str(REPORT)}
write(B/NAMES[1],search)
matrix=read(B/NAMES[2])
assert len(matrix['rows'])==77 and not any(row['id']=='Scatterbrain' for row in matrix['rows'])
for row in matrix['rows']:
    if row['id']=='PFE_Det':
        row.update(read_scope=access['PFE_Det']['scope'],author_method='FAENet preprocessing + MFAM backbone with adopted LWGA groups + HSM-SSD/CGLU MG-SSM on P3/P4; P5 Transformer and DEIM decoder/loss retained. Full Method3.1-3.4 read.',our_consequence='Explicit global-noise propagation and dense-instance merged responses already motivate local/multipath/gated modeling. Cannot claim that motivation as new.',unresolved='No author-code reproduction; whole comparisons/cross-dataset section not fully read. Combination AP and KPCA_CAM do not identify the exact VSSD S-versus-M effect.',A24_reported_table_arithmetic=str(D/'PFE_Table1_reported_data_and_arithmetic.json'))
matrix['rows'].append({'id':'Scatterbrain',**{k:paper_meta[k] for k in ['title','authors']},'url':paper_meta['publisher_url'],'tier':'NeurIPS2021','read_scope':scatter_scope,'author_method':'Sparse residual corrects the low-rank approximation on sparse support; local attention and Linear attention are explicitly allowed alternatives.','our_consequence':'Excludes generic sparse/local plus low-rank/global de-overlap as a new algorithmic idea.','unresolved':'No equivalence of full VSSD nonlinear local/gated/normalized block or UAV effect; not all proofs/AppendixC/experiments/code read.','local_PDF':str(paper),'sha256':sha(paper)})
matrix.update(updated_utc=now,latest_report=str(REPORT),latest_empirical_boundary='A22 terminal526updates,0epochs,0val; A24CPU operator counterexample0detector calls. No trained mechanism or new algorithm.')
write(B/NAMES[2],matrix)
state=read(B/NAMES[3])
state.update(recorded_utc=now,goal_status='ACTIVE',mainline_validated=False,diagnostics_complete=False,model_forwards=2365,training_updates=528,formal_detector_training_updates=526,cap=39092,full_forward_cap_closed=True,counts_terminal=True,latest_report=str(REPORT),latest_authoritative_phase=phase,current_priority_question=phase['next'],current_goal_turn='A24_closest_method_overlap_and_CPU_numeric_counterexample')
state.setdefault('authoritative_phase_history',[]).append(phase)
write(B/NAMES[3],state)
audit=read(B/NAMES[4])
audit.update(updated_utc=now)
audit['A24_closest_operator_and_numerics']=phase
write(B/NAMES[4],audit)

files=[path for path in D.iterdir() if path.is_file()]+[paper,paper_text,meta_path,pfe,REPORT,Path(__file__)]+[B/name for name in NAMES]
receipt=dict(phase)
receipt.update(files_sha256={str(path):sha(path) for path in files},CPU_probe=cpu,source_integrity=integrity,paper_read_scope_and_access=access,previous_goal_turn_classification='progress: A23 verified terminal evidence and corrected authoritative state.')
write(RECEIPT,receipt)
for path,digest in receipt['files_sha256'].items():
    assert sha(path)==digest
assert len(read(B/NAMES[2])['rows'])==78
assert read(B/NAMES[3])['model_forwards']==2365
assert read(B/NAMES[0])['mainline_validated'] is False
print(json.dumps({'status':'RECORDED_AND_HASHES_VERIFIED','files_verified':len(receipt['files_sha256']),'matrix_rows':78,'new_detector_calls':0,'mainline_validated':False,'report':str(REPORT)},ensure_ascii=False))
