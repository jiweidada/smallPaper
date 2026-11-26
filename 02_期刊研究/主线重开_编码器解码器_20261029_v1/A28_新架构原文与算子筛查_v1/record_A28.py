"""Publish A28 source evidence and a provisional encoder question, not a mainline."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent
LIB=NEW.parent/'文献_相关原始论文_20261004'
NAMES=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
REPORT=NEW/'66_A28新架构原文对照与参考保持状态更新候选.txt'
RECEIPT=NEW/'67_A28原文源码与候选范围核验回执.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def main():
    assert not REPORT.exists() and not RECEIPT.exists()
    verification=read(ROOT/'source_and_algebra_verification.json')
    assert verification['status']=='PASS_BYTE_SCOPE_AND_SYNTHETIC_GEOMETRIC_SERIES_CHECK'
    rows=read(ROOT/'new_core_records.json')['rows'];method=read(ROOT/'P_method_draft.json')
    algebra=read(ROOT/'algebra_result.json');audit=read(ROOT/'operator_source_audit.json')
    prior=read(NEW/'65_A27任务指标与候选判定核验回执.json')
    for p,h in prior['files_sha256'].items():assert sha(p)==h,p
    backup=ROOT/'canonical_before_A28';backup.mkdir(exist_ok=False)
    for name in NAMES:shutil.copy2(NEW/name,backup/name)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    queries=[
      ['2026 tiny object detection transformer encoder decoder bottleneck multiscale site:arxiv.org','2026 UAV small object detection new encoder decoder architecture site:openaccess.thecvf.com','2026 small object detection linear attention scale normalization memory site:arxiv.org','2026 aerial object detection decoder query scale inconsistency site:arxiv.org'],
      ['vision gated delta rule object detection DeltaNet 2026','"small object" "decoder" "2026" "CVPR"','"MI-DETR" "github"','"vision" "non-causal" "delta" "2026"'],
      ['"vision" "delta rule" object detection','"vision" "DeltaNet" "non-causal"','"small object" "multi-time inquiries"','"2026" "object detection" "decoder" "frequency" "CVPR"'],
      ['"VBased" "linear attention" vision','"Revisiting Real-Time Detection Transformer with Efficient Encoder Design" github','"MI-DETR" "2503.01463"','"VBased" "github"'],
      ['"manifold-constrained" "object detection"','"mHC" "small object"','"differential attention" "UAV" detection 2026','"adaptive" "inquiries" "DETR" decoder'],
      ['DETR "encoder layers" "adaptive" "decoder" "fusion"','DETR "query" "depth" "dynamic" "memory"','"multi-time inquiries" "adaptive"','"DETR" "mixture of" "inquiry"'],
      ['"vision" "ridge" "attention" "object detection"','"non-causal" "DeltaNet" vision','"Mesa" "vision" "attention" ridge','"foreground" "background" "linear attention" "small object"'],
      ['vision Mamba "resolution" "decay" "gate"','"resolution" "DeltaNet" vision','"Mamba" "resolution-invariant"','"linear attention" "length" "resolution" "normalization"'],
      ['"Dynamic Fractal Mamba" github','"S4ND" "resolution"','"Le-DETR" "SHI-Labs"','"Adapting Linear-Time Architectures for Tabular" github'],
      ['"Mamba" "resolution" "discretization" "rescaling"','"VMamba" "resolution" "delta"','"Mamba" "resolution" "zero-order hold"','"selective" "state space" "resolution" "step size"'],
      ['VMamba MzeroMiko github vssm1_tiny d_state 1','"Mamba-3" "discretization"','"Mamba" "resolution adaptation" inference','"VMamba" "resolution" "rescale"']]
    assert len(queries)==11 and sum(map(len,queries))==44
    search={'recorded_utc':now,'review_type':'Targeted preliminary research,not a systematic review or exhaustive priority search.',
      'strategy':'Architecture discovery->closest-method full text->author source pin->operator applicability->conditional question. Public primary conference/publisher/arXiv/author sources support technical claims;secondary hits only aided discovery.',
      'query_batches':queries,'query_count':44,'batch_count':11,
      'raw_response_coverage':'Eight later search batches and primary followup opens are stored in two raw JSON files. Earliest three search responses remain in conversation tool history;their complete query strings and selected primary URLs are recorded here. Do not claim an all-raw search export.',
      'inclusion':'Direct encoder/decoder design,query-depth interaction,vision linear attention,selective update/discretization and resolution adaptation;known mechanism comparators from another task explicitly labeled.',
      'exclusion':'Anti-UAV detection-of-drones is not UAV-view ground-object detection;multimodal/event/segmentation-only papers are not direct task evidence;blogs/lists not relied upon for technical conclusions;compression/restoration branch stays retired.',
      'included_primary_records':[{'id':r['id'],'url':r['url'],'read_scope':r['read_scope']} for r in rows],
      'access_limits':['VBased official full PDF403/challenge: indexed fragments only.','mHC-DEIM direct page403: indexed author abstract only.','No verified complete Le-DETR or DF-Mamba code/weight release in this bounded search;absence is not claimed.']}
    write(ROOT/'search_strategy_and_log.json',search)
    next_text='优先核查P：参考保持的选择性状态更新校准。先排除其是否只是已有SSM重离散化/仿射流的同一应用，再冻结最多24次原生CPU扫描函数的合成接入核验；当前分配0。通过后才查完整checkpoint、VisDrone检测host和单卡资源，再决定任务试验。O当前增量不晋级，N不自动升主线，A22不重启。'
    phase={'phase':'A28','classification':'progress_source_audit_and_conditional_operator_prototype','recorded_utc':now,
      'status':'NEW_SINGLE_STATE_ENCODER_CALIBRATION_CANDIDATE_SOURCE_AND_ALGEBRA_ONLY',
      'goal_status':'active','mainline_validated':False,'complete_mainline':False,'report':str(REPORT),'receipt':str(RECEIPT),
      'previous_goal_turn_classification':'progress: A27 executed paired decoder task and rejected the one-way isolation increment.',
      'new_whole_detector_forwards':0,'new_optimizer_updates':0,'new_training_jobs':0,'new_GPU_model_calls':0,
      'new_complete_decoder_calls':0,'new_native_scan_component_calls':0,'new_real_image_task_evaluations':0,
      'new_synthetic_affine_coefficient_cases':1000,'full_model_forwards':2365,
      'actual_cumulative_detector_optimizer_updates':528,'formal_detector_training_updates':526,'historical_capacity_updates':2,
      'cumulative_complete_decoder_only_calls':48,'historical_last_layer_suffix_calls':72,
      'cumulative_authorized_protocol_cap':39092,'full_forward_cap_closed':True,'counts_terminal':True,
      'new_core_papers':10,'core_matrix_rows':92,'candidate_count':16,'new_full_PDFs_archived':8,
      'abstract_or_fragment_only_new_records':2,'new_verified_author_git_blobs':23,
      'P_native_fidelity_verified':False,'P_real_UAV_defect_measured':False,'P_task_benefit_measured':False,
      'P_novelty_cleared':False,'P_approved_as_mainline':False,'next_native_scan_allocation':0,
      'A22_restarted':False,'N_scientific_design_changed':False,'next':next_text}
    board=read(NEW/NAMES[0]);assert len(board['candidates'])==15
    assert method['candidate_id'] not in {c['id'] for c in board['candidates']}
    candidate=dict(method);candidate['id']=candidate.pop('candidate_id');candidate['A28_evidence']=phase
    candidate['related_existing_candidate']='N concerns noncausal fixed-resolution context;P concerns reference-preserving calibration of an actually recurrent selective scan. N operator/experiment is not reused as P evidence.'
    board['candidates'].append(candidate);board.update(status='SEARCHING_NO_VALIDATED_MAINLINE_A28',mainline_validated=False,
      latest_update_utc=now,latest_report=str(REPORT),current_priority_question=next_text,A28_evidence=phase)
    write(NEW/NAMES[0],board)
    log=read(NEW/NAMES[1]);log.update(supplement_utc=now)
    log['A28_architecture_and_discretization_followup']={'search_log':str(ROOT/'search_strategy_and_log.json'),
      'new_papers':10,'archived_PDFs':8,'scope_verification':verification,'operator_audit':str(ROOT/'operator_source_audit.json'),'phase':phase}
    write(NEW/NAMES[1],log)
    matrix=read(NEW/NAMES[2]);assert len(matrix['rows'])==82
    oldids={r['id'] for r in matrix['rows']};assert not(oldids & {r['id'] for r in rows})
    matrix['rows'].extend(rows);matrix.update(updated_utc=now,latest_report=str(REPORT),
      latest_empirical_boundary='A28 adds10 scoped primary records (8PDF,2fragment/abstract-only),23verified author-source blobs and a classical affine-flow prototype. P is a new unapproved encoder calibration question;no trained visual effect/native fidelity/novelty demonstrated. A27 one-way isolation screen remains negative.')
    write(NEW/NAMES[2],matrix)
    state=read(NEW/NAMES[3]);state['authoritative_phase_history'].append(phase)
    state.update(recorded_utc=now,latest_report=str(REPORT),latest_authoritative_phase=phase,
      previous_goal_turn=state['current_goal_turn'],current_goal_turn='A28_new_encoder_source_audit_and_reference_flow_candidate',
      current_priority_question=next_text,goal_status='ACTIVE',mainline_validated=False,A28_evidence=phase)
    write(NEW/NAMES[3],state)
    source=read(NEW/NAMES[4]);source.update(updated_utc=now)
    source['A28_primary_architecture_sources']={'phase':phase,'operator_source_audit':audit,'P_method_draft':str(ROOT/'P_method_draft.json'),
      'MI_DETR':'Nine exact author source blobs;parallel independent branches,Linear+LayerNorm fusion and UFI confirmed. Not a full runnable clone.',
      'Tabular_DeltaNet':'Eight exact author blobs;write decayhelper and grouping token semantics confirmed. No native VSSD sequential-beta mechanism.',
      'VMamba':'Six exact author blobs;one active Tiny config state_dim1,raw-bias-softplus ordering,native expEuler input and D*u skip inspected. No native function/model run.'}
    write(NEW/NAMES[4],source)
    maximum=max(r['reference_preserving_max_abs_error'] for r in algebra['refinement_rows'])
    citations='\n'.join(f'{r["id"]}: {r["title"]}\n  {r["url"]}\n  阅读范围：{r["read_scope"]}' for r in rows)
    report=f'''A28 新架构原文对照与参考保持状态更新候选
记录时间：{now}

当前结论：继续寻找主线，尚未达到可正式定题的证据要求。本轮形成新的编码器候选P：“变分辨率航拍检测中的选择性状态更新校准”。它有明确的原生代码入口、可计算的增量草案和适用限制；没有证实真实航拍缺陷、检测收益或新颖性。O的单向查询隔离增量继续不晋级；N仍未验证，原A22保持终止。

1. 检索策略与记录
本轮定向检索44条查询、11批，先查完整方法，再核对官方代码和单卡条件。不是系统综述，也不能以未检索到完全同名方法证明首创。核心矩阵由82增至92篇：新增8份原始PDF及2条仅片段/摘要记录；固定3个作者仓库，核验23个源文件Git blob。全文检索记录见 {ROOT/'search_strategy_and_log.json'}。
只依赖原作者/会议/出版商/arXiv技术证据；论文列表和博客仅帮助发现来源。早期三批的完整返回留于对话工具记录，后八批及相关主来源返回已保存JSON，不能声称导出了全部原始检索返回。

2. 核心文献矩阵与最近工作对比
MI-DETR已提出独立参数的并行多次询问、UFI深浅编码层连接与Lite-MI共享自注意力。其表6已有相近参数/计算量的串行层数对照，所以“多读几次”“共享自注意力”“补同预算对照”都不能直接立为新贡献。
AugDETR已按query特征对多个编码深度生成融合权重，排除仅把固定深度配对改成自适应门控的首创说法。Le-DETR（CVPR2026 Findings）已研究EfficientNAT、局部注意力及较低预训练成本；其COCO结果不是本研究的单卡或航拍验证。
VBased仅取得原始投稿的索引片段，已提到DeltaNet与局部注意力组合；mHC-DEIM仅读SSRN作者摘要，已报告mHC移植DEIM。没有把两篇记为全文阅读或已验证完整实现。
S4ND已做二维SSM分辨率适配，DF-Mamba已设计选择性步长随递归尺度变化，Mamba-3已分析并改进状态输入离散化。分辨率不变、缩放步长、高阶积分或换上现成Mamba都不是本候选的首创点。
表格ICL论文指出DeltaNet的长度/状态漂移并给出写入率衰减，但那是其他任务的原文证据；不能视为无人机或VSSD已经存在同一缺陷。

3. 可继续核查的问题与证据边界
[原文/源码] VMamba的所选官方配置 vssm1_tiny_0230s / v05_noz 使用每通道1维状态。参考函数先加delta_bias并softplus，然后采用 h'=exp(delta*A)h+delta*B*u，输出含原始D*u。论文附录明确把输入项称为一阶近似；这是已说明的模型设计，不是本轮发现的代码错误。源码commit：{audit['VMamba_commit']}。
[我的推断] 若推理网格改变，应区分状态保留与新信息写入的共同适配。直接把delta乘一个比例，未必保持原模型在参考网格上学得的离散更新行为。
[待验证假设] 这种更新尺度差异是否在真实高分辨率航拍检测中产生额外小目标损失；参考保持的联合校准能否在同分辨率、同权重和同输出预算下改善AP，而非仅改变分数或放大特征。
没有把上述假设当作已确证研究空白。二维扫描顺序、真实新增细节、卷积、padding、位置和检测头都可能造成跨分辨率变化；公式不能自动排除这些解释。VSSD的非因果KV聚合没有相同顺序beta更新，P不能沿用N的训练或把表格机制直接套上去。

4. 算法增量草案、可行性与风险
把原生单状态更新写成 h'=exp(a)h+b，其中a=delta*A，b=delta*B*u。对参考与当前网格比例rho，草案为：
  a_rho=rho*a
  b_rho=expm1(rho*a)/expm1(a)*b
a趋零时该比值取rho；rho=1直接走原生分支。rho暂拟每阶段N_reference/N_current，只是待验证的网格面积/扫描进度策略，不宣称等于真实二维物理规律。
这属于经典仿射流/SSM重离散化，公式本身不作为新数学贡献。唯一可能的增量是面向既有近似选择性扫描checkpoint的参考保持推理校准策略；它也可能与既有方法等价，仍需排查。
在state_dim1时，可将 g=expm1(rho*a)/(rho*expm1(a)) 折入每通道u，并使用rho*delta，将原始D*u保留在扫描外，可能复用现有内核。高维状态一般需要不同状态极点的不同增益，不能直接声称同一快速实现成立。完整native/CUDA接入、dtype和梯度均未核验，未改作者代码，也未运行模型。
单卡约24GB可先承担小张量原生函数核验；正式检测host、预训练权重和训练耗时还没确定。MI-DETR默认900queries、6/6层、4inquiries、batch16及多卡启动，不是已证明能在本机原配方训练的依据。更换现成骨干本身仅作对照，不列为创新。

5. 本轮完成的验证
冻结后运行1000组合成仿射系数，细分2/4/8/16倍的参考保持更新相对原更新最大绝对误差为{maximum!r}，参考步长下系数逐位保持，单状态输入折叠最大误差为{algebra['single_state_input_folding_max_abs_error']!r}。另用几何级数闭式重新核验；高维不同极点需要不同增益的反例已保留。
这只验证“细分块内系数相同”的条件代数性质，不是任意二维图像不变性、原生预训练模型前向一致、检测AP或训练可行性的证明；不是新定理或新颖性证据。本轮真实模型/扫描函数调用0、真实图像任务评价0、优化器更新0、GPU调用0。

6. 最小后续方案（草案，当前分配0）
先核查P是否已被已有参考保持校准/SSM重离散化方法覆盖。若仍有具体差异，再冻结最多24次原生CPU扫描组件调用：4个合成fixture，每个比较原生、rho1旁路、2倍朴素/联合细分、4倍朴素/联合细分。重点核对softplus顺序、原始D*u、输出dtype与有限性；rho1全返回张量必须逐位一致，块常系数细分才检查端点近似一致。它不需要新训练或真实图像。
组件通过后才验证可严格加载的Tiny权重、VisDrone检测host与单卡峰值显存/耗时，之后才决定是否训练新的有限基线。正式任务以VisDrone为主候选；检测host及独立复验数据集尚未冻结，YOLO26s可保留作常规参照，不把换骨干写成创新。新host的类别映射和输出数必须重新核对，不能直接套A27的12类后处理。
任务验证必须在相同输入分辨率下配对比较同一checkpoint，包含原生、只改步长、直接ZOH、参考保持校准及简单幅度控制，排除放大输入本身的收益。先审计既有图像使用清单，再选新的预测盲开发样本；完整验证集、额外seed/独立航拍数据是后续确认条件。统计门槛和停止条件在看结果前冻结。
{next_text}

7. 可核验参考文献与阅读范围
{citations}

方法草案：{ROOT/'P_method_draft.json'}
最小验证草案：{ROOT/'minimal_validation_DRAFT.json'}
来源与合成核验：{ROOT/'source_and_algebra_verification.json'}
本轮不是主线完成声明，也没有启动任何下一组实验。整体目标保持active。
'''
    REPORT.write_text(report,encoding='utf-8')
    remapped={str(backup/Path(p).name) if Path(p).parent==NEW and Path(p).name in NAMES else p:h for p,h in prior['files_sha256'].items()}
    for p,h in remapped.items():assert sha(p)==h,p
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files.extend([REPORT]+[NEW/n for n in NAMES])
    for row in rows:
        files.append(Path(row['access_record']))
        if 'local_pdf' in row:
            rec=read(row['access_record']);files.extend([Path(rec['file']),Path(rec['text_file'])])
    for p in [NEW/'source_audit_v1/A18_VSSD_mamba2.py']:
        files.append(p)
    receipt={**phase,'source_verification':verification,'algebra_result':algebra,
      'previous_A27_evidence_files_verified_with_canonical_backup':len(remapped),
      'previous_A27_evidence_remapped_sha256':remapped,
      'files_sha256':{str(p):sha(p) for p in sorted(set(files),key=str)}}
    write(RECEIPT,receipt)
    for p,h in receipt['files_sha256'].items():assert sha(p)==h,p
    print(json.dumps({'status':phase['status'],'candidate_count':16,'matrix_rows':92,
      'new_receipt_files_verified':len(receipt['files_sha256']),'previous_receipt_files_verified':len(remapped),
      'new_model_calls':0,'mainline_validated':False,'report':str(REPORT)},ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
