import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
B=Path(__file__).resolve().parent
D=B/'A26_查询扩容接口与缓存核验_v1'
E=B.parent
REPORT=B/'62_A26查询扩容接口与完整解码器对照核验.txt'
RECEIPT=B/'63_A26缓存还原与三层解码器核验回执.json'
NAMES=['01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json','15_当前状态与结果校验.json','16_最近工作源码核查.json']
assert not REPORT.exists() and not RECEIPT.exists()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

opfile=E/'前期调研_码率受限航拍小目标_20261028_v1'/'远端验证_20261029_v1'/'ssh_operation_receipts.jsonl'
ids=['A26_readonly_cache_path_and_loader_inventory','A26_create_isolated_CPU_interface_directory',
     'A26_upload_cpu_prefix_interface_probe.py','A26_upload_cpu_interface_protocol_FROZEN.json',
     'A26_run_three_CPU_decoder_interface_calls_once','A26_package_terminal_CPU_interface_evidence',
     'A26_download_terminal_CPU_interface_evidence']
operations={}
for line in opfile.read_text(encoding='utf-8').splitlines():
    try:x=json.loads(line)
    except ValueError:continue
    if x.get('id') in ids and x.get('action') in ['exec','upload','download']:
        assert x['id'] not in operations
        operations[x['id']]=x
assert set(operations)==set(ids)
for x in operations.values():
    if x['action']=='exec':assert x['exit_code']==0
    else:assert x['status'] in ['UPLOADED','DOWNLOADED']
write(D/'plain_SSH_operation_receipts.json',operations)
packing=json.loads(operations['A26_package_terminal_CPU_interface_evidence']['stdout'])
archive=D/'engineering_evidence.zip'
assert archive.stat().st_size==packing['bytes']==160366
assert sha(archive)==packing['sha256'] and len(packing['sha256'])==64
verified=D/'verified_engineering_evidence';verified.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert len(z.infolist())==packing['members']==15
    assert sum(x.file_size for x in z.infolist())<5*1024*1024
    for x in z.infolist():
        rel=PurePosixPath(x.filename)
        assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in x.filename
        p=(verified/x.filename).resolve();assert p.is_relative_to(verified.resolve())
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(x.filename))

protocol=read(verified/'cpu_interface_protocol_FROZEN.json')
assert sha(verified/'cpu_interface_protocol_FROZEN.json')==sha(D/'cpu_interface_protocol_FROZEN.json')=='49d65138ef66ec7259d3be7b05e42cfd6aa0b13387bfde4126eba05fb94b611d'
assert sha(verified/'cpu_prefix_interface_probe.py')==sha(D/'cpu_prefix_interface_probe.py')=='14f33a519385adf8c61ac0d2b949ee9160f5f7b39d35a3191714388a7ad8e232'
r=read(verified/'result.json')
assert r['status']=='CPU_THREE_LAYER_PREFIX_INTERFACE_PASS'
assert r['decoder_entries']==r['decoder_completed']==r['validated_outputs']==3
assert r['layer_entries']==9 and all(x==3 for x in r['layer_calls_per_arm'].values())
assert r['complete_detector_forwards']==r['optimizer_updates']==r['forbidden_full_model_attempts']==0
assert not r['CUDA_initialized'] and not r['GT_arrays_read'] and not r['AP_computed']
assert r['state_digest_before']==r['state_digest_after']==protocol['state_digest']
assert sha(verified/'predictions_no_GT.npz')==r['predictions_sha256']
run_output=[]
for line in operations['A26_run_three_CPU_decoder_interface_calls_once']['stdout'].splitlines():
    try:x=json.loads(line)
    except ValueError:continue
    if isinstance(x,dict) and x.get('status')==r['status']:run_output.append(x)
assert run_output==[r]
events=[json.loads(line) for line in (verified/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
assert [x['event'] for x in events]==['FULL_THREE_LAYER_DECODER_ENTER','FULL_THREE_LAYER_DECODER_COMPLETE']*3
assert [x['arm'] for x in events[::2]]==['BASE250','NATIVE500','ONEWAY500']

with np.load(verified/'predictions_no_GT.npz',allow_pickle=False) as z:
    independent={}
    for arm in ['NATIVE500','ONEWAY500']:
        independent[arm]={kind:float(np.max(np.abs(z[arm+'_'+kind][:,:250]-z['BASE250_'+kind]))) for kind in ['boxes','logits']}
    assert independent==r['prefix_max_absolute_differences']
    for arm in ['BASE250','NATIVE500','ONEWAY500']:
        with np.load(verified/(arm+'_raw_returned.npz'),allow_pickle=False) as raw:
            for kind in ['boxes','logits']:assert np.array_equal(raw[kind],z[arm+'_'+kind])
assert independent['ONEWAY500']['boxes']<=protocol['prefix_absolute_tolerance']['boxes']
assert independent['ONEWAY500']['logits']<=protocol['prefix_absolute_tolerance']['logits']
selection=read(verified/'fixed_query_selection.json')
assert len(selection['query_ids'])==1 and len(selection['query_ids'][0])==len(set(selection['query_ids'][0]))==500
inventory=read(D/'cache_memory_roundtrip_inventory.json')
assert inventory['cache_count']==12 and all(x['roundtrip_bitwise_equal'] for x in inventory['rows'])
assert inventory['rows'][0]['memory_sha256']==r['input_hashes']['memory']
for row in inventory['rows']:assert sha(row['file'])==row['sha256']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
verification={'status':'PASS_INDEPENDENT_ARCHIVE_COUNTERS_NUMPY_PREFIX_DIFFERENCES',
 'archive_sha256':sha(archive),'members_verified':15,'frozen_protocol_and_script_match_local':True,
 'independent_prefix_max_differences':independent,
 'max_box_component_change_in_800_input_pixels':{k:v['boxes']*800 for k,v in independent.items()},
 'model_state_unchanged':True,'whole_detector_forwards_for_verification':0,
 'decoder_forwards_for_verification':0,'GT_task_metrics_computed':False,
 'source_scope':'Known native mask property/interface verification on1 cached real-image context; not a task-harm, AP or novelty result.'}
write(D/'independent_engineering_verification.json',verification)

dataset=read(B/'A12_原生边界分布诊断_v1'/'asset_selection.json')
assert len(dataset['cases'])==12 and len({x['prefix'] for x in dataset['cases']})==12
assert all(len(x['GT'])==x['valid_GT'] and sum(g['size800']<8 for g in x['GT'])==x['tiny_GT'] for x in dataset['cases'])
assert all(g['class_id'] in range(1,11) for x in dataset['cases'] for g in x['GT'])
cohort={'source_manifest':str(B/'A12_原生边界分布诊断_v1'/'asset_selection.json'),
 'sha256':sha(B/'A12_原生边界分布诊断_v1'/'asset_selection.json'),
 'case_count':12,'filename_prefixes':12,'valid_GT':sum(x['valid_GT'] for x in dataset['cases']),
 'tiny_GT_at_800':sum(x['tiny_GT'] for x in dataset['cases']),
 'metadata_only':[ {k:x[k] for k in ['file','prefix','width','height','valid_GT','tiny_GT','annotation_sha256']} for x in dataset['cases']],
 'limits':'Existing prediction-blind annotation-selected cohort subsequently used by other diagnostics; a new O test here would be exploratory, not independent confirmation. Filename-prefix grouping does not establish independent scenes.'}
assert cohort['valid_GT']==1241 and cohort['tiny_GT_at_800']==263
write(D/'possible_task_cohort_metadata_NO_TASK_RESULTS.json',cohort)
next_design={'status':'DRAFT_NOT_ALLOCATED_NOT_EXECUTED','new_task_decoder_allocation':0,
 'proposed_scope':'Four full3-layer decoder arms over the12 existing fixed image memories; reuse the3 case00 preflight outputs only if source/state/query hashes match exactly.',
 'candidate_budgets':[250,500],
 'arms':['base250','native500','symmetric_group_isolation500','oneway_isolation500'],
 'proposed_additional_decoder_calls_if_later_frozen':45,
 'already_completed_engineering_decoder_calls':3,
 'full_detector_calls_required_if_caches_validate':0,'optimizer_updates':0,
 'required_before_execution':['Freeze task metrics and thresholds before examining O GT effects.','Resolve ignored-region/others handling from official evaluation source or explicitly label a diagnostic-only policy.','Fix output detection cap rather than following the native query-count-dependent cap.','Use the same final500 candidates for the primary arm comparison, with duplicate errors and additional coverage compensation checked.','Check prefix fidelity on every replay; do not relax tolerances after results.'],
 'interpretation_gate':'A restored old-output IoU bound follows from mask construction and is not task benefit. If there is no uncompensated harm or no matched-budget AP benefit, do not train a module around this idea.',
 'no_A22_resume':True,'no_new_training':True}
write(D/'next_task_diagnostic_DRAFT_NO_ALLOCATION.json',next_design)

REPORT.write_text('''本轮只完成O候选的接口与数值对照核验，仍未确认论文主线。现成作者Dome模型在一份真实图像缓存上已能重放完整三层decoder：真实查询扩容会改变旧查询输出，单向隔离在预设数值容差内保留旧输出。没有读取GT计算任务效应，没有AP结果；这种已知mask性质不是算法创新。

1. 检索策略与检索记录
优先查验当前工程事实：固定作者源码、现有缓存保存位置、输入投影/TopK、跨查询路径和后处理。新增2条检索式查官方VisDrone DET评价口径，原始请求及返回保存在official_VisDrone_eval_discovery.json。官方README说明忽略区域/others不计入评价，但本轮尚未固定工具包commit或读取完整评价函数，不能声称已复现官方指标。最近论文差异继续沿用A25已披露范围，不凭接口成功排清创新性。

2. 核心文献与当前宿主
核心矩阵保持82条、候选15个。O宿主为严格装载的作者Dome-S-VisDrone模型，13,240,007参数、1033个state键、3层decoder、4级feature。固定源码2dde3bc1946a3e9fad9abd0612b59fc39bd6b861；权重SHA256=42874302fd97e17c2a23a46a8ec35d2ea4d2ffe479f0d632c1327fafbf910eb7。它与已终止的VSSD新宿主训练分开，未续跑A22。

3. 源码及最近方法对比中的新事实
原生TransformerDecoder.value_op在当前eval路径对memory不做value projection/scale，只reshape、permute和按四级空间大小拆分；旧A15保存了完整value_0..3，因而可以逆变换恢复输入memory。旧target/reference/query_pos是末层输入，不能直接当作完整三层的初始查询。本轮没有用末层干预替代完整解码器实验。
原生初始化先计算encoder TopK500，再做PAQI密度筛选。本轮固定TopK500，250从同一池中切片，构成相同候选身份的前缀对照；这是受控查询初始化，不能宣称复现原生PAQI整模型精度。
原生postprocessor第46行每次把num_top_queries覆盖为raw query数量。所以以后n与m比较必须固定检测输出上限；仅有更多输出框的指标变化不算解码器机制改善。Group DETR/GroupEnsemble/DN等mask先例依然成立，本次接口性质不消除这些先例。

4. 缓存与证据边界
12份旧缓存逐一匹配A15终止清单SHA256。每份有40,000/10,000/2,500/625个位置、8头×32通道，可恢复FP32[1,53125,256] memory，每份54,400,000字节。12份全部通过逆变换再应用布局操作的逐位相等检查。没有重算图像编码器，也没有新整图forward。
这个检查针对已归档value，不存在旧uncached memory的独立hash，原路径对应关系仍依赖已核对的源码和旧捕获来源。旧缓存已被多个诊断使用，后续O任务实验只能算探索性；12个文件名前缀也不保证12个独立场景。

5. 候选算法可行性与风险
O所需的完整三层、固定图像memory、共同候选池和单向隔离已具备可运行接口。仍待证明的是“改变是否造成有害任务结果”，以及不同于已有分组/排序/KV复用的具体增量。隔离可能同时切断有用上下文，保持旧输出也可能保留错误；新增候选还可能增加重复误检。不能根据坐标变化的幅度选择正面解释。
CPU工程核验不是GPU延时/显存测试，不据此宣称满足端到端实时性能。潜在增量仍须在相同最终候选数下证明任务收益，不能把静态image memory共享、标准单向mask或已知prefix性质称为创新。

6. 本轮实测与最小后续验证
在执行前固定engineering协议SHA256=49d65138ef66ec7259d3be7b05e42cfd6aa0b13387bfde4126eba05fb94b611d，源脚本SHA256=14f33a519385adf8c61ac0d2b949ee9160f5f7b39d35a3191714388a7ad8e232。只分配3次CPU完整三层decoder重放：BASE250、NATIVE500、ONEWAY500；每组确实经过3层，共9层调用。根model/backbone/encoder/outer forward受禁止钩子保护，无尝试；整图forward0、优化器更新0、CUDA未初始化。
这是一组新的decoder-only工程实验，使用1份真实图像上下文，不能用“整图计数0”隐藏它。mask只阻止旧250行读取新增250列，其余路径保持一致。固定model state在前后均为82c3ba49d69370a3d0105e47fc0ab5d715a9e7b1ad15feb275b1b497b5b0ccce。
相对BASE250，NATIVE500旧前缀的cxcywh坐标最大差0.004677533410339355（800输入下约3.742个像素的最大分量变化），分类logit最大差2.0388760566711426。这里不是定位误差，没有与GT比较。ONEWAY500的坐标最大差5.960464477539063e-08、logit最大差1.71661376953125e-05，通过预设1e-6坐标/1e-4logit绝对容差；不是逐位相同。输入hash、全部模型state和源码文件保持原证据范围内一致。
结果ZIP160366字节，SHA256=__ARCHIVE_SHA256__，15成员。已独立核对冻结源码/协议、6条进入/返回事件、3组原始输出和NumPy差值，未为验证额外调用模型。
后续可考虑12份旧memory上的四组任务对照：base250、native500、双向组隔离500、单向隔离500。原3份输出符合所有source/state/query hash时可复用，候选新增45次完整decoder重放；目前它只是DRAFT，任务执行分配仍为0。此cohort有1241个有效GT、263个800尺度area<64 tiny GT。必须先核定忽略区域与输出上限，固定配对指标和门槛；任务判据应检查扩容损失是否被新增候选补偿、同预算AP/APsmall和重复误检，不能仅用被mask结构保证的max-IoU性质。
架构整图累计调用仍2365，参数更新仍528（526正式、2容量）。新增3次完整decoder组件重放单列，不能与历史72次末层suffix混计；本3次额度已关闭，无后续自动任务，A22仍终止。

7. 可核验来源
固定作者源码：https://github.com/RicePasteM/Dome-DETR/tree/2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
官方DET评价文档：https://github.com/VisDrone/VisDrone2018-DET-toolkit
既有最近工作：Group DETR（ICCV2023）https://doi.org/10.1109/ICCV51070.2023.00610 ；GroupEnsemble（arXivv1）https://arxiv.org/abs/2603.01847v1 。本轮没有新增全文阅读或作者代码复现声明。
本轮原始工程输出、完整调用事件与独立核验见A26目录及63回执。目标保持ACTIVE，O任务效应和创新性均未验证，N仍保留未验证假设。
'''.replace('__ARCHIVE_SHA256__',packing['sha256']),encoding='utf-8')

backup=D/'canonical_before_A26';backup.mkdir(exist_ok=False)
for name in NAMES:shutil.copy2(B/name,backup/name)
old=read(B/'61_A25完整损失与新文献核验回执.json')
for path,digest in old['files_sha256'].items():
    original=Path(path)
    check=backup/original.name if original.parent==B and original.name in NAMES else original
    assert sha(check)==digest,(path,'A25 historical evidence not unchanged')

phase={'phase':'A26','classification':'progress','recorded_utc':now,'status':'THREE_LAYER_PREFIX_INTERFACE_PASS_TASK_EFFECT_UNTESTED',
 'goal_status':'active','mainline_validated':False,'complete_mainline':False,'report':str(REPORT),'receipt':str(RECEIPT),
 'new_whole_detector_forwards':0,'new_optimizer_updates':0,'new_training_jobs':0,'raw_image_forwards':0,
 'new_complete_three_layer_decoder_calls':3,'new_decoder_layer_calls':9,'new_cached_real_image_contexts':1,
 'new_AP_evaluations':0,'new_GPU_model_calls':0,'full_model_forwards':2365,
 'actual_cumulative_detector_optimizer_updates':528,'formal_detector_training_updates':526,'historical_capacity_updates':2,
 'cumulative_authorized_protocol_cap':39092,'full_forward_cap_closed':True,'counts_terminal':True,
 'CPU_decoder_component_cap':3,'CPU_decoder_component_cap_closed':True,
 'historical_last_layer_suffix_calls':72,'new_complete_decoder_calls_not_merged_with_old_suffix':True,
 'cache_layouts_verified':12,'engineering_protocol_sha256':sha(D/'cpu_interface_protocol_FROZEN.json'),
 'O_inter_query_dependence_observed_on_one_context':True,'O_prefix_numeric_identity_verified':True,
 'O_task_harm_measured':False,'O_AP_benefit_measured':False,'O_novelty_cleared':False,
 'next_task_decoder_allocation':0,'new_core_papers':0,'core_matrix_rows':82,'candidate_count':15,
 'A22_restarted':False,'N_scientific_design_changed':False,
 'next':'O完整三层对照接口已通过；先核对官方ignore/others评价源码和12例原标注来源，再冻结任务指标、门槛与45次候选decoder-only预算。现有坐标变化不作有害性或AP证据；不重启A22，不训练新模块。'}
board=read(B/NAMES[0])
board.update(status='NO_CONFIRMED_MAINLINE_O_INTERFACE_VALIDATED_TASK_EFFECT_UNTESTED',mainline_validated=False,latest_update_utc=now,latest_report=str(REPORT),current_priority_question=phase['next'])
board['A26_evidence']=phase
for x in board['candidates']:
    if x['id']=='O_QUERY_BUDGET_EXPANSION_CONTEXT':
        x.update(status='FULL_DECODER_INTERFACE_VALIDATED_TASK_HARM_AND_NOVELTY_UNTESTED',causal_evidence=False,increment_approved=False,
                 real_model_probe_executed=True,real_model_probe_scope='Three decoder-only engineering replays of1cached real image; no GT task effects.',A26_evidence=phase)
write(B/NAMES[0],board)
search=read(B/NAMES[1])
search['A26_interface_and_eval_policy']={'recorded_utc':now,'new_queries':2,'raw_source_record':str(D/'official_VisDrone_eval_discovery.json'),
 'scope':'Official toolkit README only, not a pinned evaluation-code or benchmark reproduction; native source and archived cache inspected directly.',
 'new_papers':0,'report':str(REPORT)}
write(B/NAMES[1],search)
matrix=read(B/NAMES[2]);assert len(matrix['rows'])==82
for x in matrix['rows']:
    if x['id']=='Dome_DETR':x['A26_source_and_interface_scope']=phase
matrix.update(updated_utc=now,latest_report=str(REPORT),latest_empirical_boundary='A26:3 complete3-layer CPU decoder replays on1cached real image; fixed TopK pool, prefix numeric control pass. No task harm/AP/novelty result;0new whole-image forwards or updates.')
write(B/NAMES[2],matrix)
state=read(B/NAMES[3])
state.update(recorded_utc=now,goal_status='ACTIVE',mainline_validated=False,diagnostics_complete=False,model_forwards=2365,training_updates=528,formal_detector_training_updates=526,cap=39092,full_forward_cap_closed=True,counts_terminal=True,latest_report=str(REPORT),latest_authoritative_phase=phase,current_priority_question=phase['next'],current_goal_turn='A26_cache_recovery_and_three_layer_decoder_interface_verification',complete_decoder_only_replays_A26=3,historical_last_layer_suffix_calls=72)
state.setdefault('authoritative_phase_history',[]).append(phase)
write(B/NAMES[3],state)
audit=read(B/NAMES[4]);audit.update(updated_utc=now);audit['A26_query_prefix_interface']=phase
audit['A26_source_interface_audit']=str(D/'source_interface_audit.json')
write(B/NAMES[4],audit)

files=[p for p in D.iterdir() if p.is_file()]+[p for p in verified.iterdir() if p.is_file()]
files += [Path(x['file']) for x in inventory['rows']]
files += [REPORT,Path(__file__),B/'dome_probe_loader.py',B/'A12_原生边界分布诊断_v1'/'asset_selection.json',B/'A15_固定query的采样中心干预_v1'/'completed.json']+[B/name for name in NAMES]
receipt=dict(phase)
receipt.update(previous_goal_turn_classification='progress: A25 reproduced full-criterion synthetic risks, verified new architecture sources and opened a distinct query-budget-context question.',
               files_sha256={str(p):sha(p) for p in files},previous_A25_evidence_files_verified_with_canonical_backup=len(old['files_sha256']),
               remote_result=r,independent_verification=verification,draft_task_cohort_metadata=cohort)
write(RECEIPT,receipt)
for path,digest in receipt['files_sha256'].items():assert sha(path)==digest,path
assert len(read(B/NAMES[0])['candidates'])==15 and not read(B/NAMES[0])['mainline_validated']
print(json.dumps({'status':'RECORDED_AND_HASHES_VERIFIED','files_verified':len(receipt['files_sha256']),'whole_detector_forwards_unchanged':2365,'new_decoder_only_calls':3,'task_AP_computed':False,'mainline_validated':False,'report':str(REPORT)},ensure_ascii=False))
