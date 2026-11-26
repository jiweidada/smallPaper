"""Deliver the checked prototype and a bounded prospective screening plan."""
from pathlib import Path
import datetime,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'outputs/small_object_encoder_decoder_20261010_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    validation=read(OUT/'validation_report.json');assert validation['status']=='PASS_LOCAL_ENCODER_DECODER_PREFLIGHT_COLLECTION_AND_PARITY'
    spec=read(ROOT/'outputs/small_object_encoder_decoder_20261010_v1/design_specification.json')
    spec['status']='IMPLEMENTED_AND_ENGINEERING_PREFLIGHT_VERIFIED_NOT_TRAINED';spec['implementation_revision']=2
    spec['validation_report_sha256']=sha(OUT/'validation_report.json')
    spec['parameters']={k:validation[k] for k in ['native_parameters','prototype_parameters','added_parameters','parameter_growth_percent']}
    spec['remaining']=['Bind runnable fixed12 training/export/source/data/initialization/evaluator protocol to existing BASE12.','Check optimizer coverage, actualbatch4 memory,AMP scaler/EMA and first100input/DN RNG during original12epoch prefix; no extra training stages or preflight learning steps outside its budget.','Fine bothseeds fixed12 first; same-capacity degraded controls only if bothseed native comparison succeeds.','APs>BASE and overallAP>=BASE required independently for eachseed. No continued training or tuned follow-up after a failed12 endpoint.','A fine-vs-degraded advantage is necessary for a fine-field mechanism attribution, not proof of universal novelty.']
    save(OUT/'design_specification.json',spec)
    plan={'status':'PROSPECTIVE_SCIENTIFIC_SCREEN_PLAN_NOT_A_LAUNCH_RECEIPT','frozen_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'architecture_sha256':sha(OUT/'design_specification.json'),'candidate':'SmallObjectDEIM fine geometry field and dual state encoder/decoder','stage1':[{'variant':'fine','seed':s,'max_total_epochs':12} for s in (1,2)],'stage2_only_if_both_stage1_native_comparisons_pass':[{'variant':'degraded','seed':s,'max_total_epochs':12} for s in (1,2)],'max_new_detector_training_configurations':4,'no_third_seed_or_search':True,'recipe':'Original132 schedule prefix,epoch0..11; do not change132 to12 in scheduler/augmentation settings. Common initialization,not mature132 weights.','same_seed_BASE12_checkpoint_sha256':{'1':'ba3b6351fa8a4f7c769698c616ed6d20acf6a60ba119cf8edc8611851c4d27d4','2':'6198ee5348630ce97c322c16aa4499a50a2f7af0d08dea1ec3a4cb632d9d5046'},'initialization':'Native common SHA85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91; all new output projections zero, geometry layers copy loaded semantic weights. Manual-gradient fixture weights never used.','budget':'Each configuration at most12 total epochs,including any real training stage. Current engineering checks added0 optimizer steps/0 epochs. No learning steps outside formal12prefix for capacity probing.','primary_endpoint':'Fixed epoch12 EMA standard COCO APs; report overallAP,AP75,smallAP75,classAP and new/harmed matches with exact source/evaluator binding. Do not substitute best checkpoint.','native_screen_gate_each_seed':'APs_fine>APs_BASE12 AND AP_fine>=AP_BASE12. Bothseeds required for promotion. Positive screen is exploratory,not significance/stability.','mechanism_gate_each_seed':'After native gate, APs_fine>APs_degraded at fixed12; identical learned capacity and coordinate chart. Otherwise no evidence attributing benefit to the fine spatial field.','stopping':'Failed native fixed12 gate retires the route; no degraded queue, extensions,resume,seed selection,new radius/chart/extra tuning to salvage it. User-authorized early stop records incomplete fixed12 and preserves artifacts. Crashes do not auto-retry.','runtime_required_before_or_during_first_formal_batch':['Live resources/PID protection and468native source identity.','Train/val annotation/image/common initialization and BASE12 recipe/evaluator equivalence.','Optimizer covers all trainable parameters exactly once; unchanged native groups and explicit treatment of added GroupNorm.','Actualbatch4 capacity,finite AMP gradients/scaler/EMA update recorded within epoch0; no extra warmup training.','First100 input/target and DN RNG checks against existing same-seed baseline traces.','Full strict candidate checkpoint save/reload retains learned geometry copies; initialization copying only on native/common initial load.','Fixed12 outputs/native losses and original matching/FDR/score protocol retained.'],'formal_training_launched':False}
    save(OUT/'prospective_screen_plan.json',plan)
    text=f'''小目标专用 encoder / decoder：已实现原型，尚无训练AP

基于现有 DEIM-D-FINE-S / HGNetv2-B0 专门改造，不是从零重写整个检测器。

待验证问题
分类希望同类对象具有稳定语义；定位需要对中心、尺寸与细小空间差异敏感。共享状态是否限制了这两种需求，目前无法证实。此前完整encoder候选池诊断没有通过TopK遗漏优先门槛；两图流形邻域试算也没有证明统一特征塌缩。本结构是可以否证的机制假设。

Encoder（以下省略batch维）
HGNet stages：P2=64×160×160；P3=256×80×80；P4=512×40×40；P5=1024×20×20。
语义分支：保留原生HybridEncoder(P3/P4/P5)，输出S3/S4/S5，通道均256，展平共8400个token。
几何分支：P2经1×1投影、GroupNorm、SiLU，再经depthwise3×3及pointwise1×1；叠加从S3投影并上采样的上下文，得到G=64×160×160。
G保留较早的空间场，独立于语义token入口；没有承诺恢复stem已经丢失的信息，也没有新增P2到语义TopK排名。

Decoder
原生TopK仍选300个语义候选，类别排名规则不变。初始框头额外读G，产生零初始化delta4。
每个候选保留Qs和Qg两份256维状态；分别经过3层独立decoder，定位层初始复制语义层权重。
Qs逐层读取原生三尺度语义memory；Qg读取同一语义memory，且每层额外从G读取9个局部位置。
局部采样使用当前参考框，半径每轴max(框宽/高的一半,1个G格点)，双线性采样；坐标/采样FP32。
采样token附加8维坐标描述：相对dx/dy、log1p宽高格数、中心在格内的sin/cos周期位置。4头×16通道聚合。
类别头读取Qs；初始框与FDR角点头读取Qg。原生FDR固定参考坐标、回归范围、MAL/FGL/DDF与匹配接口保留。
LQE仍以角点质量修正类别分数；共享骨干/memory/参考框也仍存在。因此不是“分类定位完全独立”。训练的DN查询按原生方式添加及拆分；普通推理输出仍为300×10类别与300×4框。

与旧路线的关系
与旧P3局部框残差重叠的是初始框修正，不能把它当新贡献。这里待验证的是可训练G场、独立定位状态、逐层几何读取的共同作用。
旧P2简单加token、固定最终输出ROI、范围扩大、已中止小波分支均不恢复。
分类/定位分开已有ICCV2023 Decoupled DETR及CVPR2026 BHCL先例。既有112_BHCL PDF及3.2节已阅读，不重复下载；独立几何场与坐标读写是否形成有意义的差异，还需先例与消融支持。
本原型采用普通欧氏张量，不宣称学习了双曲流形或测量了曲率。

实际工程结果
修订2：17次前向、6次反向，训练/推理、FP32/AMP初始输出、共同初始化33项训练损失、原生BN/RNG、DN与成熟权重推理均通过相应核对；成熟模型激活梯度检查为37项损失。两种材料的损失项数不同，按各自原生行为保留。
零输出头获得梯度；临时成熟模型把输出投影显式置为.001*sin(arange+1)后，4类新增参数组均有有限非零梯度。此手动激活仅为连通性检查，已恢复/丢弃，不是训练初始化。
修订1的CPU/GPU权重比较错误已保留：退出1，仅运行5次原生前向，新增模型前向0；修订2独立运行并退出0。
合计22次前向、6次反向，优化器更新0，新增训练轮0。52文件完整下载、SHA/ZIP/源码/结果配对核验通过。
原生参数：{validation['native_parameters']:,}；原型参数：{validation['prototype_parameters']:,}；新增{validation['added_parameters']:,}，增加{validation['parameter_growth_percent']:.2f}%。未做完整训练显存或时延基准，不能称更轻更快。

12轮筛选设计
先fine双seed各12轮，从共同初始化开始、沿原132配方的前12轮；只比较固定第12轮EMA同seed BASE12。
两个seed均需APs提高且整体AP不下降，才进入相同容量degraded几何场对照；对照同样各最多12轮。
fine与degraded仅P2进入几何投影前是否avgpool2/恢复网格不同；所有学习模块、参数量、坐标描述、查询、损失一致。
若fine未通过基线门槛，停止这条路线；不续训、不挑最好轮、不追加调参或恢复旧实验。正式训练尚未启动，训练绑定/优化器覆盖/实际batch4 AMP与EMA仍需核验。

核验：validation_report.json；科学设计：design_specification.json；前瞻筛选：prospective_screen_plan.json（不是启动回执）；完整远端原始材料：collection/revision1与revision2。代码位于英文小论文/code/small_object_encoder_decoder_20261010_v2。
'''
    (OUT/'设计与验证结果.txt').write_text(text,encoding='utf8')
    codes=['small_object_model.py','geometry_memory.py','preflight.py','controller.py','preflight_protocol.json','prepare_revision2.py','verify_preflight.py','finalize_design.py']
    files=['design_specification.json','prospective_screen_plan.json','validation_report.json','engineering_correction.json','engineering_correction.diff','preflight_protocol.json','evidence_bundle.zip','local_validation_first_version.py','设计与验证结果.txt']
    save(OUT/'delivery.json',{'status':'SPECIALIZED_ENCODER_DECODER_PROTOTYPE_ENGINEERING_VERIFIED_NO_TRAINED_AP','observed_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{n:sha(OUT/n) for n in files},'code':{n:sha(HERE/n) for n in codes},'no_accepted_mechanism':True,'formal_training_launched':False,'AP_evidence':False,'epochs_added':0,'material_passport':validation['material_passport']})
    state_path=ROOT/'outputs/sod_roi12_20261009/execution_state.json';state=read(state_path);old_sha=sha(state_path)
    state['status']='PROBLEM_DRIVEN_RESEARCH_ACTIVE_SPECIALIZED_ENCODER_DECODER_PROTOTYPE_VERIFIED'
    state['specialized_encoder_decoder']={'status':'ENGINEERING_PREFLIGHT_VERIFIED_NOT_TRAINED','delivery':str(OUT/'delivery.json'),'delivery_sha256':sha(OUT/'delivery.json'),'architecture':str(OUT/'design_specification.json'),'prospective_screen_plan':str(OUT/'prospective_screen_plan.json'),'extra_parameters':validation['added_parameters'],'parameter_growth_percent':validation['parameter_growth_percent'],'max_epochs_each_configuration':12,'current_training_epochs_added':0,'AP_evidence':False,'new_mechanism_accepted':False}
    pilot=ROOT/'outputs/manifold_neighborhood_pilot_20261010_v1';state['manifold_neighborhood_pilot']={'delivery':str(pilot/'delivery.json'),'delivery_sha256':sha(pilot/'delivery.json'),'status':'ANALYZED_TWO_IMAGE_METHOD_PILOT','GT_count':288,'tiny_lt4_count':11,'images':2,'distinct_features':6,'curvature_measured':False,'root_cause_established':False}
    history=state.setdefault('research_progress_history',[]);history.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'manifold_pilot_and_specialized_encoder_decoder_prototype','delivery_sha256':sha(OUT/'delivery.json'),'epochs_added':0,'accepted_mechanism':False})
    state['last_goal_turn']={'classification':'PROGRESS','evidence':'Bounded two-image manifold neighborhood pilot and specialized encoder/decoder code with17forward6backward zero-update engineering preflight; preserved failed revision1 five native forwards;52files local verified.','scientific_scope':'ANALYZED_DESIGN_ONLY','AP_evidence':False,'new_mechanism_accepted':False,'next':'Bind bounded12epoch prefix runtime and perform fine dualseed screen with samecapacity degraded controls only after successful native gate.'}
    (OUT/'execution_state_before.json').write_bytes(state_path.read_bytes());state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    save(OUT/'state_update_receipt.json',{'previous_sha256':old_sha,'current_sha256':sha(state_path),'goal_status':'active','goal_complete':False})
    print(json.dumps({'status':'DELIVERED_PROTOTYPE','epochs_added':0,'AP_evidence':False,'goal':'active'}))
if __name__=='__main__':main()
