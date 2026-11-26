"""Evidence-bound selection report; requires the complete UAVDT audit and never launches training."""
import datetime
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent;DATA=HERE.parents[1]/'data/07_第二数据集_20261017'
EVIDENCE=DATA/'协议核查_v1'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    uav_path=EVIDENCE/'UAVDT完整原始包验收_v1.json'
    uav=read(uav_path)
    assert uav['state']=='complete_original_detection_archive_verified'
    assert uav['full_zip_crc_verified'] and uav['all_images_sha_and_JPEG_headers_verified']
    assert uav['image_counts']=={'train':24143,'test':16592}
    scorer_path=EVIDENCE/'UAVDT官方DET参考曲线核验_v1.json';scorer=read(scorer_path)
    assert scorer['reference_curve_agreement_passed']
    tiny_path=EVIDENCE/'original_tiny_audit_B.json';raw_tiny=read(tiny_path)
    tiny_prepared_path=EVIDENCE/'TinyPerson原版更新_B.json';prepared=read(tiny_prepared_path)
    by_path={row['path']:row for row in raw_tiny['annotations']}
    tiny_train=next(v for k,v in by_path.items() if k.endswith('/annotations/tiny_set_train.json'))
    tiny_test=next(v for k,v in by_path.items() if k.endswith('/annotations/tiny_set_test.json'))
    assert not tiny_train['missing_images'] and not tiny_test['missing_images']
    def bins(row):
        cumulative=row['normal_size_bins_stretch640'];total=row['normal']
        counts={'<4':cumulative['4'],'4-8':cumulative['8']-cumulative['4'],
                '8-16':cumulative['16']-cumulative['8'],'16-32':cumulative['32']-cumulative['16'],
                '>=32':total-cumulative['32']}
        return {'counts':counts,'percent':{k:v/total*100 for k,v in counts.items()},'total':total}
    uav_counts={r['sequence']:r['images'] for r in uav['per_sequence']}
    proposal_tag='uavdt-validation-v1-20261017'
    order=sorted(uav['splits']['train'],key=lambda s:hashlib.sha256((proposal_tag+':'+s).encode()).hexdigest())
    val=[];val_count=0
    for sequence in order:
        if val_count>=math.ceil(.2*uav['image_counts']['train']):break
        val.append(sequence);val_count+=uav_counts[sequence]
    uav_training=uav['image_counts']['train']-val_count
    tiny_training=prepared['counts']['train.json']['images']
    # The reference is a real completed, single-trainer VisDrone96 run, not an assumed new-dataset speed.
    reference_path=HERE.parents[1]/'data/06_第二轮_20261014/每半小时跟进_v4/latest_snapshot.json'
    original=read(reference_path)
    reference_status=original['servers']['A']['snapshot']['jobs']['lowpass_full96_seed0']['status']
    assert reference_status['state']=='complete' and reference_status['returncode']==0
    reference_hours=reference_status['elapsed']/3600
    steps={'VisDrone_reference':1617,'UAVDT_proposed':uav_training//4,'TinyPerson_prepared':tiny_training//4}
    hours={key:reference_hours*steps[key]/steps['VisDrone_reference'] for key in ['UAVDT_proposed','TinyPerson_prepared']}
    budget={}
    for name,value in hours.items():
        budget[name]={'per_arm_96epochs_image_step_projection_hours':value,
                      'six_trainings_projection_GPU_hours':value*6,
                      'six_trainings_two_cards_projection_days':value*6/2/24,
                      'eight_trainings_projection_GPU_hours':value*8,
                      'eight_trainings_two_cards_projection_days':value*8/2/24,
                      'measured_on_this_dataset':False}
    sources=[uav_path,scorer_path,tiny_path,tiny_prepared_path,reference_path]
    record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'UAVDT':{'original_images':uav['image_counts'],'sequences':50,'test_normal_GT':sum(r['positive_boxes'] for r in uav['per_sequence'] if r['split']=='test'),
                     'test_size_original_median':uav['size_summary']['test']['original_sqrt_area_quantiles_10_25_50_75_90'][2],
                     'test_size_input640_median':uav['size_summary']['test']['input640_stretch_sqrt_area_quantiles_10_25_50_75_90'][2],
                     'test_input640_bins':uav['size_summary']['test']['input640_bins'],
                     'official_metric':'VOC AP at IoU>=0.7, class-agnostic vehicle detection; author reference PR curve reproduced exactly',
                     'validation_budget_scenario_only':{'tag':proposal_tag,'groups':val,'images':val_count,'remaining_train_images':uav_training,
                                                        'training_split_not_yet_frozen':True}},
            'TinyPerson':{'original_images':{'train':746,'test':786},'prepared_train_val_test_images':{'train':586,'val':160,'test':786},
                          'test_normal_GT':tiny_test['normal'],'test_size_original_median':tiny_test['normal_size_median_original'],
                          'test_size_input640_median':tiny_test['normal_size_median_stretch640'],'test_input640_bins':bins(tiny_test),
                          'official_metric':'Official AP50_tiny and other TinyPerson AP/AR, original-coordinate area bins and IOD ignore matching',
                          'current_prepared_input':'Whole-image stretch640, no tiling/TTA; controlled adaptation, not the original benchmark crop recipe'},
            'cost_projection':{'reference_hours':reference_hours,'reference_job':'VisDrone lowpass_full96_seed0',
                               'reference_elapsed_seconds':reference_status['elapsed'],'batch_size':4,'steps_per_epoch':steps,
                               'datasets':budget,'UAVDT_to_TinyPerson_training_step_ratio':steps['UAVDT_proposed']/steps['TinyPerson_prepared'],
                               'limitation':'Image-step projection only; dataset I/O, augmentation, evaluation and concurrency differ. Requires actual GPU timing after selection, and excludes final independent evaluation.'},
            'recommendation':{'preferred_for_current_aerial_small_object_paper':'UAVDT',
                              'reasons':['Preserves the intended UAV detection setting with a complete official video split.',
                                         'Native evaluation is independently verified against the author reference curve.',
                                         'Keeps an additional tiling/merging design decision out of the current whole-image640 study.'],
                              'TinyPerson_role':'Additional extreme tiny-person boundary test; if used with tiling, freeze and validate the same tiling/merge protocol for every arm.',
                              'not_a_positive_result_prediction':True,'final_selection_not_committed':True,'new_training_started':False,
                              'minimum_key_pairing_proposal':['baseline','c','c_mean'],'seeds':[0,1]},
            'before_UAVDT_training':['Freeze a validation split by complete training sequences; never select weights on the official test set.',
                                     'Integrate the verified native scorer into DEIM validation and bind stage2 checkpoint selection to its declared validation metric.',
                                     'Freeze the category mapping and training ignore-region treatment; document any validation adaptation separately from the exact official test evaluator.',
                                     'Predefine size/localization diagnostics and check ignored/out-of-range GT handling; do not call the native AP70 COCO APs.',
                                     'Run CPU protocol/checkpoint tests, then measure real batch timing only after dataset selection.'],
            'source_files':[{'path':str(p.resolve()),'sha256':sha(p)} for p in sources]}
    out=DATA/'UAVDT与TinyPerson完整证据比较_20261017_v1.json'
    with out.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    uv_med=record['UAVDT']['test_size_input640_median'];tiny_med=record['TinyPerson']['test_size_input640_median']
    uv_small=uav['size_summary']['test']['input640_bins'];uv_total=sum(uv_small.values());tiny_bins=bins(tiny_test)
    text=f'''第二数据集选择建议（完整UAVDT核验之后）
日期：2026-10-17。结论：以当前无人机小目标检测论文为目标，优先推荐UAVDT；TinyPerson保留为极小人物目标的进一步适用边界实验。推荐尚未写成开训授权，第二数据集选择锁保持关闭，新训练没有启动。

已核验的完整性
UAVDT完整DET/MOT图像包6790452113字节，SHA与固定原始格式镜像一致，全部ZIP CRC通过；全部40735张JPEG已读取头部和逐图SHA，30训练序列24143图、20测试序列16592图，逐序列连续帧与GT/忽略框帧覆盖通过。原始属性M0701文件名空格仅在逻辑映射时兼容，源数据未改。
TinyPerson原版非dense训练746图、测试786图，无缺失；已准备按原训练视频组留出验证的586/160划分。官方测试不参与权重选择。

比较表
维度                     UAVDT                              TinyPerson
当前应用                 无人机车辆检测                      远距离极小人物检测
原始训练/测试图数        24143 / 16592                       746 / 786
验证集安排               从30训练视频中整组留出，待冻结       已固定586训练/160验证
测试GT数                 {record['UAVDT']['test_normal_GT']}                              {tiny_test['normal']}
640输入目标中位sqrt(wh)  {uv_med:.2f}px                             {tiny_med:.2f}px
640输入小于4px比例       {uv_small['<4']/uv_total*100:.2f}%                              {tiny_bins['percent']['<4']:.2f}%
640输入小于8px比例       {(uv_small['<4']+uv_small['4-8'])/uv_total*100:.2f}%                              {(tiny_bins['counts']['<4']+tiny_bins['counts']['4-8'])/tiny_test['normal']*100:.2f}%
原生主要评价             class-agnostic车辆AP70               AP50_tiny等TinyPerson指标
官方评价复核             作者FRCNN整条PR曲线差0               固定官方scorer及忽略/尺度单测通过

为什么当前优先UAVDT
它直接检验VisDrone之后同一应用范围内的可复现性。官方视频划分、忽略框和评价现在有完整证据；原始car/truck/bus是GT属性，原生overall匹配不读取类别字段。原生AP70较严格，但小目标主张仍应另外报告预先定义的尺度/定位诊断，不能把AP70改名为COCO APs。
TinyPerson原图正常目标中位约{tiny_test['normal_size_median_original']:.2f}px，整图压到640后约{tiny_med:.2f}px，约{tiny_bins['percent']['<4']:.1f}%低于4px。它可以检验更极端的分辨率边界；结果同时受整图压缩损失影响。作者还发布切片方案，若改用保留原尺度的切片，需要统一验证所有臂的裁剪、边界、合框和评价，现有整图640准备不能直接称为复现该切片基准。
这些判断来自任务与协议，未使用任何第二数据集模型成绩，不预测哪个数据集更容易得到正增益。

算力必须重估
参考已完成VisDrone96轮耗时{reference_hours:.2f}小时、batch4每轮1617步。预算场景按固定哈希视频组约20%留验证：UAVDT验证{val_count}图、剩余训练{uav_training}图（只是估算场景，尚未冻结训练划分）；TinyPerson已准备训练586图。
UAVDT每轮约{steps['UAVDT_proposed']}步，TinyPerson约{steps['TinyPerson_prepared']}步，相差约{steps['UAVDT_proposed']/steps['TinyPerson_prepared']:.1f}倍。按旧耗时与步数直接外推，UAVDT约{hours['UAVDT_proposed']:.1f}小时/臂、TinyPerson约{hours['TinyPerson_prepared']:.1f}小时/臂；不是新数据集实测速率，数据读取、评估与并发会改变耗时。
若先用baseline/C/C-mean两个种子共6次96轮，UAVDT外推约{hours['UAVDT_proposed']*6:.0f}卡时，两张卡理想折算约{hours['UAVDT_proposed']*6/2/24:.1f}天；若保留低通共8次，约{hours['UAVDT_proposed']*8:.0f}卡时、{hours['UAVDT_proposed']*8/2/24:.1f}天。均不含最后独立评估，不能沿用“第二数据集总共42卡时”的旧估算。开始前应在选定数据集上实测真实批次，再给可靠预算。

建议的下一步
选UAVDT作为主要第二数据集时，先冻结视频组验证划分和原生AP70/小尺度诊断口径，再准备baseline、C、C-mean同种子主配对，均从公共初始化按同一96轮配方训练。低通可作为完整8臂方案的保留项，避免为省预算破坏C/C-mean主配对。两种子只给配对描述，不能声称等效或显著；这验证跨数据集适用性，不是跨域泛化。
官方参考评价核验不等于UAVDT训练入口已经准备完成。开训前还须把原生scorer接入DEIM的验证和stage2最佳权重绑定，固定类别映射、训练忽略区域处理与验证集评价规则并通过CPU协议测试；原生官方测试保留作者规则。按序列留验证只从原训练序列中选，不能用官方测试挑模型。序列帧高度相关，后续抽样不确定性分析也应以序列为单位，不能把每帧当独立训练重复。
TinyPerson若作为后续扩展，明确是整图640的极小目标边界，或重新冻结统一的切片方案。不得根据第二数据集的中间模型结果更换数据集、种子或主指标。

来源
UAVDT作者项目页：https://sites.google.com/view/daweidu/projects/uavdt
TinyPerson作者原始数据与切片规则：https://raw.githubusercontent.com/ucas-vg/PointTinyBenchmark/TinyBenchmark/dataset/README.md
本地完整图像验收、官方参考曲线核验、原版TinyPerson核查及精确预算输入均列在同名JSON，包含来源SHA。
'''
    report=DATA/'UAVDT与TinyPerson完整证据比较_20261017_v1.txt'
    with report.open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'comparison':str(out),'report':str(report),'recommended':'UAVDT','selection_committed':False,
                     'UAVDT_input640_median':uv_med,'TinyPerson_input640_median':tiny_med,
                     'UAVDT_projected_hours_per_arm':hours['UAVDT_proposed'],'new_training_started':False},ensure_ascii=False))

if __name__=='__main__':main()
