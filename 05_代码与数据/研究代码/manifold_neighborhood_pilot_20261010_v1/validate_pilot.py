"""Post-result integrity checks; no changed scientific protocol or new fit."""
from pathlib import Path
import csv,datetime,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'outputs/manifold_neighborhood_pilot_20261010_v1'
CAP=ROOT/'outputs/model_artifact_recorder_20261010_v1/collection_seed1/capture_seed1_final132_images0_1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    protocol=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
    assert sha(HERE/'pilot.py')==protocol['code_sha256']==summary['code_sha256']
    assert sha(OUT/'protocol.json')==summary['protocol_sha256']
    manifest=read(CAP/'manifest.json');assert sha(CAP/'manifest.json')==protocol['source']['source_manifest_sha256']
    assert all(sha(CAP/k)==v for k,v in manifest['files'].items())
    with (OUT/'per_point_neighborhood.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==summary['rows']==3328
    metrics=['k10_background_fraction','background_fraction_minus_prior','k10_same_label_fraction','same_label_fraction_minus_prior','local_PCA_effective_dimension','local_top3_variance_fraction']
    for group in summary['groups']:
        subset=[r for r in rows if r['stage']==group['stage'] and int(r['image'])==group['image'] and r['input_size_bin']==group['input_size_bin']]
        valid=[r for r in subset if r['valid_cosine_sample']=='True']
        assert len(subset)==group['points'] and len(valid)==group['valid_cosine_points']
        for key in metrics:
            values=[float(r[key]) for r in valid if r[key]!='']
            expected=float(np.mean(values)) if values else None
            assert expected==group[key+'_mean'],(group,key,expected)
    duplicate_checks=[]
    for image in (0,1):
        with np.load(CAP/f'image_{image:06d}/tensors.npz',allow_pickle=False) as z:
            keys=['encoder.fpn_blocks.1@0.value','encoder@0.value.0','decoder.input_proj.0@0.value']
            tensors=[z[k] for k in keys]
            assert all(np.array_equal(tensors[0],v) for v in tensors[1:])
            duplicate_checks.append({'image':image,'full_tensor_keys':keys,'shape':list(tensors[0].shape),'full_array_equal':True,'all_three_share_sha256':hashlib.sha256(tensors[0].tobytes()).hexdigest()})
    stage=summary['stages'][0]
    tiny=[r for r in rows if r['stage']==stage and r['input_size_bin']=='lt4']
    assert len(tiny)==11 and sum(int(r['image'])==0 for r in tiny)==9
    unsupported=[r for r in rows if r['stage']==stage and r['kind']=='GT_center' and float(r['reference_same_label_prior'])==0]
    fallacies={
      '1_Simpson':'Two images, all size/class records and priors retained separately. No pooled neighborhood change used as universal collapse.',
      '2_Ecological':'Eleven tiny annotations from2images are not11independent scenes. Repeated features/stages are not independent data.',
      '3_Berkson':'Existing two-image engineering fixture, not representative image sample. Its selection and background-coordinate rule are disclosed.',
      '4_Collider':'GT-center/background selection and other-image neighborhood composition can bias observed associations.',
      '5_Base_rates':'Background and same-class fractions interpreted with available-reference priors. Missing reference classes explicitly flagged.',
      '6_Regression_to_mean':'One mature checkpoint only; no before/after trained intervention or chosen best epoch.',
      '7_Survivorship':'All288GT plus128coordinate controls retained; all3328 rows nonzero, no feature-dependent replacement. Duplicate tensors disclosed.',
      '8_Look_elsewhere':'Many earlier negative routes remain closed. This pilot has no mechanism-priority success threshold or AP claim.',
      '9_Forking_paths':'Protocol before neighborhood metrics. Duplicate-array and CSV checks after results are disclosed integrity checks only.',
      '10_Correlation_causation':'A PCA participation ratio or kNN purity is not curvature, causal localization information loss, or evidence an attention module improves AP.',
      '11_Reverse_causality':'The embedding may reflect native learned losses, scene composition and object scale. Purity/low dimension cannot identify the layer causing localization errors.'}
    report={'status':'PASS_PILOT_FILE_GROUP_PARITY_AND_DUPLICATE_TENSOR_DISCLOSURE','material_passport':{'skill':'academic-research-suite','workflow':'experiment-agent','mode':'run+validate','scientific_scope':'ANALYZED_PILOT_ONLY','engineering_verification':'VERIFIED_SOURCE_HASHES_CSV_GROUPS_AND_FULL_DUPLICATE_TENSORS'},'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files_verified':len(manifest['files']),'CSV_rows':len(rows),'groups_recomputed':len(summary['groups']),'input_short_lt4_GT':11,'image0_tiny':9,'image1_tiny':2,'duplicate_checks':duplicate_checks,'distinct_stage_feature_boundaries':6,'missing_reference_class_GT':[{'image':int(r['image']),'annotation_id':int(r['annotation_id']),'category':int(r['category'])} for r in unsupported],'fixed_equal_radius_remapping_rank_changes':summary['fixed_radius_distance_remapping_neighbor_changes'],'fallacy_scan':{'checked':11,'total':11,'items':fallacies},'AP_evidence':False,'curvature_measured':False,'root_cause_established':False,'model_forward_calls':0,'optimizer_steps':0,'epochs_added':0}
    save(OUT/'validation_report.json',report)
    text='''流形邻域分析：两图方法试算，不能判定本质问题

材料：seed1成熟BASE132，普通验证图0/1；288个GT、128个未标注坐标对照。小于4输入像素的目标只有11个（9+2）。采样的是GT中心附近特征，并非纯目标嵌入，也不是精确感受野。
统计：跨图k10同类/背景邻域比例，并减去另一图的类别/背景先验；k32局部PCA有效维数。全部3328条记录保留，80组均值从CSV复算一致。

试算没有支持“特征随网络统一塌缩”的结论。两图中，从encoder输入投影到FPN，同类邻域比例均增加，同时有效PCA维数下降。维数下降本身不是信息丢失；同类邻域变纯，也不等于精确定位信息保留。
encoder FPN P3、encoder输出P3、decoder输入投影P3在两图整张张量逐元素完全相同；8个命名边界只有6份不同特征，不能重复计作三份支持证据。

把所有归一化特征映到相同半径r=.5的Poincare球，仅改变距离公式，8个命名阶段的完整邻居排序变化均为0。这是等半径距离的单调关系；不能据此否定真正学习不同特征/半径的双曲模型。
本试算没有测曲率，没有证明流形存在，没有预测AP，也没有确定问题发生在哪层。图间类别/场景差异很大，缺失参考类别已单列。

可用的思路：分类邻域结构和位置/尺寸可辨识性分别检查。后续需要固定双seed图集、控制输入位移/缩放的特征响应，并用干预及检测结果检验定位假设。当前专门encoder/decoder设计属于待验证假设，不能从本试算直接推出它有效。

代码与冻结协议：protocol.json、pilot.py；逐点结果：per_point_neighborhood.csv；完整核验与11项统计误区审查：validation_report.json。新增前向/训练轮/优化器更新全部为0。
'''
    (OUT/'结论与边界.txt').write_text(text,encoding='utf8')
    files=['protocol.json','per_point_neighborhood.csv','summary.json','validation_report.json','结论与边界.txt']
    save(OUT/'delivery.json',{'status':report['status'],'files':{n:sha(OUT/n) for n in files},'code':{n:sha(HERE/n) for n in ['pilot.py','validate_pilot.py']},'scientific_scope':'ANALYZED_PILOT_ONLY'})
    print(json.dumps({'status':report['status'],'rows':len(rows),'tiny_GT':11,'distinct_boundaries':6}))
if __name__=='__main__':main()
