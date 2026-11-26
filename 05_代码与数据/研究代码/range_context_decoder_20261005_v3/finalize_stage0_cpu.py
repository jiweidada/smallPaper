"""Save evidence-based Chinese review, pause navigation, and byte audit."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import bootstrap
from bootstrap import ROOT
from verify_stage0_cpu import PROJECT

OUT = PROJECT / '英文小论文/data/00_实验计划与证据索引/逐边范围实验_20261005_v1/offline_recheck_v1'
REPORT = OUT / 'DS收敛设计复核与CPU阶段0结果_20261005.txt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if REPORT.exists() or (ROOT / 'stage0_delivery_audit.json').exists():
        raise RuntimeError('Preserve previous delivery')
    functional = json.loads((ROOT / 'stage0_functional_cpu.json').read_text(encoding='utf8'))
    training = json.loads((ROOT / 'stage0_training_identity_cpu.json').read_text(encoding='utf8'))
    init = json.loads((ROOT / 'stage0_common_init_cpu.json').read_text(encoding='utf8'))
    geometry = json.loads((ROOT / 'implemented_controls_frozen_cpu.json').read_text(encoding='utf8'))
    feature = json.loads((ROOT / 'real_value_diversity_cpu_v2.json').read_text(encoding='utf8'))
    for report in (functional, training, init):
        assert all(report['checks'].values())
    checks = sum(len(report['checks']) for report in (functional, training, init))
    g = geometry['groups']['small_firstIoU_ge_0.1']['within_GT_then_equal_GT_then_equal_image']
    hard = geometry['groups']['small_firstIoU_0.5_to_0.75']['within_GT_then_equal_GT_then_equal_image']
    v = feature['groups']['small_firstIoU_ge_0.1']['metrics']
    hv = feature['groups']['small_firstIoU_0.5_to_0.75']['metrics']
    body = f'''DS收敛设计复核与CPU阶段0结果（2026-10-05）

判断：认可DS关于k=2、避免重复尺寸对照、补逐边错配和检查value多样性的方向。不能照搬它的硬闸门、同宽归因、零初始化尺度比较或“置信区间跨零即失败”。当前候选保留为可检验的网络结构改进；工程与作用空间检查通过，不等于新颖性、必要性或检测收益已经通过。

本轮仅在隔离v3中修订模块并运行CPU前向/反向与冻结轨迹分析。没有优化器更新，没有GPU、上传、远端进程操作或队列启动。两账号继续按用户要求暂停，本轮没有重新实时查GPU状态。v1、准备v2、旧协议、共享训练源码、原始/失败材料和权重均保留。

一、已实现什么

新代码目录：英文小论文/code/range_context_decoder_20261005_v3。
裁块第1层FDR逐边物理跨度 → r=q90-q10（即旧半跨度×2） → 每边均值附近3点，共12个裁块编码P3点；第2层query由对应整图P4的4点上下文条件化，再读取局部证据、零初始化投影加回query。原生cross-attention、学习偏移、FDR、LQE和输出合并保持原机制。
两次读取均改为：有有效点时只在有效点中归一化；全部无效时才使用固定零null。各头有效权重和为1；整图不再通过与始终有效null竞争单独改变总权重质量。各头仍可能通过相对权重改变不同value的组合。
新增同初始化、同参数和读取预算的更新后均值框尺寸、同轴均分、左右/上下对边互换对照。对边互换保留每查询每轴的半径多重集合和总和；不采用混淆长宽轴的四边循环换位。旧初始参考尺寸只保留离线比较，不另列一个正式训练对照。

二、CPU实现证据

功能检查{len(functional['checks'])}项、真实训练模式恒等检查{len(training['checks'])}项、公共初始化检查{len(init['checks'])}项，合计{checks}项通过。
检查包括：常值value而key/位置不同的条件下，改变query不能产生隐藏幅度门控；只有1个有效value时query不影响读出；全无效NaN/Inf坐标严格零读出；零初始化真实整图/裁块预测逐张量一致；DN/aux输出、逐项损失、BN更新和随机状态一致；原生梯度在既定容差内一致；恢复投影得到首步梯度。无优化器更新。
用非零0.05单位投影作明确的诊断干预后，真实整图输入替换可以改变裁块预测（最大logit差{functional['real_query_global_input_max_logit_change']:.6f}）；裁块上下文对照在eval下无此整图输入路径。这只证明连接可达，不证明真实语义有效。
公共seed0初始化SHA固定验证；旧第54轮EMA只用作CPU特征探针，不能作新训练共同初始化。6个配置载入共同初始化后全部状态张量一致，参数形状不增加。
新增模块仍为{functional['new_reader_parameters']:,}参数。成本需要同时报告整个协议：平均1张整图+5.1861张裁块=6.1861视图；5.1861是裁块数，不是总视图数或实测加速比。模块不新增视图前向，但投影、插值和读数不是零FLOPs。本轮未测GPU成本/时延。

三、DS所谓“同尺度”需要收窄

全170个npz重新逐个核对SHA，使用真实align_corners=False的px/8-0.5及原图/裁块掩码。实现与冻结均值框的最大差{geometry['max_mean_box_difference_px']:.8f}px，k2半径最大差{geometry['max_k2_radius_difference_px']:.8f}px。
主组3850个查询窗口/1730个GT，按同GT跨窗口均值→等GT→等图汇总。自适应半径{g['adaptive/radius_P3_cells']:.3f}个P3单元，均值框尺寸对照{g['updated_size/radius_P3_cells']:.3f}；定位困难组（首层0.5≤IoU<0.75）1373查询窗口/772GT为{hard['adaptive/radius_P3_cells']:.3f}对{hard['updated_size/radius_P3_cells']:.3f}，候选仍平均宽{(hard['adaptive/radius_P3_cells']/hard['updated_size/radius_P3_cells']-1)*100:.1f}%。
因此k=2有合理的开发尺度依据，且无需先上P2；但不是严格匹配每个查询/人群的范围宽度，也不是理论最优常数。候选胜尺寸对照，可能还包含困难查询上范围更宽的作用。要归因于逐边对应，仍需要同轴均分/对边互换等保持每轴总范围的干预。k=2选自已检查的开发验证图，不能包装成独立泛化证据。
主组三点在同一实际插值基格的边比例：自适应{g['adaptive/valid_triplet_same_base_cell_fraction']*100:.2f}%，尺寸对照{g['updated_size/valid_triplet_same_base_cell_fraction']*100:.2f}%。同基格并不等于相同双线性读数。

四、对照可辨识性复核：不能用单元集合代替读取一致

主组核L1：自适应vs更新尺寸{g['adaptive_vs_updated_size/kernel_L1']:.3f}，vs同轴均分{g['adaptive_vs_axis_mean/kernel_L1']:.3f}，vs对边互换{g['adaptive_vs_opposite_swap/kernel_L1']:.3f}。
自适应与同轴均分有{g['adaptive_vs_axis_mean/unordered_valid_base_cell_set_identical']*100:.1f}%的查询单元集合一样，但对应12点插值核全一样（容差1e-6）的只有{g['adaptive_vs_axis_mean/all_corresponding_kernels_identical_tol_1e_minus_6']*100:.1f}%。同一集合内系数、点次序和位置编码仍可不同，不能说一半查询“读数完全相同”或效应必然减半。
旧初始参考尺寸vs更新后均值框尺寸核L1={g['fixed_vs_updated_size/kernel_L1']:.4f}，集合相同{g['fixed_vs_updated_size/unordered_valid_base_cell_set_identical']*100:.1f}%，支持省掉冗余训练臂；但不是数学等价。
核L1≥0.30/identical≤30%没有经训练或统计功效校准，不设为硬阈值。几何检查能发现严格等价或退化，不会决定AP对比能否识别。对边互换几何干预更强，也不保证更大的AP效应。

五、value多样性：未发现本批样本的读数接近常值风险

复用此前固定的4张图（33、176、371、544）及全部20个窗口，不按结果筛图。全部6000查询也汇总；主小目标组233查询窗口/108GT，困难组68查询窗口/39GT。使用真实冻结旧EMA的编码P3；范围来自哈希固定的首层分布坐标。
主小目标组k1/k2的原始P3成对平均cosine为{v['offline_adaptive_k1/raw_P3/pairwise_cosine_mean']:.3f}/{v['adaptive_range_global_context/raw_P3/pairwise_cosine_mean']:.3f}；k2成对L2为{v['adaptive_range_global_context/raw_P3/pairwise_L2_mean']:.3f}，按成对向量RMS范数归一的L2为{v['adaptive_range_global_context/raw_P3/pairwise_relative_L2_mean']:.3f}。
实际新增V投影未经训练（固定seed1），k2成对cosine为{v['adaptive_range_global_context/untrained_V/pairwise_cosine_mean']:.3f}。每头分别计算相对均匀读出的最大凸包移动，组合上界/有效value的RMS范数为{v['adaptive_range_global_context/untrained_V/uniform_to_headwise_hull_bound_relative_RMSnorm']:.3f}；困难组为{hv['adaptive_range_global_context/untrained_V/uniform_to_headwise_hull_bound_relative_RMSnorm']:.3f}。该小目标样本有效点≤1比例为0，按描述性1e-3阈值判近常值的比例也为0。
定义：u_h是该头有效v的均值，B_h=max_i||v_ih-u_h||；任何每头有效softmax的读出距离均不超过sqrt(Σ_h B_h²)。这是放宽query/key可达约束后的几何上界，不是“最好/最坏任务证据”。多头权重不同，不能直接把整个128维输出看作一个共用权重的凸组合。
初始恢复投影严格为0，无法与其作有效量级比值。这里报告投影前可选择空间；不是最终残差、定位误差或AP的上界。
结论仅限本批：读取集合有明显非零变化，不支持“value接近常数导致整图通路先天空转”的担心；不需要据此强行增加P2或读取点。整图能否把这些差异选成有用定位证据，仍未知。
零query、未训练投影的路由探针中，整图vs无上下文的读出变化/有效value RMS范数为{v['routing_probe/global_vs_no_context_relative_RMSnorm']:.4f}，整图vs裁块上下文为{v['routing_probe/global_vs_local_context_relative_RMSnorm']:.4f}。实际初始路由变化远小于放宽后的上界；不是接近“最优凸组合”，也不能外推训练后收益。

六、技术失败与本轮限制，原件保留

第一次value探针使用bool掩码作减法，触发类型错误，已显式转换dtype；脚本和失败日志保留。
第二次尝试将新前向的原生query编号直接接到旧轨迹：前三图通过，图544最大框坐标差234.2px，未通过逐query对应检查。原因尚未证实，不能直接归咎于top-k或浮点，也没有放宽容差或删掉该图。
多样性及凸包上界本来不需要query，因此完成版使用固定轨迹几何+真实新采样特征。路由部分明确只用共同的零投影query，不称恢复了原生query或测了已训练上下文增益。真正网络连接可达性由第二节的独立真实预测干预验证。失败文件见v3/failed_attempts以及real_value_diversity_cpu.log、real_value_diversity_cpu_v2.log。
4图样本很少、均为已反复研究的开发材料；新V/K尚未训练；首层离线匹配不是最终COCO检测匹配；“裁块截断”不能解释为自然遮挡。不能把相关性或多样性直接写成科学发现或发表保证。

七、候选创新点与近邻边界

当前工作名：逐边范围反馈的跨视图细节读取结构。
潜在贡献是具体的信息流：首层逐边预测跨度约束下一层裁块局部读取范围，整图上下文参与有效局部证据选择，再反馈到检测query精炼。半径×2、双路输入、12点采样、null修正本身不是创新主张。
GFLv2与基线D-FINE已有分布质量信号；DAB-DETR已有框尺度调节注意力；ContourFormer已有预测轮廓限制下一层读取；UHR-DETR已有全局局部解码协作。当前差别需要落实到逐边分布跨度+两路证据选择的结构及必要性证据，不声称“预测反馈注意力”或“全局局部融合”首创。
原始论文已在英文小论文/文献_相关原始论文_20261004归档，本轮复用上述原件，没有新增论文检索或重复下载。相关原文：
GFLv2 https://arxiv.org/abs/2011.12885
DAB-DETR https://arxiv.org/abs/2201.12329
ContourFormer https://arxiv.org/html/2501.17688v3
UHR-DETR https://arxiv.org/html/2604.21435v1

八、后续条件性方案：不是训练放行

1. 只保留一个尺寸读取对照（更新后的共同均值框0.25×边长）。先与逐边k2结构作同预算短筛；若以后授权，24轮须明确为原96轮配方截断、相同初始化/数据/增强/视图/batch8/损失顺序，固定第24轮EMA评价，不能为它另缩短学习率周期或事后挑最有利的小目标权重。
2. 短筛只用于资源优先级。区间跨零记为证据不足，不直接否定96轮收敛的机制；几何宽度差、训练变化与单种子不确定性均需报告。不能按24轮结果承诺完整收益。
3. 全局必要性需早补“自适应整图上下文vs同几何裁块自身上下文”；逐边必要性补同轴均分和对边互换；整体价值还须对独立双路。它们是贡献闭合要求，不是当前自动安排全部96轮。
4. 旧第二账号完整96轮仍可作开发参照，不能因归一化没影响独立臂就自动认定严格可复用。旧损失累加顺序/恢复历史等差异未消失；正式阶梯保持配方/初始化/评价/合并/选权审计边界。
5. 指标在看新训练结果前确定：APs为总体小目标效果主指标，small区域AP@0.75为预先指定的定位机制指标，两者都报告；同时报告AP、AP75、全部12项、逐类结果。总AP75不替代small AP@0.75。small AP@0.85可作补充，不通过挑IoU阈值补救负结果。正式选权保持原总体AP规则。
6. 同两模型预测按相同图像抽样索引作配对bootstrap，每次在重采样后的数据集重新算AP；重复图像需完整复制GT/预测并重映射ID，不能平均每图AP。报告差值95%区间和点估计。图像/场景抽样不确定性不等于训练种子方差，仍需重复训练或独立种子；同权重仅裁块vs合并的0.0001–0.0096pp并不是种子噪声测量。有可靠序列/场景分组时作簇级敏感性分析。
7. CI跨0不能推出机制为假；若区间仍容许有实际价值的正提升，应记不确定。只有更充分比较持续不支持必要性，或精度足以排除事先确定的实际收益，才降低相应主张。

升级与失败边界：只胜独立而未胜尺寸读取，不能归因分布范围；胜尺寸但未胜同轴均分/错配，不能证明逐边对应；整图不优于裁块自身上下文，不能写整图—裁块协作为核心；有上界无训练收益，只保留探索/负结果。不通过改变成功口径掩盖。

下一步判断：结构性“读数几乎恒等”风险在本批样本中没有得到支持，可以继续完成可检验方案；但尚不足以放行GPU训练或宣布论文创新成立。当前先保存v3与本轮证据，等待用户后续明确恢复指令，任何新容量/训练继续暂停。

本轮代码与结果原件：range_context.py、stage0_functional_cpu.json、stage0_training_identity_cpu.json、stage0_common_init_cpu.json、implemented_controls_frozen_cpu.json、real_value_diversity_cpu_v2.json及其逐query行。
统计/采样核对文档（官方）：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html；https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html。
'''
    REPORT.write_text(body, encoding='utf8')
    before = json.loads((ROOT / 'source_before_revision.json').read_text(encoding='utf8'))
    original = Path(before['source'])
    old_unchanged = all(sha(original / name) == value for name, value in before['original_source_hashes'].items())
    sealed = json.loads((ROOT / 'sealed_source_before.json').read_text(encoding='utf-8-sig'))
    sealed_unchanged = all(sha(Path(sealed['source']) / row['file']) == row['sha256'] for row in sealed['files'])
    assert old_unchanged and sealed_unchanged
    sources = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'stage0_delivery_audit.json']
    receipt = {'status': 'stage0_CPU_pass_candidate_effectiveness_unverified_training_paused',
               'checks_passed': checks, 'original_v1_files_preserved': old_unchanged,
               'sealed_training_source_and_protocol_preserved': sealed_unchanged,
               'report': str(REPORT), 'report_sha256': sha(REPORT),
               'created_local': datetime.now().astimezone().isoformat(),
               'files': [{'relative_path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sources],
               'no_GPU_or_optimizer_or_remote_actions': True,
               'does_not_resume_paused_goal_or_training': True}
    (ROOT / 'stage0_delivery_audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
    preface = ('【CPU阶段0完成，训练暂停保持；2026-10-05】\n'
               '已在隔离range_context_decoder_20261005_v3实现条件null归一化、k2范围、共同均值框尺寸/同轴均分/对边互换控制，71项CPU检查通过；170条轨迹SHA复核、4图20窗口真实P3多样性与分头重加权上界已保存。仅支持实现和本批作用空间，尚无候选训练AP、新颖性或整图必要性结论。几何宽度不严格匹配，不能直接归因逐边。本轮无GPU、优化器更新、上传或远端操作；两账号与执行目标继续暂停，本轮未重新实时查进程。旧代码/协议/失败材料/权重保留。\n'
               '[DS分析复核与本轮完整结果](C:/Users/jiwei/Desktop/小论文投稿/英文小论文/data/00_实验计划与证据索引/逐边范围实验_20261005_v1/offline_recheck_v1/DS收敛设计复核与CPU阶段0结果_20261005.txt)。\n\n')
    # Preserve the complete previous navigation bytes in the CPU revision.
    for name in ('agent.md', '实验入口.md'):
        target = PROJECT / name
        content = target.read_bytes()
        (ROOT / ('导航更新前_' + name)).write_bytes(content)
        target.write_bytes(preface.encode('utf8') + content)
    print(json.dumps({'report': str(REPORT), 'checks': checks, 'old_v1_unchanged': old_unchanged,
                      'sealed_unchanged': sealed_unchanged, 'k1_cos': v['offline_adaptive_k1/raw_P3/pairwise_cosine_mean'],
                      'k2_cos': v['adaptive_range_global_context/raw_P3/pairwise_cosine_mean'],
                      'axis_feature_RMS': v['contrast/axis_mean_global_context/point_relative_RMS'],
                      'swap_feature_RMS': v['contrast/opposite_swap_global_context/point_relative_RMS']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
