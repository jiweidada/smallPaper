"""Record nearest-method clearance and completed A3/A4 evidence; never runs models or refits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
NOW=datetime.now(timezone.utc).isoformat()
A3=ROOT/'A3_中心修正一对一约束核验_v1'
A4=ROOT/'A4_冻结特征精细中心偏移读出_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def archive_audit(path,expected):
    digest=sha(path)
    assert digest==expected and len(digest)==64
    with tarfile.open(path) as tar:
        members={m.name:m for m in tar.getmembers()}
        key='file_manifest.json' if 'file_manifest.json' in members else 'result_manifest.json'
        manifest=json.load(tar.extractfile(key))
        items=manifest['files'] if isinstance(manifest,dict) else manifest
        count=0
        for item in items:
            rel=item.get('file',item.get('path'))
            data=tar.extractfile(rel).read()
            assert hashlib.sha256(data).hexdigest()==item['sha256'],rel
            count+=1
    return dict(path=str(path),sha256=digest,verified_files=count,bytes=path.stat().st_size)

def main():
    history=ROOT/'history_before_A3_A4_update_v1'
    history.mkdir(exist_ok=True)
    for name in ('01_候选判别表.json','13_架构检索增补记录.json','14_架构核心文献矩阵.json',
                 '15_当前状态与结果校验.json','16_最近工作源码核查.json'):
        dest=history/name
        if not dest.exists():
            shutil.copyfile(ROOT/name,dest)
    a3=read(A3/'summary.json');a4=read(A4/'summary.json')
    assert a3['decision']=='INCONCLUSIVE_FOR_QUERY_RESOURCE_CONFLICT_NO_ROUTING_MODULE_APPROVED'
    assert not a4['encoder_gap_gate'] and not a4['fused_readable_gate']
    assert read(A4/'completed.json')['summary_sha256']==sha(A4/'summary.json')
    a3_receipt=read(A3/'backup/receipt.json')
    a4_receipt=read(A4/'backup/receipt.json')
    a3_archive=archive_audit(A3/'backup/A3_capacity_audit.tar.gz',a3_receipt['sha256'])
    a4_archive=archive_audit(A4/'backup/A4_cached_precision_probe.tar.gz',a4_receipt['archive_sha256'])
    dape=read(ROOT/'comparison_sources_v1/DAPE_operator_source_receipt.json')
    taptr=read(ROOT/'comparison_sources_v1/TAPTRv2_operator_source_receipt.json')
    for rec in (dape,taptr):
        assert rec['status']=='ALL_LISTED_OPERATOR_FILES_GIT_BLOB_VERIFIED'
        for file in rec['files']:
            assert sha(Path(rec['local'])/file['path'])==file['sha256']
    taptr_config=(Path(taptr['local'])/'config/TAPTRv2.py').read_text(encoding='utf-8')
    assert all(key+' = True' in taptr_config for key in ('use_keyaware_ca','update_pos_in_ca','ca_update_aware'))

    matrix=read(ROOT/'14_架构核心文献矩阵.json')
    rows={r['id']:r for r in matrix['rows']}
    rows['DS_Det'].update(
        read_scope='Primary abstract/introduction and complete sections3.3-3.4; selected exact decoder operators;not whole paper.236 tracked author Git blobs verified;not all code read;no imported or executed source.',
        author_method='FLET query selection;BLPCA/DPCA cooperative branches;PoCoo loss. Inspected box update remains learned delta plus inverse-sigmoid reference,followed by sigmoid.',
        our_consequence='Query selection/cooperation and its PoCoo loss are prior art;PoCoo is not an attention-coordinate centroid operator.',
        source_commit='bc945556763a5b64e951f8e57f4995094ab36dcc')
    rows['CLSC_DETR'].update(
        read_scope='Primary introduction and complete method sections2.1-2.2;selected setup;no author code audit or full experiment reading.',
        author_method='Cross-layer local candidate support and classification reliability calibrate final quality/confidence scores in the inspected equations.',
        our_consequence='Generic cross-layer geometry/score calibration is prior art. The inspected equations do not directly replace box center coordinates;indirect training effects are not excluded.')
    additions=[
        dict(id='DAPE',title='DAPE: Harmonizing Content-Position Encoding for Versatile Dense Visual Prediction',
             url='https://doi.org/10.1609/aaai.v40i6.42480',tier='AAAI2026 publisher record verified',
             read_scope='Complete SQS and LRPE method;matching and detection paragraphs;selected ablation/setup. One fixed-commit transformer source file Git-blob verified and relevant call chain read;not full source closure or appendix.',
             author_method='SQS samples multi-scale content at proposal centers;LRPE adds low-rank box relations. Source fuses sampled content with learned query embeddings.',
             our_consequence='Resampling query content at its predicted center,and generic low-rank position relations,are existing mechanisms.',
             unresolved='Not evaluated in our TinyFormer setting. Requires matched controls before any new tiny-target claim.',code_commit=dape['commit']),
        dict(id='TAPTRv2',title='TAPTRv2: Attention-based Position Update Improves Tracking Any Point',
             url='https://proceedings.neurips.cc/paper_files/paper/2024/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html',tier='NeurIPS2024 primary conference paper',
             read_scope='Abstract/introduction,complete section3.3 KDA/APU,selected ablations. Five pinned v2 operator/config files Git-blob verified;active import chain and enabled flags inspected;no build/model run.',
             author_method='Key-aware sampled attention updates content;disentangled weights aggregate sampling offsets for position. Released code also uses position-update-aware subsequent regression.',
             our_consequence='Attention-weighted coordinate update,distinct position/content weights,and adding residual regression are already covered.',
             unresolved='Given-point video tracking differs from unknown unordered object detection;task transfer alone does not establish algorithm novelty.',code_commit=taptr['commit']),
        dict(id='LiteDETR',title='Lite DETR: An Interleaved Multi-Scale Encoder for Efficient DETR',
             url='https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Lite_DETR_An_Interleaved_Multi-Scale_Encoder_for_Efficient_DETR_CVPR_2023_paper.pdf',tier='CVPR2023 primary paper',
             read_scope='Primary key-aware deformable-attention method subsection and related efficiency passages;not whole paper/source.',
             author_method='Sample keys and values and compute query-key attention,within an interleaved encoder.',
             our_consequence='Replacing query-only attention weights with sampled-key similarity is existing work.',unresolved='No measured benefit or cost in our current detector.'),
        dict(id='DSNT',title='Numerical Coordinate Regression with Convolutional Neural Networks',
             url='https://arxiv.org/abs/1801.07372',tier='2018 primary arXiv record plus author code',
             read_scope='Paper abstract only;exact dsntnn expectation/soft-argmax functions read.22 author blobs verified including documentation links as text;no project execution.',
             author_method='Differentiable expectation of a spatial distribution returns coordinates.',
             our_consequence='Spatial weighted expectation/soft-argmax itself is not a new coordinate decoder.',unresolved='Expectation does not by itself solve query-to-instance association or multiple modes.'),
        dict(id='InvariantSA',title='Invariant Slot Attention: Object Discovery with Slot-Centric Reference Frames',
             url='https://proceedings.mlr.press/v202/biza23a.html',tier='ICML2023 primary paper/project',
             read_scope='Project abstract and primary section4.1 including equations6-7;not full experiments or code.',
             author_method='Slot-relative grids;attention-mask centroid and spread update slot positions/scales.',
             our_consequence='Slot reference-frame updates from attention moments are existing work.',unresolved='Object discovery/reconstruction with few slots is not our supervised300-query UAV setting.'),
        dict(id='BFOR',title='Background-Free Objectness Learning for Class-Agnostic Detection',
             url='https://arxiv.org/html/2608.29232v1',tier='2026 primary preprint',
             read_scope='Complete sections3.2,4.1-4.2;section3.1 partial;not whole paper/source.',
             author_method='Dense objectness/size fields with GT-box-local soft-argmax center consistency and displaced size targets.',
             our_consequence='Dense center fields plus soft-argmax consistency are prior art.',unresolved='Its local-max/NMS inference differs from a DETR decoder;no transferred implementation tested.'),
        dict(id='GeoCoSAVi',title='GeoCo-SAVi: Geometry-Consistent Slot Attention for Explicitly Editable Object Representations',
             url='https://arxiv.org/html/2609.06628v1',tier='2026 primary preprint',
             read_scope='Abstract and introduction only;methods/code not read.',
             author_method='Author describes geometry/content consistency for object-centric video slots.',
             our_consequence='Generic geometry-content consistency cannot be claimed new from the concept alone.',unresolved='Specific losses/operators and overlap with any future proposal require full method reading.'),
        dict(id='DARK',title='Distribution-Aware Coordinate Representation for Human Pose Estimation',
             url='https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html',tier='CVPR2020 publisher record verified',
             read_scope='Publisher abstract and primary PDF introduction/search-visible passages;not full method or code.',
             author_method='Unbiased subpixel heatmap encoding and distribution-aware coordinate decoding.',
             our_consequence='Subpixel heatmap refinement or a generic coarse/fine center split is not new.',unresolved='Pose heatmaps are not the frozen feature vectors in A4;no equivalence asserted.'),
        dict(id='SDDETR',title='Decoupled DETR: Spatially Disentangling Localization and Classification for Improved End-to-End Object Detection',
             url='https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html',tier='ICCV2023 publisher record verified',
             read_scope='Publisher/primary abstract and Figure3 caption only;full method/code not yet read.',
             author_method='Task-aware query generation and spatially separate localization/classification feature learning.',
             our_consequence='Merely separating classification and localization paths is prior art.',unresolved='Precise sampled-point/ROI operators must be checked before a related proposal.'),
        dict(id='UnbalancedOT',title='Unbalanced Optimal Transport: A Unified Framework for Object Detection',
             url='https://openaccess.thecvf.com/content/CVPR2023/papers/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.pdf',tier='CVPR2023 primary paper',
             read_scope='Section3.1 and selected OT matching passages;not full experiments/code.',
             author_method='Transport framework for prediction-to-GT training assignment.',
             our_consequence='Sinkhorn/transport matching itself is prior art;training assignment is not automatically inference spatial readout.',unresolved='A3 does not approve a complex inference query-routing mechanism.')]
    for row in additions:
        rows[row['id']]=row
    matrix.update(updated_utc=NOW,rows=list(rows.values()),latest_empirical_boundary='A4 does not show fine offset readout superior to a constant;A1 center ranking is not coordinate sufficiency.',
                  human_read_attestation='none inferred',latest_report='22_中心定位候选去重与A3A4判断.txt')
    write(ROOT/'14_架构核心文献矩阵.json',matrix)

    audit=read(ROOT/'16_最近工作源码核查.json')
    audit.update(updated_utc=NOW)
    audit['DAPE']=dict(receipt=str(ROOT/'comparison_sources_v1/DAPE_operator_source_receipt.json'),commit=dape['commit'],verified_files=1,
                       inspected_operators=['get_spatial_prior lines544-590','query initialization lines642-656','decoder box residual lines327-350'],
                       outcome='Proposal-center content sampling already exists;source point_map plus grouped fusion inspected;no full architecture reproduction.')
    audit['TAPTRv2']=dict(receipt=str(ROOT/'comparison_sources_v1/TAPTRv2_operator_source_receipt.json'),commit=taptr['commit'],verified_files=5,
                          active_chain='taptr.py -> point_deformable_transformer.py -> deformable_transformer.py -> MSDeformAttn_KeyAware',
                          enabled_flags=['use_keyaware_ca','update_pos_in_ca','ca_update_aware'],
                          inspected_operators=['key-aware QK/softmax in ms_deform_attn.py316-397','position offset aggregation in deformable_transformer.py1066-1088','position-update-aware regression794-812','query position refresh1173-1185'],
                          outcome='APU plus disentangled weights plus subsequent residual regression and position refresh are prior art;no compiled extension or model execution.')
    audit['DS_Det_center_update']=dict(commit='bc945556763a5b64e951f8e57f4995094ab36dcc',
                                      files=['models/dino/deformable_transformer_dsdet.py','models/dino/dino.py'],
                                      outcome='Inspected path uses bbox MLP delta plus inverse_sigmoid(ref),then sigmoid;PoCoo is a loss,not a coordinate centroid operator.')
    audit['DSNT_center_update']=read(ROOT/'comparison_sources_v1/dsnt_source_verification.json')
    audit['current_novelty_boundary']='Do not claim proposal-center resampling,key-aware weighted coordinate update,soft-argmax,or generic position/content decoupling as a new algorithm.'
    audit['new_model_forwards']=0;audit['training_updates']=0
    audit['scope_this_update']='Read-only source audit. Separately,A4 fitted32 two-response diagnostic models;detector updates remain0.'
    write(ROOT/'16_最近工作源码核查.json',audit)

    log=read(ROOT/'13_架构检索增补记录.json')
    queries=[
        ['DETR tiny object detection query content position misalignment shifted sampler center uncertainty',
         'DETR multimodal attention position update soft argmax center small object detection'],
        ['tiny object detection subpixel localization phase feature downsampling encoder',
         'DETR content position alignment subpixel localization tiny object decoder'],
        ['tiny object detection subpixel center precision downsampling phase encoder',
         'Distribution-Aware Coordinate Representation human pose DARK subpixel CVPR 2020']]
    log['query_groups'].append(dict(group='A3_A4_alignment_and_precision_clearance',utc=NOW,queries=queries,
                                    method='Targeted non-systematic search;last two groups restricted to primary arXiv/CVF sources. Author repos/publisher pages followed adaptively.',
                                    result='DAPE and TAPTRv2 directly overlap generic alignment/coordinate updates;DARK discovered as a coarse/fine coordinate prior. No novelty inferred from search absence.'))
    raw_names=[
        'center_initial_query_alignment_closest_search','center_DAPE_primary_method_locator',
        'center_DAPE_and_TAPTR_author_repos_current','center_alignment_author_code_paths',
        'center_DAPE_tree_and_TAPTR_branch_metadata','center_pinned_DAPE_code_and_unresolved_search',
        'center_QCAR_full_route_and_TAPTR_code_locator','center_alignment_operator_pages_and_precision_priors',
        'center_DAPE_bibliography_and_v2_subtree','center_TAPTRv2_disentangler_full_method_locator',
        'center_precision_gap_primary_prior_clearance','center_DAPE_method_completion_and_setup']
    log['A3_A4_evidence_inventory']=[dict(file=str(ROOT/'source_audit_v1'/f'{name}.json'),sha256=sha(ROOT/'source_audit_v1'/f'{name}.json')) for name in raw_names]
    log['latest_discovery_limits']=[
        'QCAR full publisher route still403;abstract/contribution scope only. No author full text found in these bounded searches.',
        'Slim-DETR remains third-party discovery only;not used as confirmed primary evidence.',
        'DAPE full Git clone and some raw/blob requests failed TLS. One core transformer file was obtained through public blob API and Git-SHA verified.',
        'TAPTR main branch is a landing page;v2 branch was pinned from public branch metadata. Five relevant files verified,not full repository.',
        'No SSL verification disabled. Curl trial artifact name contains_openssl,but observed curl clients use Schannel;the name does not establish an OpenSSL backend.',
        'One malformed search argument call returned no tool response;only corrected successful search calls are included above.']
    log['supplement_utc']=NOW
    write(ROOT/'13_架构检索增补记录.json',log)

    board=read(ROOT/'01_候选判别表.json')
    board.update(status='FINE_CENTER_LOCALIZATION_CAUSE_UNRESOLVED_GENERIC_ALIGNMENT_NOT_NOVEL',latest_update_utc=NOW,
                 current_priority_question='微小目标的精细中心定位误差，主要来自特征中的精细位置信息不足，还是空间聚合/回归读出不足？',
                 mainline_validated=False,
                 new_empirical_evidence='A1/A2/A3/A4:185 full forwards;detector updates0;A1 fitted32 univariate probe models,A4 fitted32 two-response probe models(64 coordinate heads).')
    for row in board['candidates']:
        if row['id']=='E_CENTER_TO_BOX_SPATIAL_CORRESPONDENCE':
            row.update(status='GENERIC_OPERATOR_OVERLAP_CONFIRMED_FINE_PRECISION_PREMISE_UNPROVEN',
                       question=board['current_priority_question'],increment_approved=False,causal_evidence=False,
                       nearest_to_clear=['DAPE SQS/LRPE primary method and pinned source','TAPTRv2 KDA/APU plus position-update-aware regression,author v2 source/config',
                                        'QCAR abstract/contributions only;full text unresolved','DARK primary abstract/introduction only','SD-DETR abstract/Figure3 only'],
                       possible_increment='No specific novel operator approved. Need a measured fine-position mechanism that remains after faithful known-method controls.',
                       evidence=row['evidence']+[
                           'A3 protects original one-to-one matches:218/277 native-class center-oracle recoveries remain with free queries;ratio.787 CI[.692,.868],prespecified routing gate inconclusive.',
                           'A4 fixed472 unique-cell<16GT: P2 radial offset error1.6985px,P3_after1.6769px,constant1.5980px;both fail the constant-advantage gate. Primary P3-P2=-.0216 CI[-.2666,.1729].'],
                       correction='A1 conditional center ranking must not be described as proof of precise center information already available. A4 negative linear results do not prove encoder information loss.',
                       next='Check one explicit fine-coordinate mechanism and known local/coordinate readout priors before additional architecture probes or training;do not tune more heads until a positive appears.')
    board['rejected_generic_novelty_claims']+= [x for x in [
        'resample query content at its predicted box center alone',
        'key-aware attention weighted sampling-coordinate update alone',
        'separate content/position attention weights plus residual box MLP alone',
        'soft-argmax or generic subpixel heatmap refinement alone'] if x not in board['rejected_generic_novelty_claims']]
    board['A3_archive']=a3_archive;board['A4_archive']=a4_archive
    write(ROOT/'01_候选判别表.json',board)

    report=ROOT/'22_中心定位候选去重与A3A4判断.txt'
    primary=a4['estimates']['lt16_P3minusP2']
    text=f'''Material Passport
Origin Skill: academic-research-suite
Origin Mode: targeted non-systematic nearest-method research plus bounded cached diagnostics
Origin Date: {NOW}
Verification Status: sources/read-scope and listed local numerical checks verified; scientific mainline NOT VALIDATED
Version Label: center_alignment_clearance_A3_A4_v1

当前结论
“无人机微小目标的中心读取与空间对齐”保留为问题候选，但普通位置对齐方案的新颖性已不成立。当前更准确的问题是：精细中心定位误差来自特征信息、空间聚合还是回归读出？原因和新算法均未确定，不足以启动数天训练。
A1曾支持局部中心响应可读；A4补充核验后，不能再把它写成“精确中心坐标已经可读，只是decoder没用好”。同时，A4线性读出未胜过常数，也不能反向证明encoder丢失了信息。
数据继续为VisDrone原图；TinyFormer-S-PBM作者EMA为诊断基线，最终算法开放。压缩恢复分支维持历史状态。

1. 检索策略与记录
以“具体算子相同吗”为主线，检索内容—位置初始化、key-aware空间读取、坐标期望/位置更新及精细坐标表示，随后追溯出版方和作者仓库。新增三组、六个明确查询见13_架构检索增补记录.json；12份原始检索/页面响应已保存并列SHA。这是定向非系统调研，没有覆盖率或“查无即空白”声明。
DAPE取得出版方完整关键方法与一个核心源码文件；TAPTRv2固定作者v2分支，取得实际导入链及配置共5文件。均逐项核对Git blob，不导入、编译或执行作者项目。
限制：QCAR全文出版方仍403，仅摘要/贡献片段；Slim-DETR仍仅第三方发现，不能用于确定性去重。DAPE的整仓库克隆失败不影响已取得核心文件的逐项核验，但也不能写成整仓库已复现。

2. 核心文献矩阵
14_架构核心文献矩阵.json已更新为{len(rows)}条，逐条给出原文方法、阅读范围、我们的判断和未解决项；不推定用户已阅读。
最直接的近邻是DAPE、TAPTRv2、Lite DETR；坐标表示近邻为DSNT、Invariant Slot Attention、B-FOR。DARK、SD-DETR目前阅读范围有限，属于后续必须核查的控制方法，不能当成完整方法审查已经完成。

3. 与最接近工作的逐项对比
（原文及源码）DAPE的SQS在预测proposal中心采样多尺度内容，随后融合为query；LRPE处理box间位置关系。它已覆盖“让内容读取位置跟着预测中心走”。来源：https://doi.org/10.1609/aaai.v40i6.42480
（源码核查）DAPE b472392f4df4892d264870917c0d1e4d8dfc8568的get_spatial_prior使用bilinear grid_sample、point_map和learnable-query融合；输入memory和初始reference均detach。我们尚未在TinyFormer中训练这种改动。
（原文及源码）TAPTRv2通过key-aware相关性更新位置，并分开内容与位置权重；v2配置还启用位置更新感知的后续回归。因此“注意力中心更新＋残差回归”也不是新的组合。来源：https://proceedings.neurips.cc/paper_files/paper/2024/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html
（差别及限制）它处理给定点的视频跟踪，我们处理未知目标集合；这一任务差别本身不足以构成创新。源码固定119d1d011305e2b1397cecfd01225247c1d0a1fb，5个方法/配置文件已核验，未编译或执行。
（原文）Lite DETR已经采用采样key计算权重。把原生query-only权重改为QK相关性本身不新：https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Lite_DETR_An_Interleaved_Multi-Scale_Encoder_for_Efficient_DETR_CVPR_2023_paper.pdf
（范围有限）DARK已讨论亚像素坐标编码与解码；SD-DETR已分开定位/分类空间读取；两者本轮未完成全文/源码审查。不能把“粗中心＋精细偏移”或分支拆分单独包装成创新。
（已排除误解）DS-Det的PoCoo是分类/IoU/尺度相关损失，不是坐标质心读出。CLSC已读完整2.1-2.2，检查到的直接作用是质量/置信分数校准；不能因此排除其间接训练效应。

4. 有依据的未解决问题，以及证据不能支持什么
（实测A3）A2的逐GT真值修正可能重复借用同一query。A3固定原生flatTop300类别/分数、每个实体query最多匹配一GT，并保护原有一对一正确覆盖。原条件队列470GT中，有277个具有相应真值中心修正潜力；保留原匹配后可用空闲query恢复218个，占78.70%，95%文件前缀组bootstrap区间[69.16%,86.83%]。允许重分配而保护覆盖时为220/277。
（推断）预设低冲突判据要求比例>=80%且区间下界>=65%；主结果未通过，也不符合高冲突判据。结论是未决，不能据此增加复杂路由/transport模块。这是真值上界审计，不是AP增益。
（实测A4）只用已保存的16图特征；复用4个分组折，472个合格且不共享4px中心格的<16GT，203个<8、269个8-16。给定GT所在格，只让192维特征预测格内dx/dy；每图等权、ridge0.01、训练折标准化，真实与图内联合打乱偏移作对照。32个二维线性读出模型，共64个坐标头，无超参搜索。
（实测A4主结果）平均径向误差：P2=1.6985px，融合后P3=1.6769px；训练折常数偏移=1.5980px，零偏移=1.5775px。预定P3减P2差值{primary['mean']:.4f}px，95%区间[{primary['low']:.4f},{primary['high']:.4f}]。二者虽优于打乱标签模型，但均未优于常数；预设encoder精度差和融合特征精细可读性两道判据均失败。
（推断）本轮没有得到支持新增encoder或decoder模块的精细位置证据。A1的粗中心排名与A2的真值修正空间之间，仍缺“可从实际输入可靠预测所需位移”的环节。
（待验证）局部邻域结构/非线性读出是否可恢复精细坐标；原生训练分配与监督是否让已有回归头未充分学到位置；这些解释仍竞争。不能由一次负线性探测证明信息消失，也不能把本模型的诊断缺口宣称为无人做过的学科空白。
局限：作者checkpoint已见过这些原训练图；分组只隔离读出拟合。GT给出正确4px格且排除了同格实例，故本试验不证明未知目标召回、密集实例分离、独立泛化或实际AP。

5. 候选增量、可行性与风险
淘汰为独立创新：普通中心热图、按预测中心采样、key-aware坐标加权更新、内容/位置权重分开再接残差MLP、泛称低秩位置关系或亚像素修正。
保留的只是有条件的设计空间：如果能证明现有空间聚合不能表达或不能读取某种可恢复的精细位置结构，再设计对应的编码或解码机制；目前没有批准具体算子，也没有新算法有效性结论。
主要风险：近邻重合；用GT条件读出冒充全图能力；小样本线性头不适配；新增参数/采样点/训练预算解释全部收益；作者checkpoint和开发集选择带来的乐观偏差。
单卡24GB：既有严格EMA前向已能运行；完整训练峰值显存、原生loss与吞吐仍未实测。不能把推理显存或作者其他GPU配方当作训练可行性通过。

6. 最小验证路线与停止规则
当前不启动新架构长训练。先查完整DARK/SD-DETR等精细/局部读出控制，提出一个明确的竞争机制和一次固定预算干预；不因A4阴性继续搜索更多头直到出现阳性。
如果保留精细定位分支，最低要求是区分“同一特征中信息可恢复但原读出没用好”与“需要新增位置证据”；必须有无GT位置提示的读出验证、精度与参数/采样/算力匹配控制，并保留失败对照。若只是已知读出能解决，应作为强基线，不能改名为新方法。
只有形成非平凡方法差别后，才冻结VisDrone原图上的原生基线、最接近既有方法和候选的同配方比较；先做真实训练前反向容量/吞吐检查，再在数天资源上限内安排预算。正式结论需要统一AP/APs、<8与8-16召回/定位、漏检与误检、耗时显存；使用固定开发集与独立最终评估，不以当前训练图probe代替。
如果近邻已覆盖具体算子、必要控制消除收益，或收益完全由预算解释，就淘汰该候选并回到开放编解码器搜索。保留原200次前向诊断上限；本轮未消耗剩余15次。

7. 可核验参考与本地证据
DAPE：Hou et al.,AAAI2026,40(6):4780-4788,DOI10.1609/aaai.v40i6.42480；https://github.com/xiuqhou/DAPE
TAPTRv2：Li et al.,NeurIPS2024；https://proceedings.neurips.cc/paper_files/paper/2024/file/b7216f4a324864e1f592c18de4d83d10-Paper-Conference.pdf；作者v2：https://github.com/IDEA-Research/TAPTR/tree/v2
Lite DETR：Li et al.,CVPR2023；上文CVF原始PDF。
DSNT：Nibali et al.,2018；https://arxiv.org/abs/1801.07372；作者代码https://github.com/anibali/dsntnn
Invariant Slot Attention：Biza et al.,ICML2023；https://proceedings.mlr.press/v202/biza23a.html
B-FOR：2026预印本；https://arxiv.org/abs/2608.29232
CLSC DETR：2026预印本；https://arxiv.org/abs/2608.21457
QCAR：出版方摘要，DOI10.1016/j.neucom.2026.134465；https://doi.org/10.1016/j.neucom.2026.134465
DARK：Zhang et al.,CVPR2020:7093-7102；https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html
SD-DETR：Zhang et al.,ICCV2023:6601-6610；https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Decoupled_DETR_Spatially_Disentangling_Localization_and_Classification_for_Improved_End-to-End_ICCV_2023_paper.html
本地A3：A3_中心修正一对一约束核验_v1/protocol.json、summary.json及backup；归档SHA {a3_archive['sha256']}，9个文件重读核验。
本地A4：A4_冻结特征精细中心偏移读出_v1/protocol.json、summary.json、verification.json及backup；归档SHA {a4_archive['sha256']}，25个文件重读核验；472GT、5664个误差值、16模型文件及15图主结果重聚合通过。
完整阅读范围/检索记录/源码核查分别见14、13、16号JSON。研究状态以15_当前状态与结果校验.json为准；20号报告保留为A1/A2当时的历史判断，由本文件更新其主线解释。

资源及目标状态
本轮新增GPU前向0、检测模型更新0。架构诊断累计185/200次完整前向；A1拟合32个一维线性模型，A4拟合32个二维线性模型，合计64个诊断模型/96个标量响应头；它们不是检测器训练。A2/A3不增加拟合。
上一目标回合仅回答状态，归为no progress；本回合完成A4并取得DAPE/TAPTRv2关键源码与排重结论，归为progress。总目标维持ACTIVE：尚缺具体机制、非平凡算法差异、有效性与24GB训练可行性，未标记完成，也不将查证困难标记为blocked。
AI披露：单助手执行检索、编码、诊断与整理；没有子代理，没有独立人工同行复核，未推定用户阅读过任一文献。
'''
    report.write_text(text,encoding='utf-8')
    state=read(ROOT/'15_当前状态与结果校验.json')
    phase=dict(utc=NOW,goal_status='active',classification='progress',complete_mainline=False,
               previous_goal_turn='no_progress_status_answer_only',full_model_forwards=185,global_cap=200,
               main_model_training_updates=0,new_architecture_training_runs=0,
               CPU_diagnostic_linear_models=64,scalar_response_heads=96,
               accounting='A1:32 univariate models;A4:32 two-response models/64 coordinate heads. Matching/oracle analyses are not fits.',
               A3=a3_archive,A4=a4_archive,A3_decision=a3['decision'],A4_decision=a4['decision'],
               A4_verified_error_values=5664,latest_report=str(report),
               next='Identify a specific fine-coordinate mechanism and faithful closest-method control;no further head search or architecture training approved by the current scientific evidence.')
    state.update(recorded_utc=NOW,goal_status='ACTIVE',previous_goal_turn='no_progress_status_answer_only',
                 current_goal_turn='progress_A3_integrity_A4_completed_nearest_source_overlap_confirmed',mainline_validated=False,
                 model_forwards=185,training_updates=0,latest_report=str(report),latest_authoritative_phase=phase,
                 training_update_scope='Main detector updates0;64 fitted diagnostic linear models with96 scalar response heads across A1/A4.')
    write(ROOT/'15_当前状态与结果校验.json',state)
    receipt=dict(status='A3_A4_AND_NEAREST_METHOD_CLEARANCE_RECORDED',utc=NOW,goal_status='ACTIVE',
                 mainline_validated=False,report=str(report),report_sha256=sha(report),
                 matrix_records=len(rows),DAPE_verified_operator_files=1,TAPTRv2_verified_operator_files=5,
                 A3=a3_archive,A4=a4_archive,full_model_forwards=185,main_model_updates=0,
                 diagnostic_linear_models=64,scalar_response_heads=96,
                 remaining=['concrete causal mechanism','nontrivial closest-work difference','algorithm efficacy','training capacity/throughput'])
    write(ROOT/'23_A3A4完成与近邻核查回执.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    main()
