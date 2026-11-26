"""Publish a bounded method-entry screen. No experiments or model imports."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;ENG=BASE.parent
LIB=ENG/'文献_相关原始论文_20261004'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def overwrite(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    cur=load(ROOT/'CURRENT_A53.json');prior=Path(cur['receipt']);old=load(prior)
    assert sha(prior)==cur['receipt_sha256']=='11a6cb1ff2df9cf0829406e7f2fc5494492b575f92098b7db3aab9c8ee2c23ca'
    assert all(sha(Path(p))==h for p,h in old['current_files_sha256'].items())
    legacy=ENG/'code/experiment_environment_preflight_20261008_v1/03_定位方案原文复核与交付边界.txt'
    tips=LIB/'196_TIPS_WACV2025.pdf';sett=LIB/'55_SET_CVPR2025.pdf'
    assert sha(tips)=='0e006ecd8fb74e046a687dc78625220a19a5cd35973c910075a7726d79982107'
    assert sha(sett)=='25af910ba1a4276894f65dc06bcb0fd5603aad6b1942174b3dfbef2065fadf5f'
    matrix={
      'phase':'A53','goal_status':'active','mainline_confirmed':False,
      'human_read_attested':False,'reading_actor':'assistant','systematic_review':False,
      'entries':[
        {'work':'D069 original project specification','source':str(legacy),'source_sha256':sha(legacy),
         'reading_scope':'Current original project report sections1–4 and delivery limits; no new execution.',
         'facts':'Grid-relative center and fine-geometry factorization already prepared; original FDR has center/size relations. Grid-center changes gradient scaling too.',
         'decision':'A coordinate change or repeating the existing four-arm recipe is not a new candidate. D069 remains untrained/unverified, not a fixed-endpoint negative.'},
        {'work':'TIPS','title':'Improving Shift Invariance in Convolutional Neural Networks with Translation Invariant Polyphase Sampling',
         'publication':'WACV2025','authors':['Sourajit Saha','Tejas Gokhale'],
         'url':'https://openaccess.thecvf.com/content/WACV2025/papers/Saha_Improving_Shift_Invariance_in_Convolutional_Neural_Networks_with_Translation_Invariant_WACV_2025_paper.pdf',
         'local_pdf':str(tips),'pdf_sha256':sha(tips),
         'reading_scope':'Current section3.1–3.2 selected text; physical PDF page3 visually checked. Section5.3 only the primary search excerpt, not a fresh full experiment reading.',
         'facts':'Learned weighted polyphase mixing plus regularization for sampling behavior and shifts; includes object-detection evaluation.',
         'limits':'Not reproduced; printed loss formulas3/5 and their prose are not reconciled against author code in this run. No fractional-shift or exact-equivariance guarantee inferred.',
         'decision':'Generic learnable phase pooling or adding a shift regularizer is established. A distinct operator would still be eligible if concretely justified.'},
        {'work':'SET','title':'SET: Spectral Enhancement for Tiny Object Detection','publication':'CVPR2025',
         'url':'https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf',
         'local_pdf':str(sett),'pdf_sha256':sha(sett),
         'reading_scope':'Current complete section3.2–3.3 and equations5–9; physical PDF pages4/5 visually checked. Section3.1 and experiments not fully read this run.',
         'facts':'Feature perturbations use normalized loss gradients; equation8 combines detection branches including regression, and equation9 adds an auxiliary optimization path.',
         'decision':'Small-object feature adversarial training and merely switching to localization perturbations do not yet identify our separate increment.',
         'limits':'No claim that all geometry-specific optimization is covered or ineffective; no result transferred to ECDet-S.'},
        {'work':'Making Vision Transformers Truly Shift-Equivariant','publication':'CVPR2024',
         'url':'https://openaccess.thecvf.com/content/CVPR2024/papers/Rojas-Gomez_Making_Vision_Transformers_Truly_Shift-Equivariant_CVPR_2024_paper.pdf',
         'reading_scope':'Primary abstract/first-page excerpt only, not full methods.',
         'facts':'Adaptive tokenization/attention/merging/position modules for circular shift equivariance already exist.',
         'limits':'Does not establish the same guarantee for arbitrary subpixel shifts or this detector.'},
        {'work':'Shift Equivariance in Object Detection','publication':'arXiv2008.05787 preprint',
         'url':'https://arxiv.org/abs/2008.05787','reading_scope':'Primary abstract only.',
         'facts':'Detector sensitivity to small shifts is an established research question.',
         'limits':'No evidence here that shift variance is the main cause of our baseline misses.'},
        {'work':'CLSC DETR','publication':'arXiv2608.21457 preprint',
         'url':'https://arxiv.org/abs/2608.21457','reading_scope':'Primary abstract only; no method or author code verification.',
         'facts':'Cross-layer geometric support and localization/classification consistency for UAV candidate ranking are already proposed.',
         'limits':'Exact formulas and overlap remain unverified; this is not proof against every cross-layer method.'}
      ],
      'broad_entries':[{'entry':'Center/size or dedicated-geometry factorization','historical_ids':['D024','D069','A42'],
        'status':'NO_DISTINCT_OPERATOR_SELECTED','reason':'Existing project preparation/experiments and SD-DETR/FaPN/TOOD priors; a new name or host is insufficient.'},
       {'entry':'Generic shift consistency or learnable phase pooling','historical_ids':['D007','D070'],
        'status':'NO_DISTINCT_OPERATOR_SELECTED','reason':'TIPS/Shift-equivariant ViT and shift-detection literature establish the broad mechanisms; D070 is only a limited frozen check.'},
       {'entry':'Generic sharpness-aware/localization feature perturbations','historical_ids':['D037'],
        'status':'NO_DISTINCT_OPERATOR_SELECTED','reason':'SET explicitly includes multiple branches. No independently specified new perturbation constraint or simple-control difference yet.'}],
      'screen_scope':'These are unselected proposal entries, not three trained failures and not prohibitions on whole method families.',
      'new_model_forwards':0,'new_optimizer_updates':0,'new_training_queued':False
    }
    matfile=ROOT/'06_核心方法矩阵与历史逐项对照_v1.json';save(matfile,matrix)
    pages=ROOT/'selected_method_pages_v1'
    save(ROOT/'selected_method_pages_receipt_v1.json',{
        'pdfs':{str(p):sha(p) for p in [tips,sett]},'images':{str(p):sha(p) for p in pages.glob('*.png')},
        'visual_scope':['TIPS physical page3','SET physical pages4 and5'],
        'render_warning':'Poppler reported missing ArialUnicode display font; the inspected text/formulas needed for this comparison are readable in the rendered pages.',
        'acquisition_note':'SET first web-open URL omitted the word for and was wrong; use the corrected official URL and the already present local PDF. TIPS web-open returned403; local accepted-version PDF was available.',
        'human_read_attested':False})
    report=ROOT/'07_三类入口原文筛选与后续边界.txt';assert not report.exists()
    report.write_text('\n'.join([
      'A53三类入口的有界原文筛选。总目标active；没有选定新算法候选，也没有启动新训练。',
      '沿用academic-research-suite定向研究；本轮是助手选读，不是用户已读或完整系统综述。原始返回见05，方法与阅读范围见06。',
      '',
      '已有问题证据：原版固定12的4–8输入像素小目标共10200个，IoU0.50匹配召回61.725%，IoU0.75为20.539%。这个差距不自动说明应更换哪个网络层或某种结构必然有效。',
      '',
      '入口一：中心/尺度分解、定位专用读取。D069原始报告已经准备相对网格中心与细尺度几何的分因素代码，并指出梯度尺度同时改变。D024/A42另有相关实验。仅改坐标或换宿主不能成为本次新算法；D069仍是未验证备选，不能写成训练失败。旧原文对照包含FaPN、TOOD与BlurPool，已复读项目原报告相关部分。',
      '',
      '入口二：平移一致性、可学习多相位池化。TIPS原文第3.1–3.2节已有学习相位混合与移位正则；第5.3节的主来源检索片段还说明做过检测实验。Shift-equivariant ViT也已有适配token/attention/合并/位置的设计。故宽泛的“增加相位模块/一致性损失”还不够具体；这不等于所有几何一致性训练均无价值。',
      'TIPS来源：https://openaccess.thecvf.com/content/WACV2025/papers/Saha_Improving_Shift_Invariance_in_Convolutional_Neural_Networks_with_Translation_Invariant_WACV_2025_paper.pdf',
      '阅读边界：第3页视觉核查和方法选读；未做作者代码复现，未解决式3/5与文字安排的对应问题。没有把循环整数移位结论外推到任意亚像素移位。',
      '',
      '入口三：定位相关对抗扰动/锐度正则。SET第3.2–3.3节、式5–9已使用特征损失梯度产生扰动，式8明确包含分类、回归等分支的组合。这一范围比“只对分类做扰动”更广，故简单把扰动切换为回归梯度仍没有明确的独立增量。',
      'SET来源：https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf',
      '阅读边界：完整3.2/3.3及PDF第4/5页视觉核查；未通读全部背景平滑、实验和补充材料，不把论文效果当本项目效果。',
      '',
      '当前结论：这三类宽泛提案不分配训练；尚未提出能与强简单对照清楚区分的具体算子。它们不是三项训练阴性，更不排除未来在这些基础上形成不同机制。',
      '不把“完全无人做过”或“发明新理论”设为期刊小论文的前置门槛。真正欠缺的是本次具体改动、相对最近方法及旧记录的实质差异、可承担的判别对照。',
      '下一步仅收敛少数具体操作。满足具体问题、操作差异和简单对照即可冻结最多12轮最小验证；不继续围绕宽泛术语无限检索，不用频段/层位/系数/名称重启A52。',
      '本轮新增模型前向0、更新0、正式轮0；A53累计CPU COCO评价仍为2次，未再次评价预测。全项目计数仍185612整模型前向、155854正式更新加49容量更新、含容量155903更新、192正式轮。',
      '寻找主线的目标未完成，保持active。所有旧阳性、受限阴性、未训练备选、A52固定终点与A53缓存记录保留。'
    ])+'\n',encoding='utf-8')
    archive=ROOT/'before_A53_method_screen';archive.mkdir(exist_ok=False);resolved={}
    for n in ['01_候选判别表.json','15_当前状态与结果校验.json']:
        p=BASE/n;dest=archive/n;shutil.copy2(p,dest)
        assert sha(dest)==old['current_files_sha256'][str(p)]
        resolved[str(p)]={'preserved_path':str(dest),'sha256':sha(dest)}
    board=load(BASE/'01_候选判别表.json');state=load(BASE/'15_当前状态与结果校验.json')
    status='A53_METHOD_ENTRY_SCREEN_COMPLETE_CONCRETE_OPERATOR_PENDING';now=datetime.now(timezone.utc).isoformat()
    entry={'matrix':str(matfile),'report':str(report),'selected_candidate':None,'training_queued':False,'model_operations_added':0}
    for x in [board,state]:
        x.update(status=status,goal_status='active',latest_report=str(report),current_priority_candidate=None,
            current_priority_question='依据严格定位与输出覆盖缺口收敛少数具体操作；当前尚无合格新候选。')
        x['A53_execution']['method_entry_screen']=entry
    board['latest_update_utc']=now;state['recorded_utc']=now
    overwrite(BASE/'01_候选判别表.json',board);overwrite(BASE/'15_当前状态与结果校验.json',state)
    signed=[BASE/'01_候选判别表.json',BASE/'15_当前状态与结果校验.json',report,matfile,
        ROOT/'05_候选入口补检原始记录_v1.json',ROOT/'selected_method_pages_receipt_v1.json',legacy]
    receipt=ROOT/'A53_原文筛选保留回执.json';save(receipt,{'status':'PASS','phase':'A53_METHOD_ENTRY_SCREEN','goal_status':'active','mainline_confirmed':False,
        'previous_receipt':str(prior),'previous_receipt_sha256':sha(prior),'previous_signed_current_inputs_resolved':resolved,
        'current_files_sha256':{str(p):sha(p) for p in signed},'selected_candidate':None,'model_operations_added':0,'training_queued':False})
    cur.update(status=status,receipt=str(receipt),receipt_sha256=sha(receipt),latest_report=str(report),
        next='Specify at most a few distinct operators with old-ID/difference/evidence/control comparison. Do not require a new fundamental theory; do not train a generic renamed prior.')
    overwrite(ROOT/'CURRENT_A53.json',cur)
    print(json.dumps({'status':status,'mainline_confirmed':False,'new_model_operations':0,'receipt_sha256':sha(receipt)}))
if __name__=='__main__':main()
