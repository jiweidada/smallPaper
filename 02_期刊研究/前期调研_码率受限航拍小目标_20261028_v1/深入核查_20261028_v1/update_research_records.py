"""Reconcile literature/report records only. Never loads or executes model code."""
from pathlib import Path
import json, hashlib, shutil, re

root = Path(__file__).resolve().parent
base = root.parent
archive = base.parent / '文献_相关原始论文_20261004'
history = base / '历史版本_本次全文核查前'
history.mkdir(exist_ok=True)
names = ['00_调研报告.txt', '01_核心文献矩阵.txt', '02_检索与证据记录.json',
         '03_来源核验与原件存档.json', '04_主线收敛与数据算法选择.txt',
         '05_YOLO26选择与选题来源说明.txt', 'literature_records.json']
for name in names:
    if not (history / name).exists():
        shutil.copy2(base / name, history / name)

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

records = load(history / 'literature_records.json')
by_id = {r['id']: r for r in records}
by_id['R2'].update(
    data='VisDrone2019-DET 6471/548，10类；DOTA-v1.5作为相邻航空遥感证据',
    algorithm='单套学习式codec；5×5低通作频率分离，全图MSE+重建高频的冻结YOLOv5s检测loss；factorized熵模型',
    mutable='codec权重；YOLOv5s先训练300轮后冻结，检测梯度仍传至codec；不是双路传输',
    rate='先缩至640×640再编码；表5相近文件大小非严格相同bpp；报告真实编码流程，未审作者比特流实现',
    size='正式主要指标总体mAP@0.5，不是AP_s；resize是否保宽高比未充分说明',
    evidence='表5 VisDrone FSIC 24.2 mAP@0.5/3.7KB；表1原始高频重组仅为借原图信息的诊断；表6为编码耗时',
    limit='自有正式训练轮数/HF数值处理/完整代码未充分明确；不能把640域bpp与原尺寸JPEG bpp直接比较',
    reading='用户提供9页出版商PDF；正文p1–8通读；p3–8公式、图表目视核对；p9参考文献未逐条外查。结构PASS仅结构可读，未独立复现。',
    pdf_pattern='20261028_深入核查_FSIC_用户提供.pdf', abstract_only=False,
    full_body_read=True, human_read_attestation=False)
by_id['R16'].update(
    venue='Scientific Reports；本次阅读用户提供的13页Accepted Manuscript / Article in Press',
    data='HomeObjects-3K，室内12类，2285/404/311；不是UAV；接收稿两套退化协议有冲突',
    algorithm='DP从学生neck预测合成退化参数；条件IRB用L1/GAN/VGG恢复监督；训练后DP与IRB均移除',
    mutable='学生backbone/neck与辅助分支训练；式1–8未定义固定干净图检测教师的显式特征/logit蒸馏项',
    rate='包含JPEG参数监督，但未核到实际文件字节/bpp协议；p6 q30–95与p7 q5–50不一致',
    size='表2报告AP_s 23.6→32.9；具体面积坐标及完整AP实现不足以直接对齐本方案',
    evidence='图1/式1–8明确退化预测+条件恢复辅助训练；表2 AP39.3→52.2为原文报告值，未经本项目复现',
    limit='p6–7有倍率/JPEG范围冲突，并出现人脸与微表情训练段；系数/网络/阶段未充分给出；仅降低复现可确定性，不据此否定所有结果或声称我们有创新',
    reading='用户提供13页接收稿；正文p2–11通读，p4–10公式/图1/表1–5原页目视；p1版本及p12–13参考/声明已读。非最终校样认证，未独立复现。',
    pdf_pattern='20261028_深入核查_PDAD_用户提供.pdf', abstract_only=False,
    full_body_read=True, human_read_attestation=False)
by_id['R19'].update(
    limit='论文仍仅摘要；源码静态核查不证明实际航拍效果或显存；版本与推理分支必须冻结',
    reading='论文仅官方摘要/元数据；另完成v8.4.191核心静态源码调用链与v8.4.24分配对照；augment/val为固定tag官方raw网页选读，部分下载失败。未运行代码。',
    code_read=True, paper_abstract_only=True, code_commit='2004cba021c68a46429be460e26954ab171b2862')
records.append({
    'id': 'R20', 'tier': '补充关键机制查重：辅助恢复训练', 'short': 'AERIS',
    'authors': 'Cui, Z., Zhu, Y., Gu, L., Qi, G.-J., Li, X., Zhang, R., Zhang, Z., & Harada, T.',
    'year': 2022,
    'title': 'Exploring Resolution and Degradation Clues as Self-supervised Signal for Low Quality Object Detection',
    'venue': 'ECCV 2022, 473–491', 'doi': None,
    'url': 'https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690465.pdf',
    'data': 'COCO/KITTI，相邻场景，非UAV实际码率验证',
    'algorithm': '恢复辅助任务联合检测，推理不显式重建；有退化训练与混合训练比较',
    'mutable': '训练表征及检测/恢复解码器；部署保留encoder/upscaling/detection路径',
    'rate': '本次未核实际码流预算；不作为JPEG同载荷数值对照',
    'size': '本次不转引未目视核对的定量结果',
    'evidence': '再次约束“恢复辅助训练、推理去恢复支路”的宽泛新颖性',
    'limit': '有限选读；不得推断与PDAD每个细节相同；RestoreDet预印本关系未逐版核清，不重复计效果',
    'reading': '官方摘要、§4.3推理段和作者仓库README有限选读；未通读方法/实验或核表格，PDF本地存档TLS失败。',
    'pdf_pattern': None, 'abstract_only': False,
    'code_url': 'https://github.com/cuiziteng/ECCV_AERIS', 'human_read_attestation': False
})
save(base / 'literature_records.json', records)

labels = [('tier','研究位置'),('title','文献'),('authors','作者'),('year','年份'),('venue','发表载体/版本'),
          ('url','原始链接'),('data','数据与任务'),('algorithm','方法'),('mutable','可变部分'),
          ('rate','码率口径'),('size','小目标口径'),('evidence','原文证据'),('limit','限制'),('reading','实际阅读范围')]
lines = ['核心文献矩阵与阅读范围 v3', '日期：2026-10-28（America/Los_Angeles）',
         '共20条；R2与R16新增全文正文阅读，R19增加静态源码核查，R20为有限选读。不是系统综述的完整收录集。',
         '下载、文本提取、结构PASS均不等于全文阅读；按每条实际范围判断。原文结果不是本项目实验。', '']
for item in records:
    lines += [f"{item['id']} | {item['short']} | {item['year']}"]
    lines += [f'{label}：{item.get(key)}' for key,label in labels]
    if item.get('doi'):
        lines += ['DOI：' + item['doi']]
    lines += ['仅摘要：' + ('是（论文范围；如有源码阅读另列）' if item['abstract_only'] else '否；以实际阅读范围为准'), '']
(base / '01_核心文献矩阵.txt').write_text('\n'.join(lines), encoding='utf-8')

source = load(history / '03_来源核验与原件存档.json')
old = {r['id']: r for r in source['records']}
receipts = []
inventory = source['pdf_inventory'][:]
for ref, name in [('R2','FSIC'),('R16','PDAD')]:
    row = load(root / (name + '_user_source_receipt.json'))
    actual = hashlib.sha256(Path(row['archive_path']).read_bytes()).hexdigest()
    if actual != row['sha256']:
        raise ValueError('Archived paper hash mismatch')
    row.update(reading_scope=next(r for r in records if r['id']==ref)['reading'],
               full_body_read=True, human_read_attestation=False,
               visual_pages=list(range(3,9)) if ref=='R2' else list(range(4,11)),
               structural_only_preflight=True)
    save(root / (name + '_user_source_receipt.json'), row)
    row['ref_id'] = ref
    row['doi'] = next(r for r in records if r['id']==ref)['doi']
    row['original_url'] = 'https://doi.org/' + row['doi']
    receipts.append(row)
    inventory.append({'file':row['archive_path'],'sha256':row['sha256'],'pdf_pages':row['pdf_pages'],
                      'text_file':row['text_path'],'preflight_sidecar':row['preflight_path'],
                      'text_extraction_only':False,'read_scope':row['reading_scope'],'structural_verdict':'PASS',
                      'structure_scope':'STRUCTURE_ONLY, not scientific validation','ref_id':ref,'human_read_attestation':False})

source['records'] = []
for item in records:
    row = dict(old.get(item['id'], {}))
    row.update(item)
    row['human_read_attestation'] = False
    if item['id'] in ['R2','R16']:
        local = next(p for p in inventory if p['ref_id']==item['id'])
        row.update(local_pdf=local, archive_status='user_supplied_pdf_archived_and_body_read',
                   source_check='Local user-supplied original plus publisher metadata; no independent replication')
    elif item['id']=='R20':
        row.update(local_pdf=None, archive_status='PDF_download_TLS_failed; official_web_excerpts_only',
                   source_check='Official ECVA and author repository limited excerpts')
    source['records'].append(row)
source['pdf_inventory'] = inventory
source['counts'].update(total_references=len(records), local_pdfs=len(inventory), new_local_pdfs=8,
                        full_body_read_added_this_audit=2)
source['notes'].append('FSIC/PDAD supplied by user, locally archived without external upload; old access failures retained as historical attempts.')
for filename in ['complete_static_receipts.json','last_source_receipts.json']:
    source['retrieval_attempts'].extend(load(root / filename))
source['new_paper_receipts'] = receipts
source['static_audit_report'] = str(base / '06_FSIC_PDAD_YOLO26核查与主线判断.txt')
save(base / '03_来源核验与原件存档.json', source)
save(archive / '20261028_深入核查_FSIC_PDAD_原件与阅读记录.json', receipts)

search = load(history / '02_检索与证据记录.json')
search['boundaries']['hardware_budget'] = 'User confirmed single approximately 24GB GPU; training a few days acceptable; hardware/throughput not measured'
search['batches'].append({
    'id':'S_fulltext_static_followup_20261028', 'status':'completed within recorded access limits',
    'queries':['"RestoreDet" "degradation equivariant"',
               '"Exploring Resolution and Degradation Clues" "object detection"',
               'VisDrone DET evaluation ignore regions GitHub evalRes.m'],
    'tool':'web.run search_query plus primary source open/find; local supplied PDFs',
    'snapshot':str(root/'sources'/'S_补充近邻与评价检索.json'),
    'additional_primary_snapshots':[str(p) for p in sorted((root/'sources').glob('S_*.json'))],
    'prior_deep_retrieval_snapshot':str(root/'sources'/'early_deep_web_evidence.json'),
    'findings':'Full FSIC/PDAD body reading; pinned YOLO26 core source audit; AERIS mechanism prior art; no empirical detector claim',
    'limitations':'No exhaustive hit counts; some static files only web excerpts; additional git-tree verification unavailable; AERIS local PDF download failed'
})
search['reading_scope_updates'] = {r['id']:r['reading'] for r in records if r['id'] in ['R2','R16','R19','R20']}
search['execution_this_audit'] = source['execution']
save(base / '02_检索与证据记录.json', search)

report = (history / '00_调研报告.txt').read_text(encoding='utf-8')
report = report.replace('定向前期调研报告 v2（按用户追加的资源条件与主线问题补检）', '定向前期调研报告 v3（新增FSIC/PDAD正文通读与YOLO26源码核查）')
report = report.replace('文献编号R1–R19', '文献编号R1–R20')
report = report.replace('R2、R3、R7、R13、R14、R18主要依据出版商可检索的选段；完整协议和原表未全部核清。R9、R10、R12、R16仅摘要/元数据，矩阵明确标注。',
    'R2与R16已取得用户提供PDF并通读正文；R16为接收稿，其实验协议存在前后冲突，见06核查报告。R3、R7、R13、R14、R18仍主要依据出版商可检索选段。R9、R10、R12论文仅摘要/元数据，矩阵明确标注。R20为官方摘要/推理段/作者README有限选读。')
report = report.replace('R2 FSIC，2025 | 小目标任务感知压缩 | 高频分离+检测监督已有先例 | 出版商摘要/引言/数据/结论预览；非全文',
    'R2 FSIC，2025 | 小目标任务感知压缩 | 单套codec，全图MSE+高频检测loss，检测器冻结 | 用户PDF正文p1–8通读，p3–8公式图表目视')
start = report.index('3.2 R2 FSIC')
end = report.index('3.3 训练侧', start)
report = report[:start] + '''3.2 R2 FSIC：全文确认的是codec训练，不是JPEG后处理

【原文，p3–6】单套学习式codec使用全图MSE和重建高频分量检测loss，YOLOv5s预训练300轮后冻结，梯度仍可传至codec；不是双码流。先缩到640再编码，正式主要报告总体mAP@0.5。p7表5是相近文件大小，不能说严格同bpp；表6是编码耗时。p3原始高频重组是借原图信息的诊断，不是同载荷可部署恢复器。
【推断】高频/边缘与检测感知编码已经有明确先例；发送端接口与JPEG后处理不同，但用户未限定必须JPEG，不能人为用兼容性绕过该工作。
【仍未充分说明】FSIC自身训练总轮数、高频数值处理、resize细节与作者代码；详见06，不代填。现在已经取得并阅读正文，不再沿用早期“未拿到全文”的状态。

''' + report[end:]
start = report.index('3.5 用户追加')
end = report.index('4. 有依据', start)
report = report[:start] + '''3.5 新增全文核查后的训练侧近邻

【原文】LAD已有教师指导标签分配，GID已有实例选择蒸馏。PDAD正文明确DP参数预测与条件IRB恢复辅助训练，部署移除二者；式1–8未给出可还原的固定干净图检测教师特征/logit损失。接收稿p6–7出现两套退化范围和人脸/微表情训练叙述冲突，降低复现可确定性，不据此否定全部结果。AERIS[R20]也已有恢复辅助训练与推理不显式重建。
【推断】“按实例蒸馏”“退化感知”“训练辅助、推理去分支”均不能单独成为贡献。PDAD不易直接复现并不降低我们的查重标准。仍未核到直接UAV同码流、同版本、同预算下恢复对已适配检测器互补性的完整对照；这只是局部证据缺口。
更新状态：VisDrone2019-DET + YOLO26s；具体算法增量仍未确定。06给出逐项核查，07给出两阶段最小验证设计，当前没有新模型诊断结果。

''' + report[end:]
report = report.replace('G1：强基线之间缺少在同一问题上的受控比较。', 'G1：直接UAV近邻中，同文件且适配后的恢复互补性尚未闭合。')
report = report.replace('【推断】在保留同一JPEG载荷的条件下，检测器适配和恢复是否互补、收益是否仍在，值得验证。这个缺口支持一个比较研究，尚不能单独支撑“算法改进”论文。',
    '【推断】在同一JPEG载荷、同版本、同预算下，检测器适配后恢复是否仍互补，值得验证；不是声称全领域从未比较适配与恢复，AERIS已有相邻比较。此缺口本身不能单独支撑算法改进论文。')
# Supersede the old unfrozen recipe while preserving it in the history copy.
section6 = re.search(r'(?m)^6\. ', report)
section7 = re.search(r'(?m)^7\. ', report)
if section6 and section7 and section7.start()>section6.start():
    report = report[:section6.start()] + '''6. 最小验证设计（本次已细化；未执行）

见07_VisDrone_YOLO26s最小验证方案.txt与08_最小验证参数.json。固定VisDrone fit6171/dev300/val548、YOLO26s、v8.4.191 commit2004cba021c68a46429be460e26954ab171b2862、640输入。
阶段A只做D0原始训练与DA随机JPEG训练，两种配方各seed0/100轮；Q15/Q20/Q70及reference，附WebP/缩放JPEG淘汰控制。只有留下明确问题才做一个已有BRR恢复器，分别检查对D0/DA的收益。100轮是问题筛选预算，不是已证充分收敛。
共享原始分辨率JPEG文件与真实字节；固定one2many+NMS；明确ignore政策、输入小目标AP与配对漏检分析，不混称官方挑战AP或原图标准AP_s。单seed不能证明可发表效果。
设计不构成自动执行授权，本轮无模型/训练/推理/新数据压缩评价。协议或收敛不成立时记为无法判定，不强凑模块。

''' + report[section7.start():]
report = report.replace('注意：卷期为2025；DOI中的2024不等于卷期年份。非全文阅读。',
    '注意：卷期为2025；DOI中的2024不等于卷期年份。现已读用户提供的出版商PDF正文p1–8。')
report = report.replace('范围：仅出版商摘要与元数据；PDF下载失败。是相邻室内检测的机制先例，未核定量实验或实例级细节。',
    '范围：用户13页接收稿正文p2–11通读、p4–10公式/表图目视。室内相邻机制先例；定量数值仅为作者报告，接收稿协议冲突见06。')
report = report.replace('01_核心文献矩阵.txt：19条', '01_核心文献矩阵.txt：20条')
report += '''
11. 本次全文/源码核查的最终增量

FSIC是冻结检测器监督codec；PDAD是退化预测+条件恢复辅助训练，但接收稿复现协议不自洽；YOLO26已有STAL候选区域扩展，v8.4.24与v8.4.191的8–16像素边长处理不同。源码静态边界不是已测到的压缩失败，更不是新分配算法的立项证据。
本轮到调研报告与方案为止。继续价值为有条件的“小规模问题验证”，不是已确定可发表主线。完整核查及阅读证据见06；执行设计见07、08。

[R20] Cui, Z., Zhu, Y., Gu, L., Qi, G.-J., Li, X., Zhang, R., Zhang, Z., & Harada, T. (2022). Exploring resolution and degradation clues as self-supervised signal for low quality object detection. ECCV, 473–491.
官方论文：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690465.pdf
作者项目：https://github.com/cuiziteng/ECCV_AERIS
阅读范围：官方摘要/§4.3推理段/README有限选读，未通读或核定量表；本地PDF存档下载失败。
'''
(base / '00_调研报告.txt').write_text(report, encoding='utf-8')

brief = (history / '04_主线收敛与数据算法选择.txt').read_text(encoding='utf-8')
old_pdad = '[R16] PDAD（Scientific Reports，2026）摘要已提出退化感知proxy和条件引导蒸馏，并在训练后移除辅助部件；摘要明确包含compression artifacts。它来自室内检测，只能作为机制先例，不能代替UAV结果。只有摘要，不能认定其每个细节与候选A相同；但足以阻止我们把“退化感知+蒸馏+不增推理分支”当宽泛创新。'
brief = brief.replace(old_pdad, '[R16] PDAD现已通读用户接收稿正文，明确DP退化预测与条件IRB辅助恢复，训练后移除分支；式1–8不等同于已定义完整的固定教师检测KD。接收稿退化与训练协议存在冲突，见06。它仍构成机制先例，不能换成VisDrone/YOLO26就称新创新。')
brief = brief.replace('范围：仅出版商摘要/元数据；PDF未取得。不引用其具体收益数字。', '范围：用户13页接收稿正文p2–11通读、p4–10公式表图目视；报告值未经复现，协议问题见06。')
brief = '本次全文核查更新：已读FSIC/PDAD正文；具体证据与算法判断以06为准，最小验证以07/08为准。仍未跑实验，仍未确定算法创新。\n\n' + brief
(base / '04_主线收敛与数据算法选择.txt').write_text(brief, encoding='utf-8')
explanation = (history / '05_YOLO26选择与选题来源说明.txt').read_text(encoding='utf-8')
explanation = '本次源码核查更新：固定YOLO26s / v8.4.191 commit2004cba021c68a46429be460e26954ab171b2862；STAL按每条边扩候选区域，非回归GT扩框，非最终逐GT正样本保证。与v8.4.24存在规则差异。完整证据06、未来方案07/08；未运行模型。\n\n' + explanation
(base / '05_YOLO26选择与选题来源说明.txt').write_text(explanation, encoding='utf-8')

ledger = {
    'run_date':'2026-10-28', 'workflow':'academic-research-suite/deep-research/targeted lit review',
    'single_assistant':True, 'user_confirmed_question':search['confirmed_question'],
    'scope':'FSIC/PDAD full body reading, pinned YOLO26 static audit, minimal future protocol',
    'current_deliverables':[str(base/n) for n in ['06_FSIC_PDAD_YOLO26核查与主线判断.txt','07_VisDrone_YOLO26s最小验证方案.txt','08_最小验证参数.json']],
    'source_quality_limitations':['PDAD accepted-manuscript inconsistencies','No independent paper reproduction','Some source files were web excerpts only','Git tree check unavailable','AERIS limited reading and local archive failure'],
    'source_reading_receipts':[str(root/'FSIC_user_source_receipt.json'),str(root/'PDAD_user_source_receipt.json')],
    'execution':source['execution'], 'next_step':'future problem screen only if subsequently authorized; no automatic experiments',
    'research_claim':'candidate question retained conditionally; no confirmed novel algorithm or empirical detector failure'
}
save(root / 'research_run_ledger.json', ledger)
print(json.dumps({'records':len(records),'local_pdfs':len(inventory),'new_full_body_reads':2,'history_preserved':True,'model_execution':False}))
