"""Promote the exact reviewed augmentation-description edit; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from zipfile import ZipFile
import json
import os
import shutil
from lxml import etree

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第九轮候选.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第八轮备份.docx'
OLD = '37026584852e657c15bce14d522d261762acf2695c7bd681948b56ff0ccc66ac'
NEW = '41cbe7bd9c42fda9ab5fac813207209d591250076e15d62f3097647d1716fc2f'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def ref(path, scope=None):
    result = {'path': str(path), 'sha256': sha(path)}
    if scope:
        result['scope'] = scope
    return result

assert sha(CURRENT) == OLD and sha(BACKUP) == OLD and sha(CANDIDATE) == NEW
assert sha(PAPER / '稿件/MPSMA-DEIM_conference_EN.docx') == ORIGINAL
assert not (HERE / 'ninth_revision_patch.json').exists(), 'Already promoted'
patch = read(HERE / 'ninth_revision_pending_qa.json')
assert patch['source_sha256'] == OLD and patch['candidate_sha256'] == NEW
evidence = HERE / 'augmentation_schedule_audit.json'
assert sha(evidence) == patch['evidence_sha256']
audit = read(evidence)
for item in list(audit['source_files']) + [
    {'path': v['config_path'], 'sha256': v['config_sha256']}
    for v in audit['settings'].values()
]:
    assert sha(Path(item['path'])) == item['sha256']
with ZipFile(BACKUP) as old, ZipFile(CANDIDATE) as new:
    assert old.namelist() == new.namelist()
    assert [n for n in old.namelist() if old.read(n) != new.read(n)] == ['word/document.xml']
    oldxml = etree.fromstring(old.read('word/document.xml'))
    newxml = etree.fromstring(new.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    op = oldxml.findall('.//w:p', ns)
    np = newxml.findall('.//w:p', ns)
    assert len(op) == len(np)
    assert [i for i, (a, b) in enumerate(zip(op, np))
            if etree.tostring(a) != etree.tostring(b)] == [55]
    assert ''.join(np[55].itertext()) == patch['after']

previous_qa = read(REC / '引用补核_20261104/eighth_revision_visual_qa.json')
pngs = {f'page-{i:02}.png': sha(HERE / 'rendered_word' / f'page-{i:02}.png')
        for i in range(1, 11)}
assert sorted((HERE / 'rendered_word').glob('page-*.png')) == [
    HERE / 'rendered_word' / f'page-{i:02}.png' for i in range(1, 11)]
changed = [int(name[5:7]) for name, digest in pngs.items()
           if digest != previous_qa['page_png_sha256'][name]]
assert changed == [6, 7, 8, 9]
qa = {
    'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision': 9,
    'checked_local_utc': datetime.now(timezone.utc).isoformat(),
    'docx_sha256': NEW, 'previous_docx_sha256': OLD,
    'renderer': 'Word read-only ExportAsFixedFormat and bundled Poppler pdftoppm',
    'packaged_renderer_failure': 'LibreOffice soffice.exe was not found on PATH',
    'word_com_execution': 'Read-only export succeeded outside sandbox after in-sandbox COM E_ACCESSDENIED.',
    'pdf_sha256': sha(HERE / 'rendered_word/manuscript.pdf'),
    'pages': 10, 'inspected_pages': list(range(1, 11)),
    'review_method': 'Opened every full-page PNG after the edit. Checked augmentation ranges, metrics paragraph, table/caption boundaries, equations, diagrams, glyphs and references.',
    'changed_render_pages': changed,
    'identical_to_previously_reviewed_revision8_pages': [1, 2, 3, 4, 5, 10],
    'page_png_sha256': pngs,
    'structural_counts': patch['structural_counts'],
    'only_document_xml_changed': True, 'only_paragraph_index_changed': 55,
    'warnings': 'Poppler warned about Symbol and ArialUnicode; body and equation glyphs checked visually.',
    'scope': 'Description clarification only. No new training or evaluation; final paired comparison remains pending.'
}
save(HERE / 'ninth_revision_visual_qa.json', qa)

entry = PAPER / '当前修订入口.txt'
entry_old = entry.read_text(encoding='utf-8-sig')
assert entry_old.startswith('【最新稿件入口：2026-11-04，第八轮')
checklist = REMOTE / '结果并入与验收清单.txt'
checklist_old = checklist.read_text(encoding='utf-8-sig')
assert checklist_old.count('当前稿件SHA256：' + OLD) == 1
resume_paths = [REMOTE / '操作恢复说明.txt', REMOTE / '第二服务器并行执行_20261104.txt']
old_sentence = '当前第八轮修订稿MPSMA-DEIM_conference_EN_revised.docx SHA为' + OLD + '。'
resume_texts = [p.read_text(encoding='utf-8-sig') for p in resume_paths]
assert all(t.count(old_sentence) == 1 for t in resume_texts)

tmp = CURRENT.with_name(CURRENT.stem + '_promotion.tmp.docx')
assert not tmp.exists()
shutil.copy2(CANDIDATE, tmp)
os.replace(tmp, CURRENT)
assert sha(CURRENT) == NEW and sha(BACKUP) == OLD
patch.update(status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA',
             current_manuscript_sha256=NEW,
             visual_qa_sha256=sha(HERE / 'ninth_revision_visual_qa.json'))
save(HERE / 'ninth_revision_patch.json', patch)

index_dir = REC / '主要结论证据索引'
index = {
    'revision': 9, 'status': 'AUGMENTATION_DESCRIPTION_TRACED_WITH_PRIOR_EVIDENCE_INHERITED',
    'manuscript_path': str(CURRENT), 'manuscript_sha256': NEW,
    'created_local_utc': datetime.now(timezone.utc).isoformat(),
    'inherited_evidence': [ref(index_dir / ('主要结论证据索引_第八轮' + suffix),
        'Unchanged methods, historical metrics, citation checks and limits; augmentation paragraph superseded by revision9 static audit.')
        for suffix in ['.txt', '.json']],
    'unchanged_scope_proof': ref(HERE / 'ninth_revision_patch.json'),
    'augmentation_evidence': ref(evidence),
    'visual_qa': ref(HERE / 'ninth_revision_visual_qa.json'),
    'table_formula_image_citation_changes': False,
    'historical_source_version_certified': False,
    'model_executions': 0, 'evaluator_reruns': 0,
    'live_training_ledger': str(REMOTE / 'parallel_execution_ledger.json'),
    'submission_ready': False,
    'remaining': 'Both complete 320-loop training runs, four final saved-endpoint audits, signed comparison and corresponding manuscript revision.'
}
save(index_dir / '主要结论证据索引_第九轮.json', index)
(index_dir / '主要结论证据索引_第九轮.txt').write_text(
    '主要结论证据索引 第九轮\n当前稿件SHA256：' + NEW + '\n'
    '本轮仅细化第4.1节增强时序及固定输入说明。证据：增强时序核查_20261104/augmentation_schedule_audit.json，记录六份历史配置、两份冻结配置与所核查源码的SHA。\n'
    '以0为起点：Mosaic选择器及批次MixUp各0.5，仅在4—155轮启用；颜色扰动、zoom-out与IoU cropping可在4—295轮启用，选Mosaic时跳过后两者。296轮起停用这些强增强，保留翻转和640×640缩放；base_size_repeat=null，因此关闭批次多尺度缩放。\n'
    '说明限定为保存设置与所核查源码的语义；冻结源码对应当前新对照，不能倒推认证历史运行时源码字节。\n'
    '表2/3仍取各运行最高AP的完整同行；表值、公式、图、参考文献均未改。其来源与限制继承第八轮TXT/JSON（见本轮JSON所固定SHA）；第八轮又固定第七轮方法/数值核验。\n'
    '10页渲染均已打开检查，本轮JSON固定DOCX/PDF/各页PNG及修订回执SHA。原稿与第八轮备份保持。\n'
    'BASE/FULL仍在两台服务器执行；实时状态以parallel_execution_ledger.json引用快照为准。完整训练、四个最终端点和对应结果修订未完成，尚不能投稿。\n', encoding='utf-8')

note = ('\n第九轮：明确增强阶段与输入尺寸（2026-11-04）\n'
    '第4.1节按保存设置和所核查实现明确0起算轮次：Mosaic选择器与批次MixUp各0.5，仅4—155启用；其他指定强增强可在4—295启用，Mosaic分支跳过zoom-out和IoU crop；296起停用这些强增强。明确base_size_repeat=null、批次多尺度缩放关闭，并保留翻转/640×640缩放与EMA说明。表2/3选择最高AP整行的规则不变。\n'
    '证据固定六份历史配置、两份冻结配置及当前源码SHA，不认证历史运行源码。仅document.xml的一段文字改变，6表、7图、6式、5组批注锚点及文献编号保持。10页逐页检查通过；第八轮备份SHA为' + OLD + '，当前SHA为' + NEW + '。\n'
    '证据与核验见增强时序核查_20261104及主要结论证据索引_第九轮。未执行新增模型、评价或训练；两组既有训练继续，完整对照与四个最终端点尚未完成。目标active，尚不能投稿。\n')
for path in [REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt']:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(note)
entry_header = ('【最新稿件入口：2026-11-04，第九轮增强时序核查】\n'
    '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
    '第4.1节写清增强启停轮次、Mosaic分支条件和关闭批次多尺度缩放的设置；表2/3仍取最高AP整行。只改一段说明，图表公式与引用保留，10页逐页核查通过。\n'
    '第八轮备份和证据：修订记录_20261103/增强时序核查_20261104；主要结论证据索引：主要结论证据索引_第九轮.txt及JSON。\n'
    'BASE/FULL在两台RTX3090继续训练，实时状态见远程公平对照_20261103/parallel_execution_ledger.json。仍待完整320轮和四个最终端点及相应结果修订；目标active，尚不能投稿。\n'
    '以下旧稿版本及运行状态均为历史，由本段覆盖。\n\n')
entry.write_text(entry_header + entry_old, encoding='utf-8')
checklist.write_text(checklist_old.replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1), encoding='utf-8')
for path, text in zip(resume_paths, resume_texts):
    text = text.replace(old_sentence, '当前第九轮修订稿MPSMA-DEIM_conference_EN_revised.docx SHA为' + NEW + '。', 1)
    text += ('\n2026-11-04 第九轮稿件入口：增强时序、Mosaic条件及固定输入说明已核对，10页逐页核验通过。当前SHA为' + NEW + '；第八轮备份保留。详见修订记录_20261103/增强时序核查_20261104。训练协议不变，完整对照仍待完成，尚不能投稿。\n')
    path.write_text(text, encoding='utf-8')
assert sha(CURRENT) == NEW
print(json.dumps({'status': patch['status'], 'manuscript_sha256': NEW,
    'pages': 10, 'changed_paragraphs': 1, 'submission_ready': False}))
