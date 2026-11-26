"""Promote the visually reviewed evaluation-limit clarification and its evidence."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader
import copy
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDICES = REC / '主要结论证据索引'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十一轮候选.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十轮备份.docx'
OLD = '2dea983b748b1002f372d138c591db2845a36d000c890275ea750499ce09b0e0'
NEW = '06c51c534c640fe35348bd88ed6c9a2a45c540b35521e378f008fe5a93126809'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def ref(path):
    return {'path': str(path), 'sha256': sha(path)}

assert sha(CURRENT) == OLD and sha(BACKUP) == OLD and sha(CANDIDATE) == NEW
assert sha(PAPER / '稿件/MPSMA-DEIM_conference_EN.docx') == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
new_paths = [HERE / 'eleventh_revision_patch.json', HERE / 'eleventh_revision_visual_qa.json',
             HERE / 'manuscript_version_locations.json',
             INDICES / '主要结论证据索引_第十一轮.json', INDICES / '主要结论证据索引_第十一轮.txt']
assert not any(p.exists() for p in new_paths)
patch = read(HERE / 'eleventh_revision_pending_qa.json')
assert patch['source_sha256'] == OLD and patch['candidate_sha256'] == NEW
audit_path = HERE / 'evaluation_limit_audit.json'
assert sha(audit_path) == patch['evidence_sha256']
audit = read(audit_path)
evidence = (audit['archived_sources'] + audit['remote_source_capture_files']
            + list(audit['settings'].values())
            + [audit['cpu_source'], audit['saved_historical_evaluator_parameters']])
for source in evidence:
    assert sha(Path(source['path'])) == source['sha256']
assert audit['remote_package_version'] == '1.8.0'
assert audit['cpu_reevaluation_package_version'] == '1.6.5'
assert not audit['historical_runtime_version_certified']
assert audit['model_executions'] == audit['evaluator_executions'] == 0
assert audit['training_settings_changed'] is False
with ZipFile(BACKUP) as a, ZipFile(CANDIDATE) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
          'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
    ar = etree.fromstring(a.read('word/document.xml'))
    br = etree.fromstring(b.read('word/document.xml'))
    ap, bp = ar.findall('.//w:p', ns), br.findall('.//w:p', ns)
    assert len(ap) == len(bp)
    assert [i for i, (x, y) in enumerate(zip(ap, bp)) if etree.tostring(x) != etree.tostring(y)] == [50, 55]
    for item in patch['changes']:
        assert ''.join(bp[item['paragraph']].xpath('.//w:t/text()', namespaces=ns)) == item['after']
    counts = {'tables': len(br.findall('.//w:tbl', ns)),
              'equations': len(br.findall('.//m:oMath', ns)),
              'inline_images': len(br.findall('.//wp:inline', ns))}
    for name in ['commentRangeStart', 'commentRangeEnd', 'commentReference']:
        counts[name] = len(br.findall('.//w:' + name, ns))
    assert counts == patch['structural_counts']

entry = PAPER / '当前修订入口.txt'
entry_old = entry.read_text(encoding='utf-8-sig')
assert entry_old.startswith('【最新稿件入口：2026-11-04，第十轮')
checklist = REMOTE / '结果并入与验收清单.txt'
checklist_old = checklist.read_text(encoding='utf-8-sig')
assert checklist_old.count('当前稿件SHA256：' + OLD) == 1
resume_paths = [REMOTE / '操作恢复说明.txt', REMOTE / '第二服务器并行执行_20261104.txt']
resume_old = [p.read_text(encoding='utf-8-sig') for p in resume_paths]
old_sentence = '当前第十轮修订稿MPSMA-DEIM_conference_EN_revised.docx SHA为' + OLD + '。'
assert all(t.count(old_sentence) == 1 for t in resume_old)
previous_index_paths = [INDICES / ('主要结论证据索引_第十轮' + suffix) for suffix in ['.txt', '.json']]
previous_indices = [ref(p) for p in previous_index_paths]
assert read(previous_index_paths[1])['manuscript_sha256'] == OLD
previous_catalog_path = REC / '第十轮聚合路径核查/manuscript_version_locations.json'
previous_catalog = read(previous_catalog_path)
for item in previous_catalog['versions']:
    assert sha(Path(item['retained_file']['path'])) == item['sha256']

now = datetime.now(timezone.utc).isoformat()
pngs = {f'page-{i:02}.png': sha(HERE / 'rendered_word' / f'page-{i:02}.png') for i in range(1, 11)}
assert len(list((HERE / 'rendered_word').glob('page-*.png'))) == 10
assert len(PdfReader(HERE / 'rendered_word/manuscript.pdf').pages) == 10
previous_qa = read(REC / '第十轮聚合路径核查/tenth_revision_visual_qa.json')
changed_pages = [int(name[5:7]) for name, digest in pngs.items() if digest != previous_qa['page_png_sha256'][name]]
qa = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision': 11,
      'checked_local_utc': now, 'docx_sha256': NEW, 'previous_docx_sha256': OLD,
      'renderer': 'PowerShell 7.6.5, hidden read-only Word ExportAsFixedFormat, bundled Poppler pdftoppm',
      'packaged_renderer_failure': 'LibreOffice soffice.exe was not found on PATH; diagnosed before the successful Word fallback.',
      'word_export_exit_code': 0, 'render_log': ref(HERE / 'render_eleventh_revision.log'),
      'pages': 10, 'inspected_pages': list(range(1, 11)),
      'review_method': 'Opened all ten full-page PNGs. Checked the top-300 and class-aware maxDets statements on pages 5-6; also reviewed figures, equations, tables, references, glyphs and page breaks. Pages 3-4 were explicitly reopened to finish the review after context compaction.',
      'changed_render_pages': changed_pages, 'page_png_sha256': pngs,
      'pdf_sha256': sha(HERE / 'rendered_word/manuscript.pdf'),
      'structural_counts': counts, 'only_document_xml_changed': True,
      'changed_paragraph_indices': [50, 55],
      'warnings': 'Poppler warned about Symbol and ArialUnicode; body and equation glyphs were checked visually.',
      'scope': 'Evaluator-definition clarification only; metric values, tables, formulas, figures, citations and experiment settings remain unchanged.'}

temp = CURRENT.with_name(CURRENT.stem + '_promotion.tmp.docx')
assert not temp.exists()
save(HERE / 'eleventh_revision_visual_qa.json', qa)
shutil.copy2(CANDIDATE, temp)
os.replace(temp, CURRENT)
assert sha(CURRENT) == NEW and sha(BACKUP) == OLD
patch.update(status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA', current_manuscript_sha256=NEW,
             visual_qa_sha256=sha(HERE / 'eleventh_revision_visual_qa.json'))
save(HERE / 'eleventh_revision_patch.json', patch)

versions = copy.deepcopy(previous_catalog['versions'])
for item in versions:
    if item['revision'] == 10:
        item['retained_file'] = ref(BACKUP)
    item['current_canonical_has_these_bytes'] = sha(Path(item['canonical_path_recorded_at_the_time'])) == item['sha256']
versions.append({'revision': 11, 'sha256': NEW, 'canonical_path_recorded_at_the_time': str(CURRENT),
                 'retained_file': ref(CANDIDATE), 'current_canonical_has_these_bytes': True,
                 'reviewed_version_proof': ref(HERE / 'eleventh_revision_patch.json')})
catalog = {'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED', 'checked_local_utc': now,
           'current_manuscript_at_this_audit': ref(CURRENT), 'versions': versions,
           'previous_catalog': ref(previous_catalog_path),
           'rule': 'Resolve version-specific manuscript SHA to the retained_file with that exact SHA. Older canonical paths are historical entry points.',
           'scope': 'Original manuscript and revisions 7-11 in the current inheritance chain; file identity only.',
           'docx_files_modified_by_locator': False, 'historical_indices_modified': False}
save(HERE / 'manuscript_version_locations.json', catalog)
index = {'revision': 11, 'status': 'EVALUATION_LIMITS_CLARIFIED_WITH_PREVIOUS_EVIDENCE_INHERITED',
         'manuscript_path': str(CURRENT), 'manuscript_sha256': NEW, 'created_local_utc': now,
         'inherited_evidence': previous_indices,
         'inherited_scope': 'Unchanged methods, metrics and citation checks; the evaluator-limit definitions in paragraphs 50 and 55 are superseded by this static audit.',
         'unchanged_scope_proof': ref(HERE / 'eleventh_revision_patch.json'),
         'evaluation_limit_evidence': ref(audit_path), 'visual_qa': ref(HERE / 'eleventh_revision_visual_qa.json'),
         'manuscript_version_locations': ref(HERE / 'manuscript_version_locations.json'),
         'server_evaluator_version': '1.8.0', 'cpu_reevaluation_evaluator_version': '1.6.5',
         'historical_source_version_certified': False, 'table_formula_image_citation_changes': False,
         'configuration_vectors_not_citations': ['[1, 10, 100]', '[4, 156, 296]', '[0, 1024]', '[6, 6]', '[3, 6, 3]'],
         'model_executions': 0, 'evaluator_reruns': 0,
         'live_training_ledger': str(REMOTE / 'parallel_execution_ledger.json'), 'submission_ready': False,
         'remaining': 'Both complete fixed-budget training runs, four final saved-endpoint audits, signed comparison and corresponding manuscript revision.'}
save(INDICES / '主要结论证据索引_第十一轮.json', index)
(INDICES / '主要结论证据索引_第十一轮.txt').write_text(
    '主要结论证据索引 第十一轮\n当前稿件SHA256：' + NEW + '\n'
    '第4.1及4.2节明确：后处理按每图、跨类别输出300个预测；useCats=1的评价器按图像与类别分组应用maxDets=[1, 10, 100]，AR@100采用其中100档。修正原“每图最多100个检测”的简化说法。\n'
    '静态证据见第十一轮评价上限核查/evaluation_limit_audit.json：两台服务器的四份Python源文件逐字节相同并已归档，八份保存配置均为top300。阅读范围为Python分组、截断和调用参数，未单独核查或执行原生C++实现。\n'
    '当前服务器评价器版本1.8.0与先前本地CPU重评1.6.5分别记录；不认证历史训练时的运行版本，不将版本差异解释为旧指标差异的原因。\n'
    '仅改两段评价定义，其他段落、6表、7图、6式、11条引用及5组批注锚点保持；10页逐页核验通过。[1, 10, 100]为配置向量，不是参考文献号。\n'
    '其他证据继承第十轮TXT/JSON，SHA链见同名JSON；旧稿按本轮manuscript_version_locations.json定位。未运行新模型或评价器，也未修改训练设置。\n'
    '两组已授权训练继续；完整320轮、四个最终端点、带符号的公平主对照和相应正文修订尚待完成，目标active，尚不能投稿。\n', encoding='utf-8')

note = ('\n第十一轮：评价器检测数上限（2026-11-04，本地日期）\n'
        '第4.1及4.2节区分每图跨类别top300后处理与useCats=1下按图像/类别应用的maxDets=[1, 10, 100]，修正原AR@100“每图最多100个检测”的简化表述。\n'
        '归档并核对两台服务器一致的四份Python源码及八份保存配置；未单独审查或执行原生C++后端。服务器评价器1.8.0、本地既有CPU重评1.6.5和未认证的历史运行版本分别记载，未归因旧指标差异。\n'
        '仅修改两段评价定义，其他段落、表值、公式、图像、引用与批注锚点保持。10页全部打开核验；打包渲染器因缺少soffice不可用后，PowerShell7/隐藏只读Word导出成功。\n'
        '第十轮备份SHA为' + OLD + '；当前第十一轮SHA为' + NEW + '。证据、修订回执、版式记录和版本位置均见第十一轮评价上限核查；当前证据索引已同步。\n'
        '没有新增模型或评价器执行，没有改变训练设置。两组已授权训练继续；完整主对照与四个最终端点仍未完成，目标active，尚不能投稿。\n')
for p in [REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt']:
    with p.open('a', encoding='utf-8') as handle:
        handle.write(note)
entry_header = ('【最新稿件入口：2026-11-04，第十一轮评价上限核查】\n当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
    '已区分每图跨类别300个后处理预测与按图像、类别应用的maxDets=[1, 10, 100]，修正AR@100的检测数上限说明。仅改评价定义两段，10页逐页核验通过。\n'
    '第十轮备份、静态证据及核验见修订记录_20261103/第十一轮评价上限核查；当前证据索引为主要结论证据索引_第十一轮.txt及JSON；旧稿按本轮版本位置记录查找。\n'
    '服务器评价器1.8.0、本地既有CPU重评1.6.5及未认证的历史版本分别记录；未新增模型或评价器执行。\n'
    'BASE/FULL仍在两台RTX3090训练，实时状态见远程公平对照_20261103/parallel_execution_ledger.json。完整320轮、四个最终端点及相应结果修订尚待完成；目标active，尚不能投稿。以下旧版本和状态均为历史。\n\n')
entry.write_text(entry_header + entry_old, encoding='utf-8')
checklist.write_text(checklist_old.replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1), encoding='utf-8')
for path, text in zip(resume_paths, resume_old):
    text = text.replace(old_sentence, '当前第十一轮修订稿MPSMA-DEIM_conference_EN_revised.docx SHA为' + NEW + '。', 1)
    text += '\n2026-11-04 第十一轮稿件入口：top300后处理与class-aware maxDets上限已区分，10页核验通过；第十轮备份保留。当前SHA为' + NEW + '。服务器评价器1.8.0与既有CPU重评1.6.5分别记录；训练协议、预算和验收端点不变，完整对照尚未完成，尚不能投稿。\n'
    path.write_text(text, encoding='utf-8')
assert sha(CURRENT) == NEW and all(sha(Path(p['path'])) == p['sha256'] for p in previous_indices)
print(json.dumps({'status': patch['status'], 'manuscript_sha256': NEW, 'pages': 10,
                  'changed_paragraphs': [50, 55], 'changed_render_pages': changed_pages,
                  'submission_ready': False}))
