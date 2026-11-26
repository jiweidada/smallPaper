"""Promote the reviewed aggregation clarification and synchronize revision records."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from zipfile import ZipFile
from lxml import etree
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
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十轮候选.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第九轮备份.docx'
OLD = '41cbe7bd9c42fda9ab5fac813207209d591250076e15d62f3097647d1716fc2f'
NEW = '2dea983b748b1002f372d138c591db2845a36d000c890275ea750499ce09b0e0'

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
assert not (HERE / 'tenth_revision_patch.json').exists()
patch = read(HERE / 'tenth_revision_pending_qa.json')
assert patch['source_sha256'] == OLD and patch['candidate_sha256'] == NEW
audit_path = HERE / 'aggregation_path_audit.json'
assert sha(audit_path) == patch['evidence_sha256']
audit = read(audit_path)
for source in audit['sources'].values():
    assert sha(Path(source['path'])) == source['sha256']
for setting in audit['settings'].values():
    assert sha(Path(setting['config_path'])) == setting['config_sha256']
assert sha(Path(audit['source_manifest_path'])) == audit['source_manifest_sha256']
with ZipFile(BACKUP) as a, ZipFile(CANDIDATE) as b:
    assert a.namelist() == b.namelist() and b.testzip() is None
    assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    ap = etree.fromstring(a.read('word/document.xml')).findall('.//w:p', ns)
    bp = etree.fromstring(b.read('word/document.xml')).findall('.//w:p', ns)
    assert len(ap) == len(bp)
    assert [i for i, (x, y) in enumerate(zip(ap, bp)) if etree.tostring(x) != etree.tostring(y)] == [25, 27]
    for item in patch['changes']:
        assert ''.join(bp[item['paragraph']].xpath('.//w:t/text()', namespaces=ns)) == item['after']

now = datetime.now(timezone.utc).isoformat()
pngs = {f'page-{i:02}.png': sha(HERE / 'rendered_word' / f'page-{i:02}.png') for i in range(1, 11)}
assert len(list((HERE / 'rendered_word').glob('page-*.png'))) == 10
previous_qa = read(REC / '增强时序核查_20261104/ninth_revision_visual_qa.json')
changed_pages = [int(name[5:7]) for name, digest in pngs.items() if digest != previous_qa['page_png_sha256'][name]]
assert changed_pages == [3, 4, 6, 7, 8, 9]
qa = {'status': 'VISUAL_AND_STRUCTURAL_QA_PASSED', 'revision': 10,
      'checked_local_utc': now, 'docx_sha256': NEW, 'previous_docx_sha256': OLD,
      'renderer': 'PowerShell 7.6.5, hidden read-only Word ExportAsFixedFormat, bundled Poppler pdftoppm',
      'packaged_renderer_failure': 'LibreOffice soffice.exe was not found on PATH; failure diagnosed and recorded.',
      'compatibility_retry': 'Explicit Windows PowerShell 5.1 invocation reported PSReference failure at Quit. Retried with the previously verified PowerShell 7 host; read-only export exited 0 and produced 10 pages.',
      'failed_render_process_cleanup': 'Only PID50144, created 2026-11-04T17:42:07.953334Z with exact Word Automation command line, was stopped after identity revalidation. Subsequently confirmed absent.',
      'pages': 10, 'inspected_pages': list(range(1, 11)),
      'review_method': 'Opened all ten full-page PNGs; checked the full-width projection text and Figure 3 caption, figure consistency, equation glyphs, table boundaries, references and page breaks.',
      'changed_render_pages': changed_pages, 'page_png_sha256': pngs,
      'pdf_sha256': sha(HERE / 'rendered_word/manuscript.pdf'),
      'structural_counts': patch['structural_counts'], 'only_document_xml_changed': True,
      'changed_paragraph_indices': [25, 27],
      'warnings': 'Poppler warned about Symbol and ArialUnicode; body and equation glyphs were checked visually.',
      'scope': 'Method and caption clarification only. No numerical results, formula, image, citation or experiment changes.'}
save(HERE / 'tenth_revision_visual_qa.json', qa)

entry = PAPER / '当前修订入口.txt'
entry_old = entry.read_text(encoding='utf-8-sig')
assert entry_old.startswith('【最新稿件入口：2026-11-04，第九轮')
checklist = REMOTE / '结果并入与验收清单.txt'
checklist_old = checklist.read_text(encoding='utf-8-sig')
assert checklist_old.count('当前稿件SHA256：' + OLD) == 1
resume_paths = [REMOTE / '操作恢复说明.txt', REMOTE / '第二服务器并行执行_20261104.txt']
resume_old = [p.read_text(encoding='utf-8-sig') for p in resume_paths]
old_sentence = '当前第九轮修订稿MPSMA-DEIM_conference_EN_revised.docx SHA为' + OLD + '。'
assert all(t.count(old_sentence) == 1 for t in resume_old)
previous_index_paths = [INDICES / ('主要结论证据索引_第九轮' + suffix) for suffix in ['.txt', '.json']]
previous_indices = [ref(p) for p in previous_index_paths]
assert read(previous_index_paths[1])['manuscript_sha256'] == OLD
previous_catalog_path = REC / '增强时序核查_20261104/manuscript_version_locations.json'
previous_catalog = read(previous_catalog_path)
for item in previous_catalog['versions']:
    assert sha(Path(item['retained_file']['path'])) == item['sha256']

temp = CURRENT.with_name(CURRENT.stem + '_promotion.tmp.docx')
assert not temp.exists()
shutil.copy2(CANDIDATE, temp)
os.replace(temp, CURRENT)
assert sha(CURRENT) == NEW and sha(BACKUP) == OLD
patch.update(status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA', current_manuscript_sha256=NEW,
             visual_qa_sha256=sha(HERE / 'tenth_revision_visual_qa.json'))
save(HERE / 'tenth_revision_patch.json', patch)

versions = copy.deepcopy(previous_catalog['versions'])
for item in versions:
    if item['revision'] == 9:
        item['retained_file'] = ref(BACKUP)
    item['current_canonical_has_these_bytes'] = sha(Path(item['canonical_path_recorded_at_the_time'])) == item['sha256']
versions.append({'revision': 10, 'sha256': NEW, 'canonical_path_recorded_at_the_time': str(CURRENT),
                 'retained_file': ref(CANDIDATE), 'current_canonical_has_these_bytes': True,
                 'reviewed_version_proof': ref(HERE / 'tenth_revision_patch.json')})
catalog = {'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED', 'checked_local_utc': now,
           'current_manuscript_at_this_audit': ref(CURRENT), 'versions': versions,
           'previous_catalog': ref(previous_catalog_path),
           'rule': 'Resolve version-specific manuscript SHA to the retained_file with that exact SHA. Canonical current paths in older records are historical entry points.',
           'scope': 'Original manuscript and revisions 7-10 in the current inheritance chain; file identity only.',
           'docx_files_modified_by_locator': False, 'historical_indices_modified': False}
save(HERE / 'manuscript_version_locations.json', catalog)
index = {'revision': 10, 'status': 'AGGREGATION_PROJECTION_CLARIFIED_WITH_PREVIOUS_EVIDENCE_INHERITED',
         'manuscript_path': str(CURRENT), 'manuscript_sha256': NEW, 'created_local_utc': now,
         'inherited_evidence': previous_indices, 'inherited_scope': 'Unchanged methods and historical metric/citation checks; aggregation description is superseded by the revision10 static source audit.',
         'unchanged_scope_proof': ref(HERE / 'tenth_revision_patch.json'),
         'aggregation_evidence': ref(audit_path), 'visual_qa': ref(HERE / 'tenth_revision_visual_qa.json'),
         'manuscript_version_locations': ref(HERE / 'manuscript_version_locations.json'),
         'historical_source_version_certified': False, 'table_formula_image_citation_changes': False,
         'model_executions': 0, 'evaluator_reruns': 0,
         'live_training_ledger': str(REMOTE / 'parallel_execution_ledger.json'), 'submission_ready': False,
         'remaining': 'Both complete fixed-budget training runs, four final saved-endpoint audits, signed comparison and corresponding manuscript revision.'}
save(INDICES / '主要结论证据索引_第十轮.json', index)
(INDICES / '主要结论证据索引_第十轮.txt').write_text(
    '主要结论证据索引 第十轮\n当前稿件SHA256：' + NEW + '\n'
    '本轮在第3.3节及图3图注明确完整配置的聚合路径：HG特征拼接后，一层1×1 ConvBNAct直接投影到输出宽度C，再接阶段门；本地基线是经过C/2中间宽度的两层1×1 ConvBNAct。因此完整结构变更包含投影路径，不能把差别仅归给门。\n'
    '第十轮聚合路径核查/aggregation_path_audit.json固定8份保存配置、4份所核查源码及相关行号、分支和参数。BASE/FULL源码匹配冻结清单；hgnetv2_silu.py不在该冻结清单中，只记录为N3现存源码核查，不认证其历史版本。\n'
    '除方法段落与图注两处外，其他段落、6表、7图、6式、11条引用及5组批注锚点保持。10页渲染均已打开检查。数据、原实验代码和新训练设置没有改变。\n'
    '其余证据与限制继承第九轮TXT/JSON，完整SHA链见同名JSON；旧稿路径按本轮manuscript_version_locations.json定位到各版留档，旧索引保持。\n'
    '两组已授权训练继续；仍须完整320轮、四个最终端点、带符号的完整对照及相应结果修订，尚不能投稿。\n', encoding='utf-8')

note = ('\n第十轮：完整聚合投影路径与图注（2026-11-04）\n'
        '第3.3节及图3图注明确HG块聚合直接用一层1×1 ConvBNAct到C后接阶段门，本地基线是两层1×1 ConvBNAct且中间宽度C/2。此结构差别包含投影路径，不能仅归因为新增门；图中原聚合投影图例与该定义一致。\n'
        '静态核对8份保存配置与4份现存源码；BASE/FULL源码与冻结清单一致，Silu源码未列入该清单，阅读范围单列，不认证历史训练版本。仅改方法段落与图注，表值、公式、图像及引用保持，10页逐页核验通过。\n'
        '第九轮备份SHA为' + OLD + '；当前第十轮SHA为' + NEW + '。证据、修订回执及版式记录见第十轮聚合路径核查；证据索引和历史稿件定位已同步。\n'
        '已保留导出兼容性失败日志，改用PowerShell7只读导出成功；仅清理经身份核实的失败Word实例。没有新模型或评价器执行，两组既有训练继续。目标active；完整公平对照和最终端点尚未完成，尚不能投稿。\n')
for p in [REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt']:
    with p.open('a', encoding='utf-8') as handle:
        handle.write(note)
entry_header = ('【最新稿件入口：2026-11-04，第十轮聚合路径核查】\n当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
    '第3.3节及图3图注明确：完整配置用单层1×1直接投影到输出宽度C再接门，本地基线为经过C/2的两层聚合。该差别包含投影路径。只改方法说明与图注，10页逐页核验通过。\n'
    '第九轮备份、静态证据与核验见修订记录_20261103/第十轮聚合路径核查；当前主要结论证据索引为主要结论证据索引_第十轮.txt及JSON，历史稿件按本轮版本定位记录查找。\n'
    'BASE/FULL在两台RTX3090继续训练，实时状态见远程公平对照_20261103/parallel_execution_ledger.json。仍需完整320轮、四个最终端点及相应结果修订；目标active，尚不能投稿。以下旧版本和状态均为历史。\n\n')
entry.write_text(entry_header + entry_old, encoding='utf-8')
checklist.write_text(checklist_old.replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1), encoding='utf-8')
for path, text in zip(resume_paths, resume_old):
    text = text.replace(old_sentence, '当前第十轮修订稿MPSMA-DEIM_conference_EN_revised.docx SHA为' + NEW + '。', 1)
    text += '\n2026-11-04 第十轮稿件入口：完整聚合投影路径及图3图注已明确，10页核验通过；第九轮备份保留。当前SHA为' + NEW + '。训练协议与验收端点不变，完整对照尚待完成，尚不能投稿。\n'
    path.write_text(text, encoding='utf-8')
assert sha(CURRENT) == NEW and all(sha(Path(p['path'])) == p['sha256'] for p in previous_indices)
print(json.dumps({'status': patch['status'], 'manuscript_sha256': NEW, 'pages': 10,
                  'changed_paragraphs': [25, 27], 'submission_ready': False}))
