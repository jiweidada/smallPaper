"""Validate the third revision and refresh handoff records; no model execution.

The DOCX is never modified here. Historical preparation/proposal JSON and all
evaluation artifacts remain untouched. Visual observations are human-reviewed
records from the current assistant session, not automated layout assertions.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import zipfile

from docx import Document
from lxml import etree

OUT = Path(__file__).resolve().parent
PAPER = OUT.parent
CPU = PAPER / '本地CPU核验_20261103'
BACK2 = OUT / '第二轮留档'
DOC = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
SOURCE = DOC.with_name('MPSMA-DEIM_conference_EN.docx')
ORIGINAL_BACKUP = DOC.parent / '备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
RENDER = OUT / 'rendered_cpu_20261103'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
readj = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))


def writej(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


patch = readj(OUT / 'cpu_revision_patch.json')
assert sha(DOC) == patch['after_sha256'] == '2cd8e5af9afb3379f5b7cb946c8dde9f9d0173d1da5f9af975bac70596102dd6'
assert sha(BACK2 / DOC.name) == patch['before_sha256']
assert sha(SOURCE) == sha(ORIGINAL_BACKUP) == 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
assert sha(CPU / 'comparison_report.json') == patch['cpu_evidence_sha256']

doc = Document(DOC)
prior_doc = Document(BACK2 / DOC.name)
assert len(doc.tables) == 6 and len(doc.inline_shapes) == 7
assert [rows(t) for t in doc.tables[:5]] == [rows(t) for t in prior_doc.tables]
assert rows(doc.tables[5]) == patch['new_table']
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(DOC) as z, zipfile.ZipFile(BACK2 / DOC.name) as old:
    media = {p: hashlib.sha256(z.read(p)).hexdigest() for p in z.namelist() if p.startswith('word/media/')}
    old_media = {p: hashlib.sha256(old.read(p)).hexdigest() for p in old.namelist() if p.startswith('word/media/')}
    assert media == old_media
    xml = etree.fromstring(z.read('word/document.xml'))
    old_xml = etree.fromstring(old.read('word/document.xml'))
    eqs = xml.xpath('//m:oMath', namespaces=ns)
    assert len(eqs) == 6
    assert [etree.tostring(e) for e in eqs] == [etree.tostring(e) for e in old_xml.xpath('//m:oMath', namespaces=ns)]
    comments = etree.fromstring(z.read('word/comments.xml'))
    ids = set(comments.xpath('/w:comments/w:comment/@w:id', namespaces=ns))
    assert len(ids) == 5
    for tag in ['commentRangeStart', 'commentRangeEnd', 'commentReference']:
        assert set(xml.xpath(f'//w:{tag}/@w:id', namespaces=ns)) == ids
    assert not xml.xpath('//w:p[count(w:pPr)>1]', namespaces=ns)
    assert not xml.xpath('//w:ins|//w:del|//w:pBdr', namespaces=ns)

paragraphs = [p.text for p in doc.paragraphs]
ref_start = paragraphs.index('References')
body_text = '\n'.join(paragraphs[:ref_start])
refs = [int(m[1]) for s in paragraphs[ref_start + 1:] if (m := re.match(r'^\[(\d+)\]', s))]
assert refs == list(range(1, 12))
cited = set()
for m in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', body_text):
    numbers = [int(s.strip()) for s in m[1].split(',')]
    if all(1 <= n <= 11 for n in numbers):
        cited.update(numbers)
assert cited == set(range(1, 12))
for n in range(1, 7):
    assert f'Fig. {n}.' in body_text and f'Table {n}.' in body_text

comparison = readj(CPU / 'comparison_report.json')
ledger = readj(CPU / 'execution_ledger.json')
assert comparison['status'] == 'THREE_AUTHORIZED_EVALUATIONS_COMPLETED_AND_ARTIFACTS_CHECKED'
assert ledger['status'] == 'COMPLETED' and ledger['auto_retry'] is False
assert ledger['run_order'] == [r['run_id'] for r in ledger['runs']] == ['N0', 'N4', 'S1']
assert len(ledger['runs']) == 3 and all(r['status'] == 'COMPLETED' and r['exit_code'] == 0 for r in ledger['runs'])
evidence = []
for index, run in enumerate(comparison['runs'], 1):
    run_id = run['run_id']
    folder = CPU / 'results' / run_id
    receipt = readj(folder / 'receipt.json')
    metric = readj(folder / 'metrics.json')
    loading = readj(folder / 'loading_diagnostics.json')
    for name, key in [('metrics.json', 'metric_file_sha256'), ('receipt.json', 'receipt_file_sha256'), ('loading_diagnostics.json', 'loading_file_sha256')]:
        assert sha(folder / name) == run[key]
    assert receipt['status'] == 'COMPLETED' and receipt['completed_images'] == 548 and receipt['inference_passes'] == 1
    assert receipt['device'] == 'cpu' and receipt['weight_source'] == 'ema.module'
    assert all(receipt[key] is False for key in ['training', 'deployment_conversion', 'profiling', 'timing_benchmark'])
    assert receipt['runner_sha256'] == sha(CPU / 'evaluate_cpu.py')
    assert receipt['authorization_sha256'] == sha(CPU / 'authorization.json')
    assert receipt['source_manifest_sha256'] == sha(CPU / 'source_manifest.json')
    assert receipt['image_manifest_sha256'] == sha(CPU / 'validation_image_manifest.json')
    assert receipt['configuration_sha256'] == sha(CPU / 'configs' / f'{run_id}.cpu_evaluation.json')
    assert receipt['checkpoint_sha256'] == metric['checkpoint_sha256'] == run['checkpoint_sha256']
    assert receipt['metrics_sha256'] == sha(folder / 'metrics.json')
    assert metric['predictions_sha256'] == sha(folder / 'predictions.json')
    assert loading['strict'] is True and all(not loading[key] for key in ['missing_keys', 'unexpected_keys', 'shape_mismatches'])
    assert metric['protocol']['image_ids'] == list(range(548))
    assert metric['protocol']['category_ids'] == list(range(10))
    assert metric['protocol']['max_dets'] == [1, 10, 100]
    assert run['predictions_count'] == 164400 and run['full_metric_vector_exact_match'] is False
    m = {entry['metric']: entry for entry in run['metrics']}
    wanted = [run_id, f"{m['AP']['historical_best_row']:.6f}", f"{m['AP']['cpu_re_evaluation']:.6f}",
              f"{m['AP']['delta_percentage_points']:+.6f}", f"{m['APs']['historical_best_row']:.6f}",
              f"{m['APs']['cpu_re_evaluation']:.6f}", f"{m['APs']['delta_percentage_points']:+.6f}"]
    assert rows(doc.tables[5])[index] == wanted
    evidence.append({'run_id': run_id, 'images': 548, 'inference_passes': 1, 'strict_load': True,
                     'checkpoint_sha256': run['checkpoint_sha256'], 'AP': m['AP']['cpu_re_evaluation'],
                     'AP_delta_percentage_points': m['AP']['delta_percentage_points']})

# The unchanged earlier content retains its established source checks; verify
# the local archived bytes again without re-running old finalization scripts.
archived_counts = {}
for filename in ['source_manifest.json', 'benchmark_source_manifest.json']:
    manifest = readj(OUT / filename)
    for item in manifest:
        assert sha(item['snapshot']) == item['sha256']
    archived_counts[filename] = len(manifest)

pages = sorted(RENDER.glob('page-*.png'))
assert len(pages) == 10
page_hashes = {p.name: sha(p) for p in pages}
first_hashes = readj(RENDER / 'first_render_page_hashes.json')
unchanged = [i for i, p in enumerate(pages, 1) if page_hashes[p.name] == first_hashes[p.name]]
changed = [i for i in range(1, 11) if i not in unchanged]
assert unchanged == list(range(1, 8)) and changed == [8, 9, 10]
visual = {
    'date': '2026-11-03', 'revision': 3, 'docx_sha256': sha(DOC), 'pdf_sha256': sha(RENDER / 'manuscript.pdf'),
    'renderer': 'Microsoft Word read-only ExportAsFixedFormat, then bundled Poppler pdftoppm',
    'fallback_reason': 'Packaged render_docx.py could not locate soffice.exe. Hidden read-only Word export succeeded with approved filesystem/COM access.',
    'pages': 10, 'inspected_pages': list(range(1, 11)),
    'inspection': 'All ten pages of the first third-revision render were visually opened. After Table 6 caption formatting was corrected, pages 1-7 were byte-identical to that reviewed render; final pages 8-10 were reopened and visually reviewed. Tables, equations, figures, new comparison text and all eleven references are legible. No observed clipping, overlaps, isolated table headers or missing content. Table 6 and its caption remain together.',
    'review_reused_for_byte_identical_pages': unchanged, 'final_render_pages_reopened': changed,
    'corrections_this_revision': ['Matched Table 6 caption font, size, bold style and centering to other table captions.'],
    'renderer_warnings': 'Poppler display-font warnings for Symbol and ArialUnicode were checked against the visible page images.',
    'page_png_sha256': page_hashes, 'scope': 'Visual artifact QA only; no conference template or submission-readiness certification.'
}
writej(OUT / 'visual_qa.json', visual)
patch['render_status'] = 'FINAL_WORD_EXPORT_VISUALLY_REVIEWED_10_PAGES'
patch['validation_report'] = str(OUT / 'validation_report.json')
writej(OUT / 'cpu_revision_patch.json', patch)

validation = {
    'date': '2026-11-03', 'revision': 3, 'source_and_backup_sha256': sha(SOURCE), 'revised_docx_sha256': sha(DOC),
    'second_revision_backup_sha256': sha(BACK2 / DOC.name), 'tables': 6,
    'historical_tables_1_to_5_unchanged': True, 'new_table_6_matches_cpu_comparison_report': True,
    'references_count': 11, 'all_references_cited': True, 'figures': 6, 'inline_images': 7,
    'native_equations': 6, 'media_and_equations_unchanged_from_second_revision': True,
    'comments_with_matching_start_end_reference_anchors': 5, 'duplicate_paragraph_property_nodes': 0,
    'archived_source_files_reverified': archived_counts, 'authorized_new_evaluations': evidence,
    'comparison_report_sha256': sha(CPU / 'comparison_report.json'), 'execution_ledger_sha256': sha(CPU / 'execution_ledger.json'),
    'no_training_or_tuning': True, 'no_new_timing_benchmark': True, 'submission_ready': False,
    'layout_review': 'visual_qa.json: 10 pages; final changed pages reopened.',
    'note': 'Current checkpoint/evaluation linkage is established. Exact historical reproduction, matched training comparisons and deployment accuracy/timing linkage remain unverified. These artifact checks are not an ARS integrity PASS.'
}
writej(OUT / 'validation_report.json', validation)

# Retain the previous acronym inputs and generate checks for the current prose.
for filename in ['acronym_input.txt', 'acronym_report.json', 'acronym_allowlist.txt']:
    if not (BACK2 / filename).exists():
        shutil.copy2(OUT / filename, BACK2 / filename)
abstract = paragraphs[paragraphs.index('Abstract') + 1]
start = paragraphs.index('1. Introduction')
body = []
for p in doc.paragraphs[start:ref_start]:
    body.append(('## ' if p.style.name.startswith('Heading') else '') + p.text)
(OUT / 'acronym_input.txt').write_text('# Abstract\n\n' + abstract + '\n\n# Body\n\n' + '\n\n'.join(body) + '\n', encoding='utf-8')
allow_path = OUT / 'acronym_allowlist.txt'
allow = allow_path.read_text(encoding='utf-8-sig').splitlines()
for acronym in ['AMD', 'CPU', 'SHA']:
    if acronym not in allow:
        allow.append(acronym)
allow_path.write_text('\n'.join(allow) + '\n', encoding='utf-8')

notes = (BACK2 / '修订说明与未解决问题.txt').read_text(encoding='utf-8-sig')
replacements = {
    '更新日期：2026-11-03，第二轮补充；': '更新日期：2026-11-03，第三轮（已纳入授权的本地 CPU 重评）；',
    '未运行新训练、推理、重评、profiling、时延测试或远程任务。': '本轮已按用户明确授权完成 N0/N4/S1 各一次 548 图本地 CPU 重评；未训练、调参、profiling、测速或使用远程算力。公开依赖下载另获授权。',
    '本轮修订稿SHA256：d38bcbb479f2ed815385330064508b88849abef279c68c6c989bb680963ad451': '本轮修订稿SHA256：' + sha(DOC),
    'revise_manuscript.py 是本轮从原稿重建的脚本；后续有人工编辑时不要直接重跑覆盖，应先备份并合并更改。': 'revise_manuscript.py 与 finalize_audit.py 属于前两轮，不能直接重跑覆盖当前稿件或报告。第三轮 incorporate_cpu_results.py 从第二轮留档生成稿件，finalize_cpu_revision.py 只核验和更新记录；后续人工编辑仍须先备份并合并。',
    '正文4.1和指标定义已据此更新。未重算AP。': '正文4.1和指标定义已据此更新。第二轮未重算AP；第三轮独立重评见新增第4.5节和Table 6。',
    '需要原保存回执/控制台记录，或另行授权后重评指定SHA权重。': '本轮已授权并重评N0/N4/S1指定SHA权重，建立当前评价链；整体AP接近但完整12指标不同，仍不能认证历史最佳行完全对应。N2尚未重评，原保存回执/控制台及版本证据仍缺失。',
    '六、最小补实验方案（方案已具体化，尚未授权和运行）': '六、补实验方案及授权边界（C项三份权重核验已执行；A/B/D仍未授权和运行）',
    'C. 解决权重与指标链：优先只读找保存回执；不足时，申请对指定SHA checkpoint进行固定评价器验证，并保存加载键检查、预测文件及评价结果。这仍属于新重评，需要明确授权。': 'C. 权重与指标链：用户已明确授权三份留存权重各重评一次；N0/N4/S1均完成，加载键检查、预测、完整评价结果及SHA回执均已保存。当前评价链已补证，不能据此补全历史不可变版本。N1/N2/N3及任何追加重评仍不在本次范围。',
    '验证记录\n': '验证记录\n',
    'validation_report.json：原稿/备份相同、表格数字与来源、11条引用对应、6幅编号图（7个内嵌图像）6式、5个问题批注、图6源图SHA及22份源码/运行快照核验（具体计数以JSON为准）。': 'validation_report.json：原稿/备份相同、旧5表及图片/公式未变、新Table 6对应本次指标、11条引用、6幅编号图（7个内嵌图像）6式、5条批注完整锚点，以及三份评测回执和产物哈希。第二轮图6来源及数据核对保留在第二轮留档和dataset_eval_audit。',
    'visual_qa.json：最终DOCX用Word导出并经Poppler逐页检查；': 'visual_qa.json：本轮最终DOCX共10页，用Word导出并经Poppler逐页检查；',
    '本轮验收：主要数值可追到日志；': '本轮验收：历史数值可追到日志，新增N0/N4/S1当前指标可追到指定SHA权重、配置、代码、数据和预测；'
}
for old, new in replacements.items():
    assert old in notes, old
    notes = notes.replace(old, new)
notes += '\n八、第三轮新增证据与结论边界\n'
notes += '用户明确回复“授权本地 CPU 重评”及“允许下载公开依赖”，原文和范围见本地CPU核验_20261103/authorization.json。独立依赖位于该目录.runtime/site-packages；代码、图片、权重未上传。\n'
notes += '三组均CPU/FP32、严格EMA及后处理加载、完整548图、每图300预测，汇总maxDets=[1,10,100]；N0验证batch16，N4/S1为4。实际每组完整评测仅一次，退出码均0；未部署转换。\n'
for r in comparison['runs']:
    m = {x['metric']: x for x in r['metrics']}
    notes += f"{r['run_id']}：历史AP {m['AP']['historical_best_row']:.9f} → 本次CPU AP {m['AP']['cpu_re_evaluation']:.9f}，差 {m['AP']['delta_percentage_points']:+.9f} 个百分点。\n"
notes += '最大AP差约0.00147个百分点；完整12指标最大差约0.08462个百分点（N0大目标召回率）。不按事后容差宣布“完全复现”；差异原因未隔离，不擅自归因CPU/GPU或软件版本。\n'
notes += '新增第4.5节和Table 6独立说明此次协议及结果；更新证据限制、结论和两条批注。原历史配置表保持原数值与条件，不拼接CPU新值形成消融排名。\n'
notes += '保留N4/S1初次兼容性检查因GBK输出特殊字符失败的回执；UTF-8修正后预检成功。预检不含前向，不能误计为多次完整重评。未使用的calflops演示导入采用遇调用即抛错的占位模块，选定神经网络源码未改；详见本地重评结果与限制.txt。\n'
notes += '当前内容修订与已授权核验完成；公平训练对照仍是投稿前的主要缺口。下一步A项最少两组同条件新训练已具体化，但未获授权；不得将本次本地重评授权扩展为训练或新实验。\n'
(OUT / '修订说明与未解决问题.txt').write_text(notes, encoding='utf-8')

entry = PAPER / '当前修订入口.txt'
entry_backup = OUT / '入口更新前_含历史授权状态.txt'
if not entry_backup.exists():
    shutil.copy2(entry, entry_backup)
entry.write_text(f'''更新日期：2026-11-03，第三轮内容修订及已授权 CPU 重评完成。
当前修订稿：{DOC}
修订说明及未解决问题：{OUT / '修订说明与未解决问题.txt'}
本地重评结果：{CPU / '本地重评结果与限制.txt'}
完整数值与哈希：{CPU / 'comparison_report.json'}
授权及执行回执：{CPU / 'authorization.json'}；{CPU / 'execution_ledger.json'}
版式与产物核验：{OUT / 'visual_qa.json'}；{OUT / 'validation_report.json'}

N0/N4/S1各完成一次548图CPU重评，严格EMA加载，无训练或调参。当前指标已绑定指定权重及输入/源码哈希；历史日志表与本次重评表分开保留。
稿件已更新第4.5节、Table 6、限制、结论及相关批注；最终10页已核查。原稿及前两轮版本保留。
尚不能投稿：原训练batch和聚合结构混杂仍在，历史不可变版本及部署测量链未闭合，完整12指标也不完全一致。会议尚未选择。
完整目标尚未完成；本次授权任务已执行完毕。后续训练或扩大重评范围须另获明确授权。
本地CPU准备/提案JSON中的NOT_AUTHORIZED/NOT_RUN为初次冻结时的历史状态，由上述授权和完成回执衔接，不是当前状态。旧blocked记录同样只描述当时情况。
''', encoding='utf-8')

banner = ('【2026-11-03 当前更新：已获授权并完成】\n'
          'N0、N4、S1各一次548图本地CPU重评已完成，相关结果已写入第三轮英文修订稿。\n'
          '用户已明确授权本地重评和公开依赖下载；未训练、调参或使用远程算力。\n'
          '当前结果及边界见：01_会议论文/本地CPU核验_20261103/本地重评结果与限制.txt。\n'
          '下方保留的是授权前的历史准备/提案，不代表当前状态，也不授权追加实验。\n\n')
for path in [CPU / '准备说明.txt', OUT / '本地CPU可行性与准备状态.txt', OUT / '下一步最小核验_待授权.txt']:
    text = path.read_text(encoding='utf-8-sig')
    if not text.startswith('【2026-11-03 当前更新：已获授权并完成】'):
        path.write_text(banner + text, encoding='utf-8')

history_path = OUT / '目标验收与阻塞记录.txt'
history = history_path.read_text(encoding='utf-8-sig')
marker = '第三轮已授权本地重评完成（2026-11-03）'
if marker not in history:
    history += '\n\n' + marker + '\n'
    history += '用户明确授权本地CPU重评和公开依赖下载，先前缺少这两项授权的阻塞已经解除。目标已由用户恢复为active，本轮未调用update_goal将其完成或重新阻塞。\n'
    history += '已完成三组完整548图评测，执行回执均exit0；第三轮修订稿和10页版式核查完成，详细路径见当前修订入口.txt。原始历史日志与当前CPU指标分表。\n'
    history += '本次当前权重与指标链已建立；完整12指标差异、历史代码/环境版本、公平训练比较和部署准确率/时延关联仍未闭合，整体投稿验收未通过。原始同一授权阻塞不能沿用；今后阻塞判断须基于恢复后的新状态和剩余问题。\n'
    history_path.write_text(history, encoding='utf-8')

print(json.dumps({'status': 'THIRD_REVISION_ARTIFACT_CHECKS_PASSED', 'docx_sha256': sha(DOC),
                  'tables': 6, 'pages': 10, 'authorized_evaluations_completed': 3,
                  'acronym_check': 'INPUT_REFRESHED_RUN_CHECKER_NEXT', 'submission_ready': False}, ensure_ascii=False))
