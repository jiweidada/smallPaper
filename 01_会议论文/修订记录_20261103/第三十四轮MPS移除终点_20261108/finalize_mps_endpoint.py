"""Verify and promote revision 34 using existing, audited experiment evidence."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent
PAPER = RECORDS.parent
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev34_v4.docx'
OLD_SHA = 'b3aae17cf12667f34797179f399dcfd608edfab724633f023921c9703b0d4cee'
NEW_SHA = '2653ab4164cc454b3fad4d21c4c2e02e80febe00e80e4fc3e13759a7173114ac'
ORIGINAL_SHA = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
NOW = datetime.now(timezone.utc).isoformat()
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
digest = lambda p: sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
value = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))
relative = lambda p: p.relative_to(PAPER).as_posix()

def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def atomic_json(path, data):
    temporary = path.with_suffix('.rev34.tmp')
    write_json(temporary, data)
    os.replace(temporary, path)

assert digest(MASTER) == digest(BACKUP) == OLD_SHA
assert digest(CANDIDATE) == NEW_SHA
for path in [PAPER / '稿件/MPSMA-DEIM_conference_EN.docx',
             PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx']:
    assert digest(path) == ORIGINAL_SHA
assert read(HERE / 'orphan_reference_resolution.json')['candidate_sha256'] == NEW_SHA
assert read(HERE / 'render_metadata_v4.json')['page_count'] == 8
page_paths = sorted((HERE / 'rendered_word_v4').glob('page-*.png'))
assert len(page_paths) == 8 and all(p.stat().st_size > 100000 for p in page_paths)
assert 'soffice.exe was not found' in (HERE / 'packaged_render_v4.log').read_text(encoding='utf-8-sig')

comparison_path = BUNDLE / 'NO_MPS_FULL_endpoint_comparison_20261108.json'
comparison = read(comparison_path)
primary = comparison['comparisons']['primary_fixed_round320']
lineage_path = BUNDLE / 'NO_MPS/lineage_through_round320.json'
lineage = read(lineage_path)
assert lineage['status'] == 'VERIFIED_ABLATION_LINEAGE_AND_SAVED_ENDPOINTS'
assert lineage['completed_rounds'] == 320
assert lineage['learning_rates']['recorded_attempts'] == 517440
assert lineage['learning_rates']['maximum_absolute_difference'] == 0
assert lineage['restore_event_count'] == 17
assert comparison['NO_MPS_best_and_primary_coincide'] is True
assert primary['fractions']['NO_MPS'] == lineage['endpoints']['best']['metrics']
assert primary['endpoint_checkpoint_sha256']['NO_MPS'] == lineage['endpoints']['best']['checkpoint_sha256']

with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
    assert after.testzip() is None
    assert set(before.namelist()) == set(after.namelist())
    changed_parts = [n for n in before.namelist() if before.read(n) != after.read(n)]
    assert set(changed_parts) == {'word/document.xml', 'word/comments.xml'}
    old = etree.fromstring(before.read('word/document.xml'))
    new = etree.fromstring(after.read('word/document.xml'))
    comments = etree.fromstring(after.read('word/comments.xml'))
    assert all(b'\xef\xbf\xbd' not in after.read(n) for n in changed_parts)
assert [etree.tostring(n) for n in old.findall('.//m:oMath', NS)] == [etree.tostring(n) for n in new.findall('.//m:oMath', NS)]
assert len(new.findall('.//m:oMath', NS)) == 6
assert [etree.tostring(n) for n in old.findall('.//wp:inline', NS)] == [etree.tostring(n) for n in new.findall('.//wp:inline', NS)]
assert len(new.findall('.//wp:inline', NS)) == 6
old_tables, tables = old.findall('.//w:tbl', NS), new.findall('.//w:tbl', NS)
assert len(old_tables) == 1 and len(tables) == 2
assert etree.tostring(old_tables[0]) == etree.tostring(tables[0])
rows = [[value(c) for c in r.findall('w:tc', NS)] for r in tables[1].findall('w:tr', NS)]
expected = [['Model', 'AP', 'APs', 'AR@100', 'ARl', 'Train params/M']]
for arm in ['FULL', 'NO_MPS']:
    expected.append([arm] + [f'{primary["fractions"][arm][k]:.4f}' for k in ['AP', 'APs', 'AR100', 'ARl']] +
                    [f'{comparison["trainable_parameters"][arm] / 1e6:.4f}'])
assert rows == expected
for tag in ['w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference']:
    assert sorted(n.get('{' + W + '}id') for n in new.findall('.//' + tag, NS)) == ['0', '1', '2', '3', '4']
assert sorted(n.get('{' + W + '}id') for n in comments.findall('w:comment', NS)) == ['0', '1', '2', '3', '4']
paragraphs = [value(p) for p in new.findall('w:body/w:p', NS)]
old_paragraphs = [value(p) for p in old.findall('w:body/w:p', NS)]
references = [p for p in paragraphs if re.match(r'^\[\d+\]', p)]
assert len(references) == 13
assert references == [p for p in old_paragraphs if re.match(r'^\[\d+\]', p)]
body = '\n'.join(paragraphs[:paragraphs.index('References')])
cited = set()
for match in re.findall(r'(?<![=\w])\[([\d, ]+)\]', body):
    cited.update(int(v) for v in match.split(',') if v.strip())
assert cited == set(range(1, 14))
assert all(s in body for s in ['NO_MSCA remains in progress', 'NO_MPS began at round zero',
                             'excluding 11 later rows', 'Differences are computed from unrounded metrics',
                             '7,540,942', '7,535,617', '3,699,662', '2.02 times',
                             'one eight-head layer', 'three-layer decoder', '300 regular queries',
                             'six sampling points per feature level', 'drops its final incomplete batch'])
assert 'Both removal runs are ongoing' not in body
for k, rounded in [('AP', '1.57'), ('APs', '1.77'), ('ARl', '-1.69')]:
    assert f'{primary["FULL_minus_NO_MPS_percentage_points"][k]:.2f}' == rounded
    assert rounded in body
assert comparison['FULL_minus_NO_MPS_trainable_parameters'] == 5325
assert '5,325' in body
abstract = next(p for p in paragraphs if p.startswith('Small objects in unmanned aerial vehicle'))
assert len(abstract.split()) == 196
assert 'sigmoid-weighted linear unit (SiLU)' in abstract

manual_path = HERE / 'acronym_manual_adjudication.json'
qa_path = HERE / 'mps_endpoint_final_visual_qa.json'
note_path = HERE / 'MPS终点并入与未完成事项.txt'
index_path = RECORDS / '主要结论证据索引/主要结论证据索引_第三十四轮.json'
checklist_path = PAPER / '远程公平对照_20261103/结果并入与验收清单.txt'
checklist_backup = HERE / '结果验收清单_更新前.txt'
for path in [manual_path, qa_path, note_path, index_path, index_path.with_suffix('.txt'), checklist_backup]:
    assert not path.exists(), path
write_json(manual_path, {
    'status': 'APPLICABLE_FIRST_USE_ISSUES_FIXED_REMAINING_HEURISTIC_FLAGS_ADJUDICATED',
    'candidate_sha256': NEW_SHA,
    'body_raw_report_sha256': digest(HERE / 'acronym_body_v4_report.json'),
    'abstract_raw_report_sha256': digest(HERE / 'acronym_abstract_v4_report.json'),
    'coverage': 'Separate standalone abstract and body inputs; raw reports are partial, not zero findings.',
    'fixed_this_revision': ['SiLU expanded in the standalone abstract', 'ARl defined with large-area threshold and maxDets=100'],
    'inherited_adjudication': relative(RECORDS / '第三十三轮实验主线重组_20261108/acronym_manual_adjudication.json'),
    'remaining_named_methods': ['DEIM', 'D-FINE/FINE/DFINE', 'MPSMA-DEIM/MPSMA', 'SimAM', 'DMF-Net/DMF', 'TF-DEIM-DFINE/TF', 'HG block/HG'],
    'remaining_metrics_math_or_labels': ['HW: spatial height times width', 'AP50/AP75: explicitly defined IoU metrics',
                                         'ARl: explicitly defined large-object recall', 'BASE/FULL: defined experimental roles',
                                         'NO: checker fragment of NO_MPS/NO_MSCA labels'],
    'remaining_names': ['CUDA', 'NVIDIA', 'RTX'],
    'manual_first_use_review': ['UAV', 'MPS', 'MSCA', 'SiLU', 'NMS', 'ESE', 'ReLU', 'COCO', 'IoU', 'AP', 'APs', 'AR', 'EMA', 'BN', 'GAP', 'LAB', 'FP32', 'TF32', 'GPU']})
write_json(qa_path, {
    'status': 'PASSED', 'revision': 34, 'observed_local_utc': NOW,
    'candidate_sha256': NEW_SHA, 'source_sha256': OLD_SHA,
    'renderer': 'Hidden macro-disabled read-only Word PDF export and bundled Poppler',
    'packaged_renderer_log': 'packaged_render_v4.log',
    'packaged_renderer_limitation': 'Missing soffice.exe; preserved canonical failure log, hidden Word fallback used.',
    'pages_visually_inspected_individually': list(range(1, 9)),
    'visual_result': 'All eight final v4 pages individually inspected. Tables, equations, diagrams, full curves and references readable without clipping or overlap. Reference 13 no longer occupies a separate ninth page. Font sizes and styles were not reduced.',
    'page_png_sha256': {p.name: digest(p) for p in page_paths},
    'changed_docx_parts': changed_parts, 'equations_preserved': 6, 'figures_preserved': 6,
    'primary_pair_table_preserved_exactly': True, 'new_module_table_recomputed': rows,
    'references_preserved_and_still_cited': 13, 'valid_comment_anchors': 5,
    'abstract_word_count': 196, 'new_model_forwards': 0, 'new_evaluations': 0,
    'submission_ready': False})
note = (
    '第三十四轮 MPS 条件移除终点并入\n\n'
    'NO_MPS 已完成 320 轮并真实退出 0。其完整轨迹、517,440 条学习率、17 次后期恢复、原始模型与 EMA、保存的主/最高 AP/阶段一权重、548 图预测和评价数组均已核验。'
    '主端点与最高 AP 次端点同为实际第 320 轮，权重 SHA-256 fdd663b4b0be38208a382f8807d5b252feb2afc5b60308a6976b69963a2eab4e。核验整理既有产物，没有新增模型推理或评价器运行。\n\n'
    'Table 2 新增固定第 320 轮 FULL/NO_MPS 条件比较，AP 为 0.2255/0.2099，APs 为 0.1539/0.1362。'
    'FULL−NO_MPS 为 AP +1.565409、APs +1.765416 个百分点；同时保留大目标 ARl −1.686903 个百分点，不能声称所有指标改善。'
    '差值用未取舍的原始指标计算，正文显示 +1.57/+1.77/−1.69 个百分点。参数 7,540,942/7,535,617，差 5,325。\n\n'
    '此比较保留 FULL 的其余结构、SiLU 和聚合设置，替换 stem 来自原 BASE。NO_MPS 从零开始，FULL 在第一轮后断点恢复，不能声称训练轨迹逐位相同、统计显著、机制独立于容量或两模块协同。'
    '共同源码、依赖、配方、数据、首 100 批和实际初始化的补核证据仍保留。BASE 不是双移除组。\n\n'
    '摘要、4.4 节、限制和结论同步更新；补充实际 n 配置的一层八头编码器、三层解码器、300 个常规查询和每层六采样点，说明训练尾批丢弃及 ARl 定义。'
    '原主对照表、六幅图、六条原生公式和十三条引用保留；五条有效批注及锚点保留。最终 v4 为 8 页，逐页检查通过。原稿、原稿备份及第三十三轮精确备份均未改变。\n\n'
    '尚未完成：NO_MSCA 的完整 320 轮终点、全轨迹和退出/备份核验；依据其真实结果更新条件贡献分析并对全文再核对及渲染；选会后的模板、页数和投稿规则适配。'
    '目前尚不能投稿。30219 的 NO_MPS 已结束；30848 正在运行既有授权的 NO_MSCA，不自动增加其他训练。\n')
with note_path.open('x', encoding='utf-8') as stream:
    stream.write(note)
pair_path = PAPER / '远程公平对照_20261103/断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
previous_index = RECORDS / '主要结论证据索引/主要结论证据索引_第三十三轮.json'
evidence_paths = [pair_path, comparison_path, lineage_path, BUNDLE / 'NO_MPS/lineage_through_round309.json',
                  BUNDLE / 'NO_MPS/completed_live_20261108.json',
                  BUNDLE / 'NO_MPS/status_user_20261109_0042.json', BUNDLE / 'NO_MSCA/status_user_20261109_0042.json',
                  BUNDLE / 'module_shared_conditions_verified_20261108.json',
                  BUNDLE / 'module_conditions_evidence_supplement_20261108.json',
                  BUNDLE / 'NO_MSCA/lineage_through_round005.json',
                  HERE / 'mps_endpoint_candidate.json', HERE / 'mps_endpoint_polish.json',
                  HERE / 'unrounded_differences_clarification.json', HERE / 'orphan_reference_resolution.json',
                  manual_path, qa_path, note_path]
index = {'revision': 34, 'status': 'MAIN_PAIR_AND_CONDITIONAL_MPS_ENDPOINTS_VERIFIED_NO_MSCA_PENDING',
         'date': '2026-11-08', 'observed_local_utc': NOW, 'path_basis': 'relative to 01_会议论文',
         'manuscript': {'path': relative(MASTER), 'sha256': NEW_SHA, 'content_revision': 34},
         'previous_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
         'previous_manuscript_backup': {'path': relative(BACKUP), 'sha256': OLD_SHA},
         'original_manuscript_sha256': ORIGINAL_SHA,
         'evidence_files': {relative(p): digest(p) for p in evidence_paths},
         'conditional_MPS_primary': primary,
         'conditional_MPS_secondary': comparison['comparisons']['secondary_highest_AP'],
         'negative_result_preserved': 'FULL minus NO_MPS ARl = -1.6869030784671835 percentage points.',
         'open_items': ['NO_MSCA complete 320-round endpoints, lineage and genuine exit',
                        'Complete conditional analysis and final whole-paper review/render', 'Venue, template and page limit'],
         'scope': 'Single seed, conditional removals, disclosed continuation/RNG and capacity differences, local COCO-style evaluation; no significance or interaction claim.',
         'submission_ready': False}
write_json(index_path, index)
with index_path.with_suffix('.txt').open('x', encoding='utf-8') as stream:
    stream.write('第三十四轮主要结论证据索引\n\n稿件 SHA-256: ' + NEW_SHA + '\n\n' + note +
                 '\n证据路径与 SHA-256:\n' + '\n'.join(p + '  ' + h for p, h in index['evidence_files'].items()) + '\n')
with checklist_backup.open('xb') as stream:
    stream.write(checklist_path.read_bytes())
assert digest(MASTER) == OLD_SHA and digest(CANDIDATE) == NEW_SHA
temporary = MASTER.with_suffix('.rev34.tmp')
with temporary.open('xb') as stream:
    stream.write(CANDIDATE.read_bytes())
os.replace(temporary, MASTER)
assert digest(MASTER) == NEW_SHA

state_path = BUNDLE / 'current_execution_state.json'
state = read(state_path)
state.update(paper_revision=34, paper_sha256=NEW_SHA, final_comparisons_pending=True, observed_local_utc=NOW)
for arm in ['NO_MPS', 'NO_MSCA']:
    events = [json.loads(line) for line in (Path(state[arm]['backup_root']) / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted = next(e for e in reversed(events) if e['status'] == 'EPOCH_BACKUP_VERIFIED')
    state[arm]['last_accepted_round'] = accepted['epoch'] + 1
    state[arm]['last_acceptance_utc'] = accepted['observed_utc']
    wrapper_path = BUNDLE / arm / 'status_user_20261109_0042.json'
    snapshot = json.loads(read(wrapper_path)['stdout'])
    detail = snapshot['arms']['no_mps_20261107_v1' if arm == 'NO_MPS' else 'no_msca_20261107_v1']
    old_observation = state[arm].get('last_remote_live_observation')
    state[arm]['previous_remote_live_observation'] = old_observation
    state[arm]['last_remote_live_observation'] = {
        'file': str(wrapper_path), 'wrapper_sha256': digest(wrapper_path),
        'server_observed_utc': snapshot['observed_utc'], 'boot_id': snapshot['boot_id'],
        'supervisor_status': detail['supervisor.json']['status'], 'receipt': detail['receipt'],
        'gpu': snapshot['gpu'], 'gpu_processes': snapshot['gpu_processes']}
state['NO_MPS'].update(status='COMPLETED_AND_BACKED_UP', completed_rounds=320,
                       independent_lineage_audit_round=320, terminal_and_best_endpoints_verified=True,
                       completed_client_expected_to_be_stopped=True)
atomic_json(state_path, state)
header = (
    '【最新稿件与证据入口：2026-11-08，第三十四轮 MPS 终点并入】\n'
    '持续修订稿已更新至第三十四轮，SHA256 ' + NEW_SHA + '，8 页逐页核验通过。BASE/FULL 主对照与 NO_MPS 条件移除均已完成 320 轮，产物、全轨迹与退出/备份已核验。\n'
    'Table 2 报告 FULL/NO_MPS，AP 和 APs 差 +1.57/+1.77 个百分点，同时保留 ARl −1.69 个百分点与 5,325 参数差；差值按未取舍指标计算。摘要和结论同步更新，没有宣称全指标改善、统计显著或模块协同。\n'
    'NO_MSCA 正在运行，尚待完整终点及全轨迹核验。已有运行继续，不自动增加训练。当前轮次见模块消融/current_execution_state.json 与逐轮备份回执。\n'
    '原稿、第三十三轮备份、主对照表、六幅图、六条公式、十三条引用和五条有效批注保留。详见第三十四轮证据索引及第三十四轮MPS移除终点_20261108/MPS终点并入与未完成事项.txt。尚不能投稿；以下为历史记录。\n\n')
for path in [PAPER / '当前修订入口.txt', RECORDS / '修订说明与未解决问题.txt']:
    old_text = path.read_text(encoding='utf-8-sig').replace('【最新稿件与证据入口：2026-11-08，第三十三轮实验主线重组】',
                                                         '【历史稿件与证据入口：2026-11-08，第三十三轮实验主线重组】', 1)
    path.write_text(header + old_text, encoding='utf-8')
checklist_header = (
    '2026-11-08 当前结果并入与验收状态：第三十四轮\n\n'
    '已完成：BASE/FULL 均 320 轮，主及最高 AP 次端点和配对轨迹已核验；NO_MPS 320 轮真实退出 0，主/最高 AP/阶段一权重、预测、评价数组和 517,440 条学习率及 17 次恢复已核验。\n'
    '持续稿第三十四轮，SHA-256 ' + NEW_SHA + '，8 页完成逐页检查。Table 1 保留主对照，Table 2 新增 FULL/NO_MPS：AP +1.565409、APs +1.765416、ARl −1.686903 个百分点；5325 个参数差。正文四舍五入为 +1.57/+1.77/−1.69。\n'
    '尚待 NO_MSCA 完整 320 轮及全轨迹、权重—预测—评价—指标绑定和真实退出/备份核验。此后再据真实结果完善条件贡献分析与全文核对/渲染，选会后适配模板和页数。\n'
    '共同源码、依赖、数据、首 100 批与初始化补核已通过；单种子、验证选择、容量差、断点/RNG 差及本地评价口径等限制继续适用。不提供 MPS×MSCA 交互结论，不自动新增训练。尚不能投稿。\n'
    '索引：修订记录_20261103/主要结论证据索引/主要结论证据索引_第三十四轮.json。以下为历史状态，不覆盖上方当前结论。\n\n')
checklist_path.write_text(checklist_header + checklist_backup.read_text(encoding='utf-8-sig'), encoding='utf-8')
state['acceptance_checklist_sha256'] = digest(checklist_path)
atomic_json(state_path, state)
with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## 2026-11-08 第三十四轮 MPS 终点并入\n\n' + note + '\n')
print(json.dumps({'status': index['status'], 'revision': 34, 'pages': 8,
                  'manuscript_sha256': digest(MASTER), 'NO_MPS_completed': 320,
                  'NO_MSCA_accepted_round': state['NO_MSCA']['last_accepted_round'],
                  'submission_ready': False}))
