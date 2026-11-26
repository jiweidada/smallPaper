"""Publish the content-reviewed and visually inspected revision 33."""
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
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
RECORDS = HERE.parent
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev32_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev33_v3.docx'
OLD_SHA = 'f8c5739e2394375160a7f606aec3aa47d632d1dab5376d90f14f148db3914c79'
NEW_SHA = 'b3aae17cf12667f34797179f399dcfd608edfab724633f023921c9703b0d4cee'
ORIGINAL_SHA = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'
NOW = datetime.now(timezone.utc).isoformat()
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def text(element):
    return ''.join(element.xpath('.//w:t/text()', namespaces=NS))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def relative(path):
    return path.relative_to(PAPER).as_posix()


def main():
    assert digest(MASTER) == digest(BACKUP) == OLD_SHA
    assert digest(CANDIDATE) == NEW_SHA
    assert digest(PAPER / '稿件/MPSMA-DEIM_conference_EN.docx') == ORIGINAL_SHA
    assert digest(PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx') == ORIGINAL_SHA
    assert read(HERE / 'acronym_terminology_changes.json')['candidate_sha256'] == NEW_SHA
    assert read(HERE / 'render_metadata_final.json')['page_count'] == 8
    page_paths = sorted((HERE / 'rendered_word_final').glob('page-*.png'))
    assert len(page_paths) == 8 and all(path.stat().st_size > 100000 for path in page_paths)
    pair_path = PAPER / '远程公平对照_20261103/断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
    pair = read(pair_path)
    curve = read(HERE / 'learning_curve_provenance.json')
    assert curve['pair_report_sha256'] == digest(pair_path)
    archive = read(HERE / 'historical_experimental_context_rev32.json')
    assert archive['source_sha256'] == OLD_SHA
    assert sum(block['type'] == 'table' for block in archive['blocks']) == 7
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert after.testzip() is None
        assert set(after.namelist()) - set(before.namelist()) == {'word/media/paired_validation_curves.png'}
        changed_parts = [name for name in before.namelist() if before.read(name) != after.read(name)]
        assert set(changed_parts) == {'word/document.xml', 'word/comments.xml', 'word/_rels/document.xml.rels'}
        old = etree.fromstring(before.read('word/document.xml'))
        new = etree.fromstring(after.read('word/document.xml'))
        comments = etree.fromstring(after.read('word/comments.xml'))
        assert b'\xef\xbf\xbd' not in after.read('word/document.xml')
        assert b'\xef\xbf\xbd' not in after.read('word/comments.xml')
        assert sha256(after.read('word/media/paired_validation_curves.png')).hexdigest() == curve['figure_sha256']['paired_validation_curves.png']
        for name in before.namelist():
            if name.startswith('word/media/'):
                assert before.read(name) == after.read(name)
    assert [etree.tostring(n) for n in old.findall('.//m:oMath', NS)] == [etree.tostring(n) for n in new.findall('.//m:oMath', NS)]
    assert len(new.findall('.//m:oMath', NS)) == 6
    assert [etree.tostring(n) for n in old.findall('.//wp:inline', NS)[:5]] == [etree.tostring(n) for n in new.findall('.//wp:inline', NS)[:5]]
    assert len(new.findall('.//wp:inline', NS)) == 6
    tables = new.findall('.//w:tbl', NS)
    assert len(tables) == 1
    actual_rows = [[text(cell) for cell in row.findall('w:tc', NS)] for row in tables[0].findall('w:tr', NS)]
    expected = [['Endpoint', 'Model', 'Round', 'AP', 'AP50', 'AP75', 'APs', 'AR@100']]
    for label, role in [('Primary', 'terminal'), ('Secondary', 'best')]:
        for arm in ['BASE', 'FULL']:
            comparison = pair['comparisons'][role]
            metrics = comparison['absolute_percent'][arm]
            expected.append([label, arm, str(comparison['epoch_indices'][arm] + 1)] +
                            [f'{metrics[key] / 100:.4f}' for key in ['AP', 'AP50', 'AP75', 'APs', 'AR100']])
    assert actual_rows == expected
    for tag in ['w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference']:
        assert sorted(n.get('{' + W + '}id') for n in new.findall('.//' + tag, NS)) == ['0', '1', '2', '3', '4']
    assert sorted(n.get('{' + W + '}id') for n in comments.findall('w:comment', NS)) == ['0', '1', '2', '3', '4']
    paragraphs = [text(p) for p in new.findall('w:body/w:p', NS)]
    references = [p for p in paragraphs if re.match(r'^\[\d+\]', p)]
    assert len(references) == 13
    body_text = '\n'.join(paragraphs[:paragraphs.index('References')])
    cited = set()
    for match in re.findall(r'(?<![=\w])\[([\d, ]+)\]', body_text):
        cited.update(int(value) for value in match.split(',') if value.strip())
    assert cited == set(range(1, 14))
    assert 'excluding 11 later rows' in body_text and 'Both removal runs are ongoing' in body_text
    assert '44.30' not in body_text and '0.2876' not in body_text and 'Table 7' not in body_text
    assert all(phrase in body_text for phrase in ['7,540,942', '3,737,611', '7,535,617', '3,699,662',
                                                  '3.53', '3.23', '2.02 times', 'Common Objects in Context (COCO)',
                                                  'Average recall (AR)', 'TensorFloat-32 (TF32)'])
    acronym_review_path = HERE / 'acronym_manual_adjudication.json'
    write_json(acronym_review_path, {
        'status': 'APPLICABLE_FIRST_USE_ISSUES_FIXED_REMAINING_TOOL_FLAGS_ADJUDICATED',
        'candidate_sha256': NEW_SHA,
        'checks': {'body': digest(HERE / 'acronym_body_final_report.json'),
                   'standalone_English_abstract': digest(HERE / 'acronym_abstract_final_report.json')},
        'note': 'The abstract and body were checked separately to avoid carrying definitions across scopes. The checker labels both standalone text inputs as body. Its raw reports remain unchanged and have partial heuristic coverage; this is not a claim of zero automated findings.',
        'fixed': ['COCO expanded in the abstract and body', 'AR expanded before AR@100',
                  'FP32, TF32 and GPU clarified at first body use'],
        'remaining_named_methods': ['DEIM', 'D-FINE/FINE/DFINE', 'MPSMA-DEIM/MPSMA', 'SimAM', 'DMF-Net/DMF', 'TF-DEIM-DFINE/TF', 'HG block/HG'],
        'remaining_math_metrics_or_labels': {'HW': 'H times W in the adjacent variance explanation',
                                            'AP50/AP75': 'IoU-specific AP metrics explicitly defined in Section 4.1',
                                            'BASE/FULL': 'Experimental roles explicitly defined in Section 4.2',
                                            'NO': 'Checker splits the NO_MPS/NO_MSCA run identifiers'},
        'remaining_hardware_or_software_names': ['CUDA', 'NVIDIA', 'RTX'],
        'manual_definition_review': ['UAV', 'MPS', 'MSCA', 'SiLU', 'NMS', 'ESE', 'ReLU',
                                     'COCO', 'IoU', 'AP', 'APs', 'AR', 'EMA', 'BN', 'GAP', 'LAB']})
    qa_path = HERE / 'reorganization_final_visual_qa.json'
    write_json(qa_path, {
        'status': 'PASSED', 'revision': 33, 'observed_local_utc': NOW,
        'candidate_sha256': NEW_SHA, 'source_sha256': OLD_SHA,
        'renderer': 'Hidden macro-disabled read-only Word PDF export and bundled Poppler',
        'packaged_renderer_log': 'packaged_render_final.log',
        'packaged_renderer_limitation': 'soffice.exe unavailable; diagnosed from the preserved canonical-render log.',
        'pages_visually_inspected_individually': list(range(1, 9)),
        'visual_result': 'All eight final PNGs reviewed. All method diagrams, equations, the primary/secondary table, complete curves and references remain readable, without clipping, overlap or an orphan reference page. The full curve and its caption stay together.',
        'equations_preserved': 6, 'core_diagrams_preserved': 5,
        'archived_historical_tables': 6, 'current_tables': 1,
        'table_rows_recomputed_from_verified_pair': actual_rows,
        'all_13_references_still_cited': True, 'comments_and_valid_anchors': 5,
        'new_model_execution': False, 'new_evaluation': False, 'submission_ready': False})
    note_path = HERE / '实验主线重组与未完成事项.txt'
    note = (
        '第三十三轮实验主线重组与未完成事项\n\n'
        '正文现以新冻结 n 对照为实验主线，按数据及评价、实际训练与端点规则、完整模型结果、条件模块移除和限制组织。'
        '六张历史配置、n/s、复杂度、速度和 CPU 重评表移出正文，完整上下文及数值保存在本轮 historical_experimental_context_rev32.json/.txt，第三十二轮 DOCX 逐字节备份保留。'
        '这不是删除原始结果或用旧值替代新主对照。\n\n'
        'Table 1 仅含已核验 BASE/FULL 的实际第 320 轮主端点和第 303/305 轮最高 AP 次端点。'
        '主端点 AP/APs 的差值仍为 3.53/3.23 个百分点。摘要、引言和结论同时说明 FULL 参数为 BASE 的约 2.02 倍，不宣称轻量化或当前 n 模型的速度优势。\n\n'
        '图 6 改为既有完整日志的 320 轮 AP/APs 曲线。BASE 按原分支前 214 轮、备份 215–308 轮和恢复 309–320 轮拼接；FULL 按原第 1 轮和恢复 2–320 轮拼接。'
        '没有平滑，主端点与最高 AP 端点标识分开，APs 始终跟随 AP 选中权重，后期恢复阶段加阴影。CSV、完整行记录、绘图程序、PNG/PDF/SVG 与来源哈希留存。没有重新运行模型或评价器。\n\n'
        '补充披露 BASE 原中断分支在第 214 轮锚点后有 11 行不进入最终保留轨迹；第 308 轮恢复和部分第 309 轮重跑亦保留。'
        '320 轮表示保留轨迹长度，不表示两组总计算消耗完全相等。随机状态缺失、单种子、验证选择及本地 COCO/官方 VisDrone 差异继续说明，曲线不作为架构稳定性的证明。\n\n'
        'NO_MPS 使用原 BASE StemBlock 的参数并改用 SiLU，其他 FULL 结构保留；NO_MSCA 仅将两处 Lite 和一处 Dilated 门改为恒等映射，保留投影、ESE、SimAM、MPS 和 SiLU。'
        '两个运行尚未到完整终点，正文没有其性能数值。BASE 不等同于双移除组，这个设计不支持 MPS×MSCA 交互或协同效应结论。\n\n'
        '最终候选稿 8 页逐页检查通过，六条公式、五张核心结构图和十三条仍被引用的参考文献保留。缩略词脚本原始报告及人工判定保留，修正 COCO、AR、FP32、TF32、GPU 的适用首用说明。'
        '原稿及原稿备份 SHA-256 均重新核对无变更。\n\n'
        '待完成：NO_MPS 和 NO_MSCA 的完整 320 轮终点及全轨迹核验；按结果完善条件贡献分析并再次通读、核对和渲染；选会后的模板、页数和投稿要求适配。'
        '目前尚不能投稿。本轮是内容与证据组织修订，不是投稿完成认证。\n')
    with note_path.open('x', encoding='utf-8') as stream:
        stream.write(note)
    previous_index = RECORDS / '主要结论证据索引/主要结论证据索引_第三十二轮.json'
    evidence_paths = [pair_path, BUNDLE / 'method_source_binding_20261108.json',
                      BUNDLE / 'NO_MSCA/round005_verified_from_D.json',
                      BUNDLE / 'NO_MSCA/lineage_through_round005.json',
                      HERE / 'revision_scope.json', HERE / 'reorganization_candidate.json',
                      HERE / 'scope_terminology_polish.json', HERE / 'acronym_terminology_changes.json',
                      HERE / 'historical_experimental_context_rev32.json',
                      HERE / 'historical_experimental_context_rev32.txt',
                      HERE / 'learning_curve_provenance.json', HERE / 'paired_learning_curve_metrics.csv',
                      HERE / 'paired_learning_curve_rows.json', acronym_review_path, qa_path, note_path]
    index = {'revision': 33, 'status': 'VERIFIED_PAIR_FOCUSED_MODULE_ENDPOINTS_PENDING',
             'date': '2026-11-08', 'observed_local_utc': NOW,
             'path_basis': 'relative to 01_会议论文',
             'manuscript': {'path': relative(MASTER), 'sha256': NEW_SHA, 'content_revision': 33},
             'previous_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
             'previous_manuscript_backup': {'path': relative(BACKUP), 'sha256': OLD_SHA},
             'original_manuscript_sha256': ORIGINAL_SHA,
             'evidence_files': {relative(path): digest(path) for path in evidence_paths},
             'primary': pair['comparisons']['terminal'], 'secondary': pair['comparisons']['best'],
             'core_method_figures_and_equations': 'Preserved; four source files are byte-identical to the frozen training protocol.',
             'main_scope': 'New frozen n configuration comparison with explicit continuation and capacity limitations.',
             'historical_scope': 'Six historical tables and full context retained in exact DOCX backup and text/JSON archive; excluded from controlled comparisons.',
             'module_status': 'Both conditional removals running; no completed outcome or interaction claim.',
             'open_items': ['Both module-removal terminal endpoints and complete lineages',
                            'Conditional results, whole-paper final review and render',
                            'Venue, template and page limit'],
             'submission_ready': False}
    index_path = RECORDS / '主要结论证据索引/主要结论证据索引_第三十三轮.json'
    write_json(index_path, index)
    with index_path.with_suffix('.txt').open('x', encoding='utf-8') as stream:
        stream.write('第三十三轮主要结论证据索引\n\n稿件 SHA-256: ' + NEW_SHA + '\n\n' + note +
                     '\n证据路径与 SHA-256:\n' + '\n'.join(path + '  ' + value for path, value in index['evidence_files'].items()) + '\n')
    assert digest(MASTER) == OLD_SHA and digest(CANDIDATE) == NEW_SHA
    temporary = MASTER.with_suffix('.rev33.tmp')
    with temporary.open('xb') as stream:
        stream.write(CANDIDATE.read_bytes())
    os.replace(temporary, MASTER)
    assert digest(MASTER) == NEW_SHA
    state_path = BUNDLE / 'current_execution_state.json'
    state = read(state_path)
    state.update(paper_revision=33, paper_sha256=NEW_SHA, final_comparisons_pending=True,
                 observed_local_utc=NOW)
    for arm in ['NO_MPS', 'NO_MSCA']:
        events = [json.loads(line) for line in (Path(state[arm]['backup_root']) / 'monitor_events.jsonl').read_text(encoding='utf-8').splitlines()]
        accepted = next(event for event in reversed(events) if event['status'] == 'EPOCH_BACKUP_VERIFIED')
        state[arm]['last_accepted_round'] = accepted['epoch'] + 1
        state[arm]['last_acceptance_utc'] = accepted['observed_utc']
    state['NO_MSCA'].update(independent_state_audit_round=5, independent_lineage_audit_round=5)
    temporary = state_path.with_suffix('.rev33.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, state_path)
    header = (
        '【最新稿件与证据入口：2026-11-08，第三十三轮实验主线重组】\n'
        '持续修订稿已更新至第三十三轮，SHA256 ' + NEW_SHA + '，8 页逐页核验通过。正文现以新冻结 n 主对照为主线，'
        'Table 1 报告已核验的固定第 320 轮及最高 AP 次端点，图 6 展示完整 320 轮 AP/APs 轨迹。\n'
        '六张历史实验表与上下文完整留档，第三十二轮原样备份保留；没有将旧 s 速度数字用于新 n 权重。摘要和结论补入约 2.02 倍参数代价，'
        '披露 BASE 的 11 行中断分支排除及断点恢复等限制。\n'
        'NO_MPS/NO_MSCA 尚未完成完整终点，模块效果、交互及稳定性不作过度归因。NO_MSCA 前 5 轮状态与轨迹已独立核验，8,085 条学习率一致。'
        '最新轮次见模块消融/current_execution_state.json 与备份回执。\n'
        '详见第三十三轮证据索引及第三十三轮实验主线重组_20261108/实验主线重组与未完成事项.txt。目前尚不能投稿；以下为历史记录。\n\n')
    for path in [PAPER / '当前修订入口.txt', RECORDS / '修订说明与未解决问题.txt']:
        old_text = path.read_text(encoding='utf-8-sig')
        old_text = old_text.replace('【最新稿件与证据入口：2026-11-08，第三十二轮受控主对照】',
                                    '【历史稿件与证据入口：2026-11-08，第三十二轮受控主对照】', 1)
        path.write_text(header + old_text, encoding='utf-8')
    with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 2026-11-08 第三十三轮实验主线重组与第 5 轮核验\n\n'
                     '英文稿围绕新冻结 n 比较重组，旧六表完整归档，新增已有日志的全 320 轮 AP/APs 曲线；'
                     '补容量与中断分支限制，8 页完成逐页核验。新图只整理既有观察，没有重新运行模型或评价器。'
                     'NO_MSCA 前 5 轮状态和完整轨迹核验通过，8,085 条学习率误差为 0。两个训练与备份客户端继续运行。'
                     '全文终版仍待两个条件移除实验的终点和结果分析，尚不能投稿。\n')
    print(json.dumps({'status': index['status'], 'revision': 33, 'manuscript_sha256': digest(MASTER),
                      'pages': 8, 'NO_MPS_accepted_round': state['NO_MPS']['last_accepted_round'],
                      'NO_MSCA_accepted_round': state['NO_MSCA']['last_accepted_round'],
                      'submission_ready': False}))


if __name__ == '__main__':
    main()
