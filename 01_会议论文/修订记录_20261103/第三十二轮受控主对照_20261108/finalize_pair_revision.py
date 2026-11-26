"""Verify and publish the visually reviewed revision, with evidence links."""
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
PAPER = HERE.parent.parent
RECORDS = HERE.parent
MASTER = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_rev31_backup.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_rev32_v2.docx'
OLD_SHA = 'cb213ccd8553e8965485ce4b588b1a32752dffccd3f706581979df59b23dfe98'
NEW_SHA = 'f8c5739e2394375160a7f606aec3aa47d632d1dab5376d90f14f148db3914c79'
PAIR = PAPER / '远程公平对照_20261103/断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
BUNDLE = PAPER / '远程公平对照_20261103/模块消融'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
NOW = datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def exclusive_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def text(element):
    return ''.join(element.xpath('.//w:t/text()', namespaces=NS))


def relative(path):
    return path.relative_to(PAPER).as_posix()


def main():
    assert digest(MASTER) == digest(BACKUP) == OLD_SHA
    assert digest(CANDIDATE) == NEW_SHA
    assert read(HERE / 'paired_revision_polish.json')['candidate_sha256'] == NEW_SHA
    pair = read(PAIR)
    assert pair['status'] == 'RESUMED_PAIR_ENDPOINTS_VERIFIED_PENDING_MANUSCRIPT_REVIEW'
    assert pair['result_generated'] is True and pair['submission_ready'] is False
    assert pair['shared_conditions']['budget']['epochs_per_group'] == 320
    render = read(HERE / 'render_metadata_v2.json')
    pages = sorted((HERE / 'rendered_word_v3').glob('page-*.png'))
    assert render['page_count'] == len(pages) == 10
    with ZipFile(BACKUP) as before, ZipFile(CANDIDATE) as after:
        assert after.testzip() is None
        assert before.namelist() == after.namelist()
        changed = [name for name in before.namelist() if before.read(name) != after.read(name)]
        assert set(changed) == {'word/document.xml', 'word/comments.xml'}
        old = etree.fromstring(before.read('word/document.xml'))
        new = etree.fromstring(after.read('word/document.xml'))
        comments = etree.fromstring(after.read('word/comments.xml'))
        assert b'\xef\xbf\xbd' not in after.read('word/document.xml')
        assert b'\xef\xbf\xbd' not in after.read('word/comments.xml')
    old_tables = old.findall('.//w:tbl', NS)
    tables = new.findall('.//w:tbl', NS)
    assert len(old_tables) == 6 and len(tables) == 7
    assert [etree.tostring(t) for t in old_tables] == [etree.tostring(t) for t in tables[:6]]
    for tag, expected in [('m:oMath', 6), ('wp:inline', 7)]:
        original_nodes, new_nodes = old.findall('.//' + tag, NS), new.findall('.//' + tag, NS)
        assert len(original_nodes) == len(new_nodes) == expected
        assert [etree.tostring(n) for n in original_nodes] == [etree.tostring(n) for n in new_nodes]
    anchor_ids = None
    for tag in ['w:commentRangeStart', 'w:commentRangeEnd', 'w:commentReference']:
        ids = sorted(n.get('{' + NS['w'] + '}id') for n in new.findall('.//' + tag, NS))
        assert len(ids) == len(set(ids)) == 5
        if anchor_ids is None:
            anchor_ids = ids
        assert ids == anchor_ids
    assert sorted(n.get('{' + NS['w'] + '}id') for n in comments.findall('w:comment', NS)) == anchor_ids
    paragraphs = [text(p) for p in new.findall('w:body/w:p', NS)]
    assert [int(re.match(r'^\[(\d+)\]', p).group(1)) for p in paragraphs if re.match(r'^\[\d+\]', p)] == list(range(1, 14))
    all_text = '\n'.join(paragraphs)
    assert 'Paired results are pending.' not in all_text
    assert all_text.count('4.6 Comparison of the n Configuration') == 1
    assert all_text.count('4.7 Evidence Limits') == 1
    assert 'faster-coco-eval 1.8.0' in all_text and 'faster-coco-eval 1.6.5' in all_text
    expected_rows = [['Endpoint', 'Model', 'Round', 'AP', 'AP50', 'AP75', 'APs', 'AR@100']]
    for label, key in [('Primary', 'terminal'), ('Secondary', 'best')]:
        comparison = pair['comparisons'][key]
        for arm in ['BASE', 'FULL']:
            metrics = comparison['absolute_percent'][arm]
            expected_rows.append([label, arm, str(comparison['epoch_indices'][arm] + 1)] +
                                 [f'{metrics[metric] / 100:.4f}' for metric in ['AP', 'AP50', 'AP75', 'APs', 'AR100']])
    actual_rows = [[text(cell) for cell in row.findall('w:tc', NS)] for row in tables[6].findall('w:tr', NS)]
    assert actual_rows == expected_rows
    qa_path = HERE / 'paired_revision_visual_qa.json'
    exclusive_json(qa_path, {
        'status': 'PASSED', 'revision': 32, 'observed_local_utc': NOW,
        'candidate_sha256': NEW_SHA, 'master_before_sha256': OLD_SHA,
        'renderer': render['renderer'], 'page_count': 10,
        'visually_inspected_pages': list(range(1, 11)),
        'visual_result': 'All final page PNGs inspected individually; tables, diagrams, equations, symbols and bibliography are readable, without clipping, overlap or an orphan reference page.',
        'packaged_renderer_limitation': 'Bundled render_docx.py failed because soffice.exe is unavailable; its log is preserved. The fallback uses a hidden macro-disabled read-only Word export and bundled Poppler.',
        'changed_package_parts': changed, 'existing_tables_preserved': 6,
        'new_table_rows_verified_from_pair_report': actual_rows,
        'equations_preserved': 6, 'inline_images_preserved': 7,
        'comment_anchors_and_comments': 5, 'references': 13,
        'submission_ready': False})
    note = (
        '# 第三十二轮主对照结果与稿件修订\n\n'
        'BASE 和 FULL 均已完成 320 轮，终点权重、评测产物、日志、恢复轨迹及正常退出回执已通过配对核验。'
        '本轮将这组结果并入英文摘要、引言、4.6 节、Table 7、证据限制和结论，并更新三条批注。\n\n'
        '主端点为实际第 320 轮、后期恢复策略执行前评测的 EMA。BASE 的 AP/APs 为 '
        '19.021600%/12.153124%，FULL 为 22.551324%/15.385722%，差值为 3.529724/3.232599 个百分点。'
        '最高 AP 次端点分别为第 303 和 305 轮，所有其他指标跟随同一权重。表中使用比例值，正文差值使用百分点。\n\n'
        '共同条件包括 seed 0、batch 4、FP32、6471 张训练图、548 张验证图、320 轮及冻结源码与评价方案。'
        '已明确两组 GPU 运行使用 PyTorch 2.5.1+cu124、CUDA 12.4 和 faster-coco-eval 1.8.0，'
        '与历史权重 CPU 重评的 2.6.0/1.6.5 环境分开。FULL 的 7540942 个训练参数与 BASE 的 3737611 个参数同时报告。\n\n'
        '此对照比较完整骨干配置，包含容量、聚合路径和激活的共同变化，不能单独归因于 MPS 或 MSCA。'
        'BASE 在第 214、308 轮后恢复，FULL 在第 1 轮后恢复；最早断点缺少主进程随机状态，BASE 的部分第 309 轮重跑。'
        '这些限制继续披露，不宣称连续从头运行、位级复现、多种子稳定性或统计显著性。'
        '冻结本地 COCO-style 口径与官方 VisDrone ignore/maxDets 口径的差异保留。\n\n'
        '原稿和第三十一轮备份保留。原有六张表、七个内嵌图像、六条公式和五个批注锚点均保留；'
        '新增 Table 7 的各项数值直接按已通过核验的配对报告生成并再次核对。最终 10 页已逐页查看。\n\n'
        'NO_MPS 和 NO_MSCA 两组获授权条件移除实验仍在运行，终点和完整轨迹尚待核验；'
        '后续需据此完善组件分析和稿件组织。当前尚不能投稿。会议尚未选择，页数和模板待选会后适配。\n')
    note_path = HERE / '主对照结果与稿件修订.md'
    with note_path.open('x', encoding='utf-8') as stream:
        stream.write(note)
    evidence_paths = [PAIR, BUNDLE / 'method_source_binding_20261108.json',
                      BUNDLE / 'authorization_and_scope.json',
                      BUNDLE / 'NO_MPS/lineage_through_round297.json',
                      BUNDLE / 'NO_MSCA/first_round_verified.json',
                      BUNDLE / 'NO_MSCA/frozen_inputs_verified_20261108.json',
                      HERE / 'paired_revision_candidate.json', HERE / 'paired_revision_polish.json',
                      qa_path, note_path]
    previous_index = RECORDS / '主要结论证据索引/主要结论证据索引_第三十一轮.json'
    index = {
        'revision': 32, 'status': 'PAIRED_ENDPOINTS_INCORPORATED_MODULE_ENDPOINTS_PENDING',
        'date': '2026-11-08', 'observed_local_utc': NOW,
        'path_basis': 'relative to 01_会议论文',
        'manuscript': {'path': relative(MASTER), 'sha256': NEW_SHA, 'content_revision': 32},
        'previous_index': relative(previous_index), 'previous_index_sha256': digest(previous_index),
        'previous_manuscript_backup': {'path': relative(BACKUP), 'sha256': OLD_SHA},
        'original_manuscript_sha256': 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323',
        'evidence_files': {relative(p): digest(p) for p in evidence_paths},
        'primary': pair['comparisons']['terminal'], 'secondary': pair['comparisons']['best'],
        'claim_scope': pair['scope'],
        'open_items': ['NO_MPS/NO_MSCA 320-round endpoints and complete trajectory audit',
                       'Component-specific analysis and final manuscript organization',
                       'Venue selection and page/template adaptation'],
        'submission_ready': False}
    index_path = RECORDS / '主要结论证据索引/主要结论证据索引_第三十二轮.json'
    exclusive_json(index_path, index)
    with (index_path.with_suffix('.txt')).open('x', encoding='utf-8') as stream:
        stream.write('第三十二轮主要结论证据索引\n\n当前稿件 SHA-256: ' + NEW_SHA + '\n\n' + note +
                     '\n证据文件与 SHA-256:\n' + '\n'.join(path + '  ' + value for path, value in index['evidence_files'].items()) + '\n')
    assert digest(MASTER) == OLD_SHA and digest(CANDIDATE) == NEW_SHA
    temporary = MASTER.with_suffix('.rev32.tmp')
    with temporary.open('xb') as stream:
        stream.write(CANDIDATE.read_bytes())
    os.replace(temporary, MASTER)
    assert digest(MASTER) == NEW_SHA
    state_path = BUNDLE / 'current_execution_state.json'
    state = read(state_path)
    state.update(paper_revision=32, paper_sha256=NEW_SHA,
                 final_comparisons_pending=True, observed_local_utc=NOW)
    state['NO_MSCA']['input_byte_verification_pending'] = False
    temporary = state_path.with_suffix('.rev32.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, state_path)
    current_header = (
        '【最新稿件与证据入口：2026-11-08，第三十二轮受控主对照】\n'
        '当前持续修订稿为第三十二轮，SHA256 ' + NEW_SHA + '。BASE/FULL 完整 320 轮主次端点及进程退出回执均已核验。\n'
        '已并入摘要、引言、4.6 节 Table 7、限制和结论：主端点 AP 为 19.021600%/22.551324%，APs 为 12.153124%/15.385722%，'
        'FULL−BASE 分别为 3.529724/3.232599 个百分点。保留单种子、恢复、本地评价口径及容量差异限制。\n'
        '第三十一轮备份已保存，最终 10 页逐页检查通过。详见第三十二轮证据索引及第三十二轮受控主对照_20261108/主对照结果与稿件修订.md。\n'
        'NO_MPS 和 NO_MSCA 两组条件移除对照继续运行，最新轮次以模块消融/current_execution_state.json 与逐轮备份回执为准。'
        '模块终点及贡献分析尚未完成，当前尚不能投稿。以下为历史记录，旧状态以本段和最新运行记录为准。\n\n')
    entry_path = PAPER / '当前修订入口.txt'
    entry = entry_path.read_text(encoding='utf-8-sig')
    entry = entry.replace('【最新稿件与证据入口：2026-11-06，第三十一轮 BASE 终点核验】',
                          '【历史稿件与证据入口：2026-11-06，第三十一轮 BASE 终点核验】', 1)
    entry_path.write_text(current_header + entry, encoding='utf-8')
    issues_path = RECORDS / '修订说明与未解决问题.txt'
    issues = issues_path.read_text(encoding='utf-8-sig')
    issues = issues.replace('【当前状态摘要 2026-11-08】', '【历史状态摘要 2026-11-08，第三十一轮】', 1)
    issues_path.write_text(current_header + issues, encoding='utf-8')
    with (BUNDLE / '方案与执行记录.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 2026-11-08 第三十二轮主对照结果并稿\n\n'
                     '已将通过完整配对核验的 BASE/FULL 主次端点并入持续维护的英文稿，原稿及第三十一轮备份保留，'
                     '10 页视觉核验通过。详见第三十二轮证据索引和修订记录。'
                     'NO_MPS/NO_MSCA 继续运行，尚无条件移除终点，不据此宣称模块独立收益或投稿完成。\n')
    print(json.dumps({'status': index['status'], 'manuscript_sha256': digest(MASTER),
                      'revision': 32, 'pages': 10, 'submission_ready': False}))


if __name__ == '__main__':
    main()
