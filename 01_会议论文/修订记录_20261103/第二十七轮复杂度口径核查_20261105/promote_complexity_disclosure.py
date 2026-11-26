"""Promote the visually checked complexity disclosure and maintain the evidence trail."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDEX = REC / '主要结论证据索引'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十六轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十七轮候选.docx'
OLD = '185cbe8ffca81f86b10e8c12fab1e68eda90b722629617e870068e14a60f8993'
NEW = 'ee5aec68fc4a30cc2e814ab2bb9a08626196151e327570371bcd4deb92961588'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def ref(path):
    return {'path': str(path), 'sha256': digest(path)}


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def replace_text(path, value):
    temp = path.with_name(path.name + '.rev27.tmp')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    original = PAPER / '稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(CURRENT) == digest(BACKUP) == OLD and digest(CANDIDATE) == NEW
    assert digest(original) == digest(original_backup) == ORIGINAL
    pending_path = HERE / 'complexity_disclosure_pending_qa.json'
    qa_path = HERE / 'complexity_disclosure_visual_qa.json'
    note = HERE / '复杂度口径核查与修订说明.md'
    pending, qa = read(pending_path), read(qa_path)
    assert pending['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['page_count'] == 10
    assert qa['inspected_new_pages'] == [8, 9, 10]
    assert qa['pages_pixel_identical_to_previously_inspected_render'] == list(range(1, 8))
    assert qa['pdf_sha256'] == digest(HERE / 'rendered_word_approved/manuscript.pdf')
    for name, expected in qa['page_png_sha256'].items():
        assert digest(HERE / 'rendered_word_approved' / name) == expected
    assert digest(HERE / 'get_info_current_tree.py.txt') == pending['evidence']['local_get_info_sha256']
    assert digest(HERE / 'get_info_migrated_archive.py.txt') == pending['evidence']['migrated_get_info_sha256']
    assert digest(REC / 'source_snapshot/archived_benchmark/零训练数据汇总.md') == pending['evidence']['archived_summary_sha256']
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert b.testzip() is None and a.namelist() == b.namelist()
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']
        assert a.read('word/comments.xml') == b.read('word/comments.xml')
    previous = INDEX / '主要结论证据索引_第二十六轮.json'
    prev_index = read(previous)
    assert prev_index['revision'] == 26 and prev_index['manuscript']['sha256'] == OLD
    for v in prev_index.values():
        if isinstance(v, dict) and set(v) >= {'path', 'sha256'}:
            assert digest(Path(v['path'])) == v['sha256']
    prev_catalog = REC / '第二十四轮会议稿论述复核_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十七轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十七轮.txt'
    locations = HERE / 'manuscript_version_locations.json'
    receipt = HERE / 'complexity_disclosure_receipt.json'
    assert not any(p.exists() for p in (index_json, index_txt, locations, receipt))
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt']
    prior = [p.read_text(encoding='utf-8-sig') for p in paths]
    assert prior[3].count('当前稿件SHA256：' + OLD) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十七轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
        assert digest(saved) == digest(path)
    temp = CURRENT.with_name(CURRENT.name + '.rev27.tmp')
    assert not temp.exists()
    shutil.copy2(CANDIDATE, temp)
    assert digest(temp) == NEW
    os.replace(temp, CURRENT)
    assert digest(CURRENT) == NEW and digest(original) == digest(original_backup) == ORIGINAL
    catalog = {'revision': 27, 'previous_catalog': ref(prev_catalog),
               'versions': [{'revision': 26, 'sha256': OLD, 'retained_file': ref(BACKUP)},
                            {'revision': 27, 'sha256': NEW, 'retained_file': ref(CURRENT)}],
               'note': 'Revisions 25 and 26 left the revision-24 manuscript unchanged. Earlier indexes record the then-current path; resolve that manuscript hash to the retained backup here.'}
    write_new(locations, catalog)
    index = {'revision': 27, 'status': 'COMPLEXITY_PROFILING_LIMIT_DISCLOSED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(locations), 'previous_manuscript': ref(BACKUP),
             'patch_before_after': ref(pending_path), 'explanation': ref(note), 'visual_qa': ref(qa_path),
             'finding': 'The manuscript now states that the archived complexity summary names get_info.py but does not record the profiler version or operator coverage, and treats GFLOPs as tool-dependent estimates.',
             'limits': 'The available summary does not bind exact executed script bytes or profiler version. The different current and migrated scripts do not establish which profiler produced a historical number or which operations were omitted.',
             'models_run': 0, 'forwards': 0, 'profiler_runs': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
             'submission_ready': False, 'remaining': 'The paired BASE/FULL comparison remains incomplete; any new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十七轮\n当前稿件SHA256：' + NEW + '\n'
        '第4.4节说明复杂度汇总未记录分析器版本及算子覆盖，GFLOPs为工具相关估计；原表数值未改。\n'
        '现存新旧分析脚本不同，但无法据此认定历史数字使用的具体分析器或漏计算子。详见：' + str(note) + '\n'
        '机器可读索引：' + str(index_json) + '\n无模型、分析器、评价或训练；公平 BASE/FULL 仍不完整，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新稿件入口：2026-11-05，第二十七轮复杂度口径核查】\n'
        '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
        '第4.4节补充历史复杂度汇总的分析器版本与算子覆盖未记录，GFLOPs为工具相关估计；原表数值未改。已备份上一版并检查10页。\n'
        '详见修订记录_20261103/第二十七轮复杂度口径核查_20261105/复杂度口径核查与修订说明.md；最新证据索引为主要结论证据索引_第二十七轮.txt及JSON。\n'
        '无模型、分析器、评价或训练。公平主对照仍不完整，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    addition = ('\n第二十七轮：复杂度口径核查与说明（2026-11-05）\n'
        '历史汇总指出本地旧树get_info.py统计24.86/29.63 GFLOPs，但未记录当时分析器版本和算子覆盖；现存本地脚本调用calflops，迁移档案脚本可回退thop，但未绑定历史具体运行，不能据此断言漏计算子或解释两树差异。第4.4节补充GFLOPs为工具相关估计，表格数值及其他结果未改。\n'
        '第二十六轮备份SHA256：' + OLD + '；当前修订稿SHA256：' + NEW + '。6表、7图、6式及5组批注锚点保留；10页检查，其中前7页与此前逐页检查版像素文件一致，后三页重新查看。无模型、分析器、评价或训练；公平BASE/FULL仍不完整，尚不能投稿。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + addition)
    lead = ('最新会议稿修订（2026-11-05）：第4.4节说明归档复杂度缺少分析器版本与算子覆盖记录，GFLOPs为工具相关估计；原表数值未改。公平主对照仍不完整，尚不能投稿。详见修订记录_20261103/第二十七轮复杂度口径核查_20261105。\n')
    replace_text(paths[3], lead + prior[3].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW))
    write_new(receipt, {'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'revision': 27,
                        'manuscript': ref(CURRENT), 'previous_manuscript': ref(BACKUP),
                        'evidence_index': ref(index_json), 'visual_qa': ref(qa_path), 'original_preserved': True})
    print(json.dumps({'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'manuscript_sha256': digest(CURRENT),
                      'index_sha256': digest(index_json), 'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
