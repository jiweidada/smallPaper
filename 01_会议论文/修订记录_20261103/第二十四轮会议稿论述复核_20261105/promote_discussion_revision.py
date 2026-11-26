"""Promote the checked, concise evidence discussion and version references."""
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
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十三轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十四轮候选.docx'
OLD = '9060aee70bae6804fbf61a4adbc76ce34a2d4be7e44d17dfa2500877300adbab'
NEW = '185cbe8ffca81f86b10e8c12fab1e68eda90b722629617e870068e14a60f8993'
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
    temp = path.with_name(path.name + '.rev24.tmp')
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
    pending_path = HERE / 'twentyfourth_revision_pending_qa.json'
    qa_path = HERE / 'twentyfourth_revision_visual_qa.json'
    note = HERE / '证据限制段落修订说明.md'
    pending, qa = read(pending_path), read(qa_path)
    assert pending['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert pending['approximate_words_before'] == 250 and pending['approximate_words_after'] == 162
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert qa['pdf_sha256'] == digest(HERE / 'rendered_word/manuscript.pdf')
    for name, expected in qa['page_png_sha256'].items():
        assert digest(HERE / 'rendered_word' / name) == expected
    for name, expected in pending['source_evidence'].items():
        assert digest(Path(name)) == expected
    arrays = read(REC / '第十九轮历史评价追溯_20261105/independent_saved_array_verification.json')
    fields = read(REC / '第十九轮历史评价追溯_20261105/saved_evaluation_binding_fields.json')
    assert arrays['arrays_checked'] == 48 and arrays['maximum_abs_difference'] < 1e-12
    assert [(item['run'], item['epoch']) for item in arrays['selected_best_log_arrays']] == [('N2', 300)]
    assert not fields['files_with_checkpoint_or_hash_named_fields']
    with ZipFile(CURRENT) as current_zip, ZipFile(CANDIDATE) as candidate_zip:
        assert candidate_zip.testzip() is None and current_zip.namelist() == candidate_zip.namelist()
        assert [name for name in current_zip.namelist() if current_zip.read(name) != candidate_zip.read(name)] == ['word/document.xml']
        assert current_zip.read('word/comments.xml') == candidate_zip.read('word/comments.xml')
    previous = INDEX / '主要结论证据索引_第二十三轮.json'
    previous_catalog = REC / '第二十一轮训练超参核查_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十四轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十四轮.txt'
    locations = HERE / 'manuscript_version_locations.json'
    receipt = HERE / 'twentyfourth_revision_patch.json'
    assert not any(path.exists() for path in [index_json, index_txt, locations, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + OLD) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十四轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    temp = CURRENT.with_name(CURRENT.name + '.rev24.tmp')
    assert not temp.exists()
    shutil.copy2(CANDIDATE, temp)
    assert digest(temp) == NEW
    os.replace(temp, CURRENT)
    assert digest(CURRENT) == NEW and digest(original) == digest(original_backup) == ORIGINAL
    catalog = {'revision': 24, 'previous_catalog': ref(previous_catalog),
               'versions': [{'revision': 23, 'sha256': OLD, 'retained_file': ref(BACKUP)},
                            {'revision': 24, 'sha256': NEW, 'retained_file': ref(CURRENT)}],
               'note': 'Revisions 22 and 23 did not change the revision-21 manuscript. Earlier current-path entries described the then-current version.'}
    write_new(locations, catalog)
    index = {'revision': 24, 'status': 'EVIDENCE_LIMITS_CONDENSED_WITH_TRACE_RETAINED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(locations), 'previous_manuscript': ref(BACKUP),
             'patch_before_after': ref(pending_path), 'explanation': ref(note), 'visual_qa': ref(qa_path),
             'finding': 'Section 4.6 is shorter and retains single-run selection limits, the N2-only best-row array qualification, missing explicit checkpoint hashes, source uncertainty, CPU/historical separation and need for controlled comparisons.',
             'limits': 'The s-run late-epoch descriptive statistics and N2 conditional arithmetic remain in earlier evidence records rather than the article. No claim of historical weight identity or causal improvement is added.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
             'submission_ready': False, 'remaining': 'The paired BASE/FULL comparison is incomplete and new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十四轮\n当前稿件SHA256：' + NEW + '\n'
        '第4.6节压缩为证据限制讨论，保留验证选择、单seed、48份数组、仅N2最佳行有对应数组、缺少显式权重哈希、历史代码未认证和公平对照缺口。\n'
        '原始后期统计及N2条件性计数仍在此前证据记录及第二十三轮稿件备份中；没有改动实验数值。\n'
        '详情：' + str(note) + '\n机器可读索引：' + str(index_json) + '\n本轮无模型、推理、评价或训练；尚不能投稿。\n', encoding='utf-8')
    header = ('【最新稿件入口：2026-11-05，第二十四轮会议稿论述复核】\n'
        '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
        '第4.6节压缩证据限制叙述，保留单次验证选择、历史权重归属和公平对照缺口；细节留在此前证据文件。第二十三轮稿件已备份，10页逐页检查。\n'
        '详情见修订记录_20261103/第二十四轮会议稿论述复核_20261105/证据限制段落修订说明.md；最新证据索引为主要结论证据索引_第二十四轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。公平主对照仍不完整，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    note_text = ('\n第二十四轮：会议稿证据限制论述（2026-11-05）\n'
        '第4.6节约250词压缩至162词，保留最佳行验证选择、单次运行、48份数组与N2最佳行范围、无显式权重哈希、历史源码未认证、CPU/历史区分及公平对照需求。原后期统计及N2条件性推演保留在第十九、二十轮证据及上一版备份。\n'
        '第二十三轮备份SHA256：' + OLD + '；当前稿件SHA256：' + NEW + '。6表、7图、6式和5组批注锚点保留，10页逐页检查；摘要、实验数值和引用未变。无新模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + note_text)
    lead = ('最新会议稿论述修订（2026-11-05）：第4.6节压缩追溯细节，保留验证选择、无显式权重哈希、历史源码和公平对照等限制；原细节仍在证据记录。未新增实验，仍不能投稿。详见修订记录_20261103/第二十四轮会议稿论述复核_20261105。\n')
    replace_text(paths[3], lead + prior[3].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW))
    write_new(receipt, {'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'revision': 24,
                        'manuscript': ref(CURRENT), 'previous_manuscript': ref(BACKUP),
                        'evidence_index': ref(index_json), 'visual_qa': ref(qa_path), 'original_preserved': True})
    print(json.dumps({'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'manuscript_sha256': digest(CURRENT),
                      'index_sha256': digest(index_json), 'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
