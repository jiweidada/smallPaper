"""Correct the location of retained descriptive statistics and refresh hashes."""
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
INDEX = REC / '主要结论证据索引/主要结论证据索引_第二十四轮.json'
NOTE = HERE / '证据限制段落修订说明.md'
RECEIPT = HERE / 'twentyfourth_revision_patch.json'
AUDIT = REC / 'run_audit.json'
TIMELINE = REC / '第二十轮检查点时序追溯_20261105/n2_conditional_timeline.json'
CORRECTION = HERE / 'statistics_locator_correction.json'
OLD = '这些细节和原有限制仍保存在第十九、二十轮证据文件及上一版稿件中；'
NEW = 's 组统计见修订记录根目录的 run_audit.json，N2 条件性推演见第二十轮证据文件，上一版稿件保留原文；'
LOG_OLD = '原后期统计及N2条件性推演保留在第十九、二十轮证据及上一版备份。'
LOG_NEW = '原后期统计见run_audit.json，N2条件性推演见第二十轮证据，上一版备份保留原文。'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.locator.tmp')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    assert not CORRECTION.exists()
    index = json.loads(INDEX.read_text(encoding='utf-8-sig'))
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8-sig'))
    assert index['explanation']['sha256'] == digest(NOTE)
    assert receipt['evidence_index']['sha256'] == digest(INDEX)
    assert 'last20_AP_mean' in AUDIT.read_text(encoding='utf-8-sig')
    assert json.loads(TIMELINE.read_text(encoding='utf-8-sig'))['stage1_saved_epoch'] == 291
    paths = [NOTE, INDEX, RECEIPT, REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt']
    originals = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert originals[0].count(OLD) == 1
    assert all(originals[i].count(LOG_OLD) == 1 for i in [3, 4])
    old_index_sha = digest(INDEX)
    for i, path in enumerate(paths):
        saved = HERE / f'统计来源更正前_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    atomic(NOTE, originals[0].replace(OLD, NEW))
    for i in [3, 4]:
        atomic(paths[i], originals[i].replace(LOG_OLD, LOG_NEW))
    index['explanation']['sha256'] = digest(NOTE)
    atomic(INDEX, json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    receipt['evidence_index']['sha256'] = digest(INDEX)
    atomic(RECEIPT, json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    correction = {'status': 'STATISTICS_LOCATOR_CORRECTED', 'old_index_sha256': old_index_sha,
                  'new_index_sha256': digest(INDEX), 'updated_note_sha256': digest(NOTE),
                  'statistics_source': {'path': str(AUDIT), 'sha256': digest(AUDIT)},
                  'conditional_timeline_source': {'path': str(TIMELINE), 'sha256': digest(TIMELINE)},
                  'manuscript_unchanged': True}
    with CORRECTION.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(correction, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(correction, ensure_ascii=False))


if __name__ == '__main__':
    main()
