"""Promote the visually reviewed disclosure and record evidence/experiment gates."""
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
INDEX = REC/'主要结论证据索引'
REMOTE = PAPER/'远程公平对照_20261103'
CURRENT = PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE/'MPSMA-DEIM_conference_EN_revised_第十六轮备份.docx'
CANDIDATE = HERE/'MPSMA-DEIM_conference_EN_revised_第十七轮候选.docx'
OLD = '617ffd2024d42affb7f6b66f88cdca67204a0de3fef97946ffebf0c90f896984'
NEW = '888f5a17e991885b8581b16684e0a1d28ce06ab5ecd3f7176d2acd22e2551420'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'

def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def ref(path):
    return {'path': str(path), 'sha256': digest(path)}

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_text(path, value):
    temp = path.with_name(path.name + '.rev17.tmp')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)

def write_new_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

def main():
    original = PAPER/'稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER/'稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(CURRENT) == digest(BACKUP) == OLD and digest(CANDIDATE) == NEW
    assert digest(original) == digest(original_backup) == ORIGINAL
    patch = read(HERE/'seventeenth_revision_pending_qa.json')
    qa_path = HERE/'seventeenth_revision_visual_qa.json'
    qa = read(qa_path)
    assert patch['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert qa['pdf_sha256'] == digest(HERE/'rendered_word_elevated/manuscript.pdf')
    data_path = HERE/'train_val_byte_overlap_audit.json'
    label_path = HERE/'duplicate_annotation_audit.json'
    provenance_path = HERE/'duplicate_provenance_verification.json'
    explanation = HERE/'数据文件重复与标注差异说明.md'
    data, labels, provenance = read(data_path), read(label_path), read(provenance_path)
    assert provenance['source_audit_sha256'] == digest(data_path) == patch['data_audit_sha256']
    assert provenance['pair_annotation_audit_sha256'] == digest(label_path)
    assert digest(provenance_path) == patch['duplicate_provenance_sha256']
    assert data['status'] == 'HASHED_CURRENT_FILES' and len(labels['groups']) == 3
    assert not data['cross_split_identical_jpeg_byte_records']
    assert len([g for g in labels['groups'] if g['split'] == 'train']) == 2
    assert len([g for g in labels['groups'] if g['split'] == 'validation']) == 1
    for receipt in provenance['cpu_receipts']:
        assert digest(Path(receipt['path'])) == receipt['sha256']
    assert len(provenance['cpu_receipts']) == 3
    with ZipFile(CURRENT) as a, ZipFile(CANDIDATE) as b:
        assert b.testzip() is None and a.namelist() == b.namelist()
        assert [n for n in a.namelist() if a.read(n) != b.read(n)] == ['word/document.xml']

    index_json = INDEX/'主要结论证据索引_第十七轮.json'
    index_txt = INDEX/'主要结论证据索引_第十七轮.txt'
    location = HERE/'manuscript_version_locations.json'
    final_receipt = HERE/'seventeenth_revision_patch.json'
    assert not any(x.exists() for x in [index_json, index_txt, location, final_receipt])
    text_paths = [PAPER/'当前修订入口.txt', REC/'修订说明与未解决问题.txt',
                  REC/'目标验收与阻塞记录.txt', REMOTE/'结果并入与验收清单.txt',
                  REMOTE/'公平主对照重做提案_20261105.md']
    original_texts = [path.read_text(encoding='utf-8-sig') for path in text_paths]
    assert original_texts[3].count('当前稿件SHA256：' + OLD) == 1
    assert original_texts[4].startswith('# 公平主对照重做提案')
    for i, path in enumerate(text_paths):
        saved = HERE/f'更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)

    temp = CURRENT.with_name(CURRENT.name + '.rev17.tmp')
    assert not temp.exists()
    shutil.copy2(CANDIDATE, temp)
    assert digest(temp) == NEW
    os.replace(temp, CURRENT)
    assert digest(CURRENT) == NEW and digest(original) == digest(original_backup) == ORIGINAL

    previous = [INDEX/'主要结论证据索引_第十六轮.json', INDEX/'主要结论证据索引_第十六轮.txt']
    catalog = {'revision': 17, 'previous_catalog': ref(REC/'第十五轮相关工作补核_20261105/manuscript_version_locations.json'),
               'versions': [{'revision': 16, 'sha256': OLD, 'retained_file': ref(BACKUP)},
                            {'revision': 17, 'sha256': NEW, 'retained_file': ref(CURRENT)}],
               'note': 'Revision 16 did not edit the revision-15 manuscript. Earlier indices record the path current at their time; use SHA and the version catalog to locate retained bytes.'}
    write_new_json(location, catalog)
    record = {'revision': 17, 'status': 'CURRENT_DATA_DUPLICATE_DISCLOSURE',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
              'previous_indices': [ref(p) for p in previous], 'version_locations': ref(location),
              'data_audit': ref(data_path), 'annotation_audit': ref(label_path),
              'provenance_audit': ref(provenance_path), 'explanation': ref(explanation),
              'visual_qa': ref(qa_path), 'previous_manuscript': ref(BACKUP),
              'finding': 'Two byte-identical JPEG pairs in the current training set and one in the current validation set have differing retained box annotations; the validation pair is bound to all three CPU reevaluations.',
              'limits': 'No official image archive or historical data identity was authenticated. Exact cross-split byte matches were absent in the checked current files; near-duplicate scenes remain possible. AP impact was not measured.',
              'model_instantiations': 0, 'model_forwards': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
              'submission_ready': False,
              'remaining': 'Investigate the source of the suspect image/annotation pairs and complete a matched BASE/FULL comparison after explicit authorization.'}
    write_new_json(index_json, record)
    index_txt.write_text('主要结论证据索引 第十七轮\n当前稿件SHA256：' + NEW + '\n'
        '现存训练集两对、验证集一对JPEG字节相同但框标注不同；验证对已包含在三次CPU重评输入中，AP影响未知。\n'
        '已检查当前全部7019张JPEG，训练/验证之间无字节相同文件；不能排除近似帧，也未认证历史数据或官方压缩包。\n'
        '详情：' + str(explanation) + '\n证据继承第十六轮；机器可读索引：' + str(index_json) + '\n'
        '当前数据来源和公平BASE/FULL对照仍未闭合，尚不能投稿。\n', encoding='utf-8')

    header = ('【最新稿件入口：2026-11-05，第十七轮数据文件核查】\n'
              '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
              '第4.1节披露现存训练集两对、验证集一对JPEG字节相同但框标注不同；验证对包含在三次CPU重评中，AP影响未知。第十六轮稿件已备份，10页逐页检查。\n'
              '详情见修订记录_20261103/第十七轮证据表述复核_20261105/数据文件重复与标注差异说明.md；最新证据索引为主要结论证据索引_第十七轮.txt及JSON。\n'
              '无新增模型、评价或训练。数据来源和公平BASE/FULL对照仍待解决，重做方案待明确授权，尚不能投稿。以下较早状态均为历史。\n\n')
    write_text(text_paths[0], header + original_texts[0])
    note = ('\n第十七轮：现存数据重复与标注差异（2026-11-05）\n'
            '只读重哈希6471个训练和548个验证JPEG并核对旧清单与标注文件名，跨集合未发现完全相同字节；训练集内部两对、验证集内部一对字节相同且框标注不同。验证对框数41/40，三次CPU重评回执绑定此图像清单及验证标注。未取得独立官方图像归档，不确定原因或AP影响，不能排除近似帧。\n'
            '第4.1节仅增加这一数据限制；第十六轮备份SHA256：' + OLD + '；当前稿件SHA256：' + NEW + '。6表、7图、6式及5组批注锚点保留，10页逐页检查。无新增模型、评价或训练；应在公平对照前核查数据来源，仍不能投稿。\n')
    write_text(text_paths[1], original_texts[1] + note)
    write_text(text_paths[2], original_texts[2] + note)
    gate_header = ('最新数据核查（2026-11-05）：现存训练集有两对、验证集有一对字节相同但标注不同的JPEG；验证对已含于三次CPU重评。来源及AP影响未知。公平对照前应核对独立来源，任何数据修正均须固定新清单、重审预算和评价协议；新训练仍须明确授权。详见修订记录_20261103/第十七轮证据表述复核_20261105/数据文件重复与标注差异说明.md。\n')
    write_text(text_paths[3], gate_header + original_texts[3].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW))
    first, rest = original_texts[4].split('\n', 1)
    proposal_gate = ('\n\n## 数据来源待核查（2026-11-05）\n\n'
        '现存训练集两对、验证集一对JPEG字节相同但框标注不同；验证对含于既有三次CPU重评。原因及AP影响未知。启动新训练前应先检查可验证的独立原始数据来源并记录处理决定，保留原始数据快照，不得悄悄删改记录。原提案的6471/548记录和更新次数以沿用现存冻结数据为条件；若更正数据或改变协议，须重新固定样本、预算和端点，再取得明确授权。详见修订记录_20261103/第十七轮证据表述复核_20261105/数据文件重复与标注差异说明.md。\n')
    write_text(text_paths[4], first + proposal_gate + '\n' + rest)
    write_new_json(final_receipt, {'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA',
        'revision': 17, 'manuscript': ref(CURRENT), 'previous_manuscript': ref(BACKUP),
        'evidence_index': ref(index_json), 'visual_qa': ref(qa_path), 'original_preserved': True})
    print(json.dumps({'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'manuscript_sha256': digest(CURRENT),
                      'index_sha256': digest(index_json), 'submission_ready': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
