"""Index pinned third-party metadata while preserving the revision-17 index."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
INDEX = REC/'主要结论证据索引'
DOC = PAPER/'稿件/MPSMA-DEIM_conference_EN_revised.docx'
DOC_SHA = '888f5a17e991885b8581b16684e0a1d28ce06ab5ecd3f7176d2acd22e2551420'

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def ref(path):
    return {'path': str(path), 'sha256': digest(path)}

def main():
    pinned_path = HERE/'public_mirror_pinned_evidence.json'
    initial_path = HERE/'public_mirror_pointer_audit.json'
    note_path = HERE/'公开镜像哈希补核.md'
    pinned = json.loads(pinned_path.read_text(encoding='utf-8'))
    assert pinned['status'] == 'THIRD_PARTY_MIRROR_POINTERS_PINNED_AND_MATCH_LOCAL'
    assert pinned['initial_audit_sha256'] == digest(initial_path)
    assert len(pinned['records']) == 7 and all(r['declared_image_sha256'] == r['local_image_sha256'] for r in pinned['records'])
    assert digest(DOC) == DOC_SHA
    new_json = INDEX/'主要结论证据索引_第十七轮镜像补充.json'
    new_txt = INDEX/'主要结论证据索引_第十七轮镜像补充.txt'
    assert not new_json.exists() and not new_txt.exists()
    text_paths = [PAPER/'当前修订入口.txt', REC/'修订说明与未解决问题.txt',
                  REC/'目标验收与阻塞记录.txt', PAPER/'远程公平对照_20261103/公平主对照重做提案_20261105.md']
    old_text = [p.read_text(encoding='utf-8-sig') for p in text_paths]
    assert old_text[3].count('原因及AP影响未知。') == 1
    for i, path in enumerate(text_paths):
        backup = HERE/f'镜像补核前文本_{i}.txt'
        assert not backup.exists()
        shutil.copy2(path, backup)
    previous = INDEX/'主要结论证据索引_第十七轮.json'
    index = {'status': 'REVISION_17_THIRD_PARTY_MIRROR_SUPPLEMENT',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(DOC),
             'previous_index': ref(previous), 'pinned_mirror_evidence': ref(pinned_path),
             'initial_mirror_evidence': ref(initial_path), 'explanation': ref(note_path),
             'mirror_revision': pinned['revision'], 'official_archive_authenticated': False,
             'findings': 'All six duplicate images and one control have matching SHA256 declarations in a pinned third-party mirror. No official archive or mirror annotations were checked.',
             'model_instantiations': 0, 'model_forwards': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
             'submission_ready': False}
    with new_json.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    new_txt.write_text('主要结论证据索引 第十七轮第三方镜像补充\n当前稿件SHA256：' + DOC_SHA + '\n'
        '固定Voxel51公开镜像版本' + pinned['revision'] + '，六个重复文件及一个参照的LFS指针声明SHA与本地一致；没有下载图像或比对镜像标注。\n'
        '该镜像不是官方原始归档，不能认证历史运行数据或判断AP影响；证据继承第十七轮。\n'
        '详情：' + str(note_path) + '\n机器可读索引：' + str(new_json) + '\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第十七轮第三方镜像补核】\n'
              '当前稿件未变：' + str(DOC) + '\nSHA256：' + DOC_SHA + '\n'
              '六个重复JPEG和一个参照的哈希与固定版本第三方镜像的LFS指针一致；镜像非官方归档，未比对镜像标注，也未确定AP影响。\n'
              '详情见修订记录_20261103/第十七轮证据表述复核_20261105/公开镜像哈希补核.md；最新证据索引为主要结论证据索引_第十七轮镜像补充.txt及JSON。\n'
              '未下载图片或运行新实验。数据来源和公平主对照仍需处理，尚不能投稿。\n\n')
    text_paths[0].write_text(header + old_text[0], encoding='utf-8')
    record = ('\n2026-11-05 第十七轮镜像补核：固定Voxel51第三方镜像版本' + pinned['revision'] +
              '，六个重复文件与一个参照的公开LFS指针声明SHA均与本地一致；未下载图像、比对镜像标注或核验官方原始归档。重复不能直接归因本地复制，AP影响与历史数据身份仍未知。稿件未变，SHA256：' + DOC_SHA + '。详见公开镜像哈希补核.md。无模型、评价或训练。\n')
    for i in [1, 2]:
        text_paths[i].write_text(old_text[i] + record, encoding='utf-8')
    insert = ('原因及AP影响未知。固定版本的第三方Voxel51镜像中六个文件的图像哈希与本地一致，但该镜像不是官方归档，本次也未比对其标注。')
    text_paths[3].write_text(old_text[3].replace('原因及AP影响未知。', insert, 1), encoding='utf-8')
    assert digest(DOC) == DOC_SHA
    print(json.dumps({'status': index['status'], 'supplement_sha256': digest(new_json), 'manuscript_sha256': DOC_SHA}, ensure_ascii=False))

if __name__ == '__main__':
    main()
