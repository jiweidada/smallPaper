"""Index the unsuccessful official archive request without changing any prior evidence."""
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
    receipt = HERE/'official_val_archive_verification.json'
    data = json.loads(receipt.read_text(encoding='utf-8'))
    assert data['status'] == 'OFFICIAL_DOWNLOAD_NOT_A_ZIP' and data['confirmation_followed']
    assert not (HERE/'official_validation_release/VisDrone2019-DET-val.zip').exists()
    assert not (HERE/'official_validation_release/VisDrone2019-DET-val.zip.part').exists()
    assert digest(DOC) == DOC_SHA
    paths = [HERE/'official_val_link_head_probe.json', HERE/'official_val_confirmed_head_probe.json',
             receipt, HERE/'官方验证集归档访问说明.md']
    previous = INDEX/'主要结论证据索引_第十七轮镜像补充.json'
    output = INDEX/'主要结论证据索引_第十七轮官方源补充.json'
    output_text = INDEX/'主要结论证据索引_第十七轮官方源补充.txt'
    assert not output.exists() and not output_text.exists()
    text_paths = [PAPER/'当前修订入口.txt', REC/'修订说明与未解决问题.txt', REC/'目标验收与阻塞记录.txt']
    old = [p.read_text(encoding='utf-8-sig') for p in text_paths]
    for i, p in enumerate(text_paths):
        backup = HERE/f'官方源访问前文本_{i}.txt'
        assert not backup.exists()
        shutil.copy2(p, backup)
    index = {'status': 'OFFICIAL_ARCHIVE_DOWNLOAD_RETURNED_HTML_NOT_VERIFIED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(DOC),
             'previous_index': ref(previous), 'access_evidence': [ref(p) for p in paths],
             'scope': 'Official-repository-linked validation download returned HTML even after the public confirmation form; the reason is unknown.',
             'official_archive_acquired': False, 'images_downloaded': False,
             'model_instantiations': 0, 'model_forwards': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
             'submission_ready': False}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    output_text.write_text('主要结论证据索引 第十七轮官方源访问补充\n稿件SHA256：' + DOC_SHA + '\n'
        '官方仓库所列验证集链接与确认表单均返回HTML，未取得ZIP；原因未知，不能认证官方图像或标注。第三方镜像核验及其他证据继承前一索引。\n'
        '详情：' + str(paths[-1]) + '\n机器可读索引：' + str(output) + '\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，官方验证集链接访问】\n'
              '当前修订稿未变：' + str(DOC) + '\nSHA256：' + DOC_SHA + '\n'
              '官方仓库所列验证集下载链接及确认步骤返回HTML，未取得ZIP，原因未知；第三方镜像固定版本的六个哈希仍与本地一致，但不能替代官方归档。\n'
              '详情见修订记录_20261103/第十七轮证据表述复核_20261105/官方验证集归档访问说明.md；最新证据索引为主要结论证据索引_第十七轮官方源补充.txt及JSON。\n'
              '未下载图片或运行新实验。数据来源和公平主对照仍需处理，尚不能投稿。\n\n')
    text_paths[0].write_text(header + old[0], encoding='utf-8')
    note = ('\n2026-11-05 官方源访问补充：官方仓库列出的验证集Google Drive链接及其公开确认表单均返回HTML，未取得ZIP，原因未知。'
            '未下载图片、上传数据或执行模型/评价；不能据此判断官方归档内容。详见官方验证集归档访问说明.md；当前稿件SHA256：' + DOC_SHA + '。\n')
    for i in [1, 2]:
        text_paths[i].write_text(old[i] + note, encoding='utf-8')
    assert digest(DOC) == DOC_SHA
    print(json.dumps({'status': index['status'], 'index_sha256': digest(output)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
