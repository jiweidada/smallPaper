"""Register bounded FULL recovery evidence without changing experiment files."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDEX = REC / '主要结论证据索引'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CURRENT_SHA = '9060aee70bae6804fbf61a4adbc76ce34a2d4be7e44d17dfa2500877300adbab'
ORIGINAL_SHA = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'


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
    temp = path.with_name(path.name + '.rev23.tmp')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    original = PAPER / '稿件/MPSMA-DEIM_conference_EN.docx'
    backup = PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert digest(CURRENT) == CURRENT_SHA and digest(original) == digest(backup) == ORIGINAL_SHA
    audit_path = HERE / 'full_local_backup_audit.json'
    note = HERE / 'FULL本地备份与恢复线索.md'
    audit = read(audit_path)
    assert audit['status'] == 'BOUNDED_LOCAL_FULL_BACKUP_SEARCH_COMPLETE'
    assert audit['latest_confirmed_epoch_zero_based'] == 127 and audit['latest_snapshot_is_metadata_only']
    assert len(audit['early_epoch0_files']) == 5 and len(audit['archives']) == 2
    assert audit['external_search']['files_searched'] == 68278 and audit['external_search']['files_with_matches'] == 0
    assert audit['external_search']['exit_code'] == 1
    assert all(audit[k] == 0 for k in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    for item in [audit['early_epoch0_receipt'], audit['latest_snapshot'],
                 audit['external_search']['stats'], audit['external_search']['stderr'],
                 *audit['early_epoch0_files'].values(), *[x['archive'] for x in audit['archives']]]:
        assert digest(Path(item['path'])) == item['sha256']
    assert all(not archive['full_training_output_name_matches'] for archive in audit['archives'])
    snapshot = read(Path(audit['latest_snapshot']['path']))
    observed = json.loads(snapshot['stdout'])['arms']['FULL']
    assert observed['complete_epoch_rows'] == 128 and observed['latest_complete_epoch']['epoch'] == 127
    assert observed['latest_complete_epoch']['ema_tensor_sha256'] == audit['latest_confirmed_ema_tensor_sha256']
    previous = INDEX / '主要结论证据索引_第二十二轮.json'
    catalog = REC / '第二十一轮训练超参核查_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十三轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十三轮.txt'
    receipt = HERE / 'full_backup_search_receipt.json'
    assert not any(path.exists() for path in [index_json, index_txt, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt', REMOTE / '公平主对照重做提案_20261105.md',
             REMOTE / '实验中断与证据保全_20261105.txt']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + CURRENT_SHA) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十三轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    index = {'revision': 23, 'status': 'BOUNDED_FULL_BACKUP_SEARCH_DOCUMENTED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(catalog),
             'local_backup_audit': ref(audit_path), 'recovery_note': ref(note),
             'finding': 'The five locally saved FULL epoch-0 files match their receipt. The last snapshot describes epoch 127 but does not contain its weights. Two preflight transfer archives do not contain the searched FULL training-output names; bounded external text search found no matching reference.',
             'limits': 'Archive checks examine member names only. Other formats, larger files, excluded directories, archives and remote backups remain outside the external text search. FULL recovery alone would not complete the interrupted BASE/FULL pair.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
             'submission_ready': False, 'remaining': 'Original server or external backup location remains unknown; the matched pair has not completed.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十三轮\n当前稿件未改，SHA256：' + CURRENT_SHA + '\n'
        'FULL首轮五份文件与回执SHA匹配；断线前索引127只有本地状态快照及EMA指纹。两个准备压缩包未见所查后期训练产物名称，限定文本范围未找到相关引用。\n'
        '未搜索全部压缩包、二进制或远程备份，不能证明原实例或外部备份不存在；即使找回FULL，BASE仍未完成320轮。\n详情：' + str(note) + '\n机器可读索引：' + str(index_json) + '\n本轮无模型、推理、评价或训练；尚不能投稿。\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第二十三轮中断产物检索】\n'
        '当前稿件未改：' + str(CURRENT) + '\nSHA256：' + CURRENT_SHA + '\n'
        'FULL首轮五份已保存文件与回执哈希一致；最后完整轮索引127仅有本地快照。两个准备压缩包和限定文本范围未检索到可用后期产物；原实例或外部备份仍未知。\n'
        '详情见修订记录_20261103/第二十三轮中断产物检索_20261105/FULL本地备份与恢复线索.md；最新证据索引为主要结论证据索引_第二十三轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。BASE/FULL均未形成预定配对终点，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    addition = ('\n第二十三轮：中断FULL本地备份检索（2026-11-05）\n'
        'FULL首轮检查点、预测、评价数组、指标及回执5项与下载记录逐项SHA一致；断线前索引127的本地快照只包含元数据及EMA指纹。两个准备压缩包按成员名检查未见所查后期产物，其他三个目录的限定文本搜索未找到协议、EMA指纹或实验目录名引用；未搜索全部二进制、压缩包或远程备份。\n'
        '论文未改，SHA256：' + CURRENT_SHA + '。无模型、推理、评价或训练；找回FULL也不能自动补齐未完成的BASE，尚不能投稿。详情见第二十三轮中断产物检索_20261105。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + addition)
    lead = ('最新FULL本地备份检索（2026-11-05）：首轮5文件已重新核验，索引127仅存状态及EMA指纹；准备压缩包和限定文本范围未见可用后期产物。原实例或外部备份仍未知，即使找回FULL也不补齐未完成的BASE。详见修订记录_20261103/第二十三轮中断产物检索_20261105。\n')
    replace_text(paths[3], lead + prior[3])
    proposal = ('\n## 中断 FULL 本地备份补查（未改变授权状态）\n\n'
        'FULL 首轮的五份本地审计文件已重新核对，断线前索引 127 仅有状态快照及 EMA 张量指纹；检查的两个准备压缩包和限定文本范围未找到后期训练产物。原第二台实例或外部备份是否存在仍未知。若找回原始运行目录，先只读核对完整性和绑定；即使 FULL 完成，BASE 现有 225 轮也不足以形成预定 320 轮双组对照。任何续跑或重训须另行说明并获得授权。详见修订记录_20261103/第二十三轮中断产物检索_20261105/FULL本地备份与恢复线索.md。\n')
    replace_text(paths[4], prior[4] + proposal)
    replace_text(paths[5], prior[5] + '\n2026-11-05 FULL 本地备份补查：首轮五份文件SHA与回执一致；断线前索引127仅有状态快照及EMA张量指纹，检查的准备压缩包和限定文本范围未找到后期产物。原实例或外部备份仍未知；恢复线索及完整范围见修订记录_20261103/第二十三轮中断产物检索_20261105。未执行新实验或改动原文件。\n')
    write_new(receipt, {'status': 'RECOVERY_EVIDENCE_REGISTERED_MANUSCRIPT_UNCHANGED', 'revision': 23,
                        'manuscript': ref(CURRENT), 'index': ref(index_json), 'note': ref(note),
                        'original_preserved': True, 'submission_ready': False})
    print(json.dumps({'status': 'RECOVERY_EVIDENCE_REGISTERED_MANUSCRIPT_UNCHANGED',
                      'manuscript_sha256': digest(CURRENT), 'index_sha256': digest(index_json),
                      'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
