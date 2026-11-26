"""Verify and register the bounded archive search without changing the DOCX."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
INDEX = REC / '主要结论证据索引'
REMOTE = PAPER / '远程公平对照_20261103'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CURRENT_SHA = '9060aee70bae6804fbf61a4adbc76ce34a2d4be7e44d17dfa2500877300adbab'
ORIGINAL_SHA = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
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
    temp = path.with_name(path.name + '.rev22.tmp')
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
    audit_path = HERE / 'archived_args_inventory.json'
    check_path = HERE / 'independent_inventory_verification.json'
    hidden_path = HERE / 'hidden_noignore_rescan.json'
    note = HERE / '扩展存档基线检索说明.md'
    audit, check, hidden = read(audit_path), read(check_path), read(hidden_path)
    assert audit['status'] == 'ACCESSIBLE_ARCHIVED_ARGS_INVENTORIED'
    assert check['status'] == 'PRUNED_RESCAN_AND_CHECKSUMS_MATCHED'
    assert hidden['status'] == 'HIDDEN_NOIGNORE_RESCAN_MATCHED'
    assert check['original_inventory_sha256'] == hidden['inventory_sha256'] == digest(audit_path)
    assert check['rg_exit_code'] == hidden['rg_exit_code'] == 0
    assert check['rg_error_line_count'] == hidden['rg_error_line_count'] == 0
    assert audit['rg_error_line_count'] == 112 and audit['files_returned'] == check['args_files_verified'] == hidden['files_found'] == 143
    assert audit['distinct_args_sha256'] == 76 and check['adjacent_logs_hashed'] == audit['with_adjacent_log'] == 135
    assert not audit['coarse_320_batch4_fp32_count'] and not check['adjacent_logs_with_all_0_319']
    assert all(audit[k] == 0 for k in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    count = Counter()
    for row in audit['records']:
        path = Path(row['args_path'])
        assert digest(path) == row['args_sha256']
        data = read(path)
        cfg = data.get('yaml_cfg', data)
        amp = data.get('use_amp', cfg.get('use_amp'))
        gamma = data.get('lr_gamma', cfg.get('lr_gamma'))
        width = cfg['HybridEncoder']['hidden_dim']
        assert amp is True and gamma == 0.5 and width == 256
        if 'use_amp' in data and 'use_amp' in cfg:
            assert data['use_amp'] == cfg['use_amp']
        if 'lr_gamma' in data and 'lr_gamma' in cfg:
            assert data['lr_gamma'] == cfg['lr_gamma']
        count[str(data.get('epoches', cfg.get('epoches')))] += 1
        if row['adjacent_log_present']:
            assert digest(Path(row['adjacent_log']['path'])) == row['adjacent_log']['sha256']
    assert dict(count) == {'132': 67, '12': 2, '96': 72, '24': 2}
    assert digest(Path(audit['reference_N4']['path'])) == audit['reference_N4']['sha256']
    previous = INDEX / '主要结论证据索引_第二十一轮.json'
    catalog = REC / '第二十一轮训练超参核查_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十二轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十二轮.txt'
    receipt = HERE / 'archive_search_receipt.json'
    assert not any(path.exists() for path in [index_json, index_txt, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt', REMOTE / '公平主对照重做提案_20261105.md']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + CURRENT_SHA) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十二轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    index = {'revision': 22, 'status': 'BOUNDED_ARCHIVE_SEARCH_DOCUMENTED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(catalog),
             'inventory': ref(audit_path), 'independent_verification': ref(check_path),
             'hidden_noignore_rescan': ref(hidden_path), 'explanation': ref(note),
             'finding': 'Across 143 accessible args files (76 distinct file hashes), configured epochs are 12, 24, 96 or 132; all declare AMP, gamma 0.5 and encoder width 256. No adjacent log covers all epochs 0–319.',
             'limits': 'Files are not independent runs. The search excludes dependency directories, archives, other filenames and remote systems; no general absence claim or data/source identity is established.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0, 'submission_ready': False,
             'remaining': 'The paired BASE/FULL main comparison remains incomplete; new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十二轮\n当前稿件未改，SHA256：' + CURRENT_SHA + '\n'
        '扩展扫描三个工作区目录中143份可读取的args.json（76个不同文件哈希）；配置轮数为12/24/96/132，均声明AMP开启、gamma 0.5、编码器宽256，135份有相邻日志。\n'
        '未在所查范围找到可作320轮同条件N4主对照的记录；压缩包、其他文件名、依赖目录及远程不在范围，不能推断其他位置不存在。\n'
        '详情：' + str(note) + '\n机器可读索引：' + str(index_json) + '\n本轮无模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第二十二轮扩展基线检索】\n'
        '当前稿件未改：' + str(CURRENT) + '\nSHA256：' + CURRENT_SHA + '\n'
        '扩展检索三个工作区目录内143份args.json；配置均为12/24/96/132轮，且AMP、gamma和编码器宽度与N4不同。文件可能为重复快照，未搜索压缩包或远程。\n'
        '详见修订记录_20261103/第二十二轮扩展基线检索_20261105/扩展存档基线检索说明.md；最新证据索引为主要结论证据索引_第二十二轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。公平主对照仍待授权与完成，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    addition = ('\n第二十二轮：扩展存档基线检索（2026-11-05）\n'
        '对04_实验记录、05_代码与数据、90_历史与工具中的args.json做限定检索；包含隐藏文件且不应用忽略规则的复扫返回143个配置文件、76个不同文件哈希，135份有相邻日志。所查配置轮数为12/24/96/132，均声明AMP开启、gamma 0.5、编码器宽256，不能充当N4同条件320轮n基线。档案文件、其他文件名、依赖目录和远程未覆盖，不能推广为全局不存在。\n'
        '论文未改，SHA256：' + CURRENT_SHA + '。无模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。详情见第二十二轮扩展基线检索_20261105。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + addition)
    lead = ('最新扩展基线检索（2026-11-05）：三个工作区目录所查143份args.json配置为12/24/96/132轮且条件与N4不同；文件可能是重复快照，压缩包和远程未搜索。未找到可认定的同条件320轮基线，不能替代公平主对照。详见修订记录_20261103/第二十二轮扩展基线检索_20261105。\n')
    replace_text(paths[3], lead + prior[3])
    proposal = ('\n## 扩展存档检索补充（未改变授权状态）\n\n'
        '在 `04_实验记录`、`05_代码与数据` 和 `90_历史与工具` 的限定范围中，143 份可读取的 `args.json` 均配置 12、24、96 或 132 轮，且声明 AMP 开启、gamma 0.5、编码器宽 256；包含重复快照。所查 135 份相邻日志没有完整零基 0–319。它们不能作为 N4 的同条件 320 轮 n 基线。压缩包、其他文件名、依赖目录和远程未搜索，不能据此宣称其他位置不存在。详情见修订记录_20261103/第二十二轮扩展基线检索_20261105/扩展存档基线检索说明.md。\n')
    replace_text(paths[4], prior[4] + proposal)
    write_new(receipt, {'status': 'EVIDENCE_REGISTERED_MANUSCRIPT_UNCHANGED', 'revision': 22,
                        'manuscript': ref(CURRENT), 'index': ref(index_json), 'note': ref(note),
                        'original_preserved': True, 'submission_ready': False})
    print(json.dumps({'status': 'EVIDENCE_REGISTERED_MANUSCRIPT_UNCHANGED', 'manuscript_sha256': digest(CURRENT),
                      'index_sha256': digest(index_json), 'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
