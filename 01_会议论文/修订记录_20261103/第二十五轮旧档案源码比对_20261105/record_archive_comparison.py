"""Register the source-archive comparison while preserving the manuscript."""
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
CURRENT_SHA = '185cbe8ffca81f86b10e8c12fab1e68eda90b722629617e870068e14a60f8993'
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
    temp = path.with_name(path.name + '.rev25.tmp')
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
    report_path = HERE / 'migrated_archive_comparison.json'
    proof_path = HERE / 'independent_archive_verification.json'
    explanation = HERE / '迁移档案与现存源码比对说明.md'
    report, proof = read(report_path), read(proof_path)
    assert report['status'] == 'MIGRATED_ARCHIVE_AND_CURRENT_SOURCE_COMPARED'
    assert proof['status'] == 'ARCHIVE_MEMBERS_AND_SELECTED_SEMANTICS_INDEPENDENTLY_VERIFIED'
    assert proof['source_report_sha256'] == digest(report_path)
    assert proof['archive_sha256'] == digest(Path(report['archive']['path'])) == report['archive']['sha256']
    assert report['archive']['members'] == 2737 and not report['archive']['args_or_log_members']
    assert report['archive']['outputs_members'] == 0 and len(report['archive']['checkpoint_members']) == 2
    assert len(report['file_comparisons']) == proof['file_pairs_verified'] == 13
    assert all(report['selected_ast_comparisons_without_docstrings'].values())
    assert not report['historical_training_source_authenticated'] and not proof['historical_source_authenticated']
    assert all(report[k] == 0 for k in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    assert len(report['solver_difference']['archive_fit_torch_save_calls']) == 0
    assert len(report['solver_difference']['current_fit_torch_save_calls']) == 1
    assert 'epoch ==' in report['solver_difference']['archive_nonbest_restore_conditions'][0]['condition']
    assert any('epoch >=' in item['condition'] for item in report['solver_difference']['current_fit_restore_conditions'])
    for item in report['file_comparisons']:
        relative = item['archive_member'].removeprefix('DEIM_1572862469/')
        assert digest(HERE / 'source_snapshot/archive' / relative) == item['archive_sha256']
        assert digest(Path(item['current_path'])) == item['current_sha256']
        current_relative = Path(item['current_path']).relative_to(Path(r'C:\Users\jiwei\Desktop\DEIM'))
        assert digest(HERE / 'source_snapshot/current' / current_relative) == item['current_sha256']
    previous = INDEX / '主要结论证据索引_第二十四轮.json'
    catalog = REC / '第二十四轮会议稿论述复核_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十五轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十五轮.txt'
    receipt = HERE / 'source_archive_comparison_receipt.json'
    assert not any(path.exists() for path in [index_json, index_txt, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt', REMOTE / '公平主对照重做提案_20261105.md']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + CURRENT_SHA) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十五轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    index = {'revision': 25, 'status': 'ARCHIVED_SOURCE_VARIANTS_DOCUMENTED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(catalog),
             'archive_comparison': ref(report_path), 'independent_verification': ref(proof_path),
             'explanation': ref(explanation),
             'finding': 'The enhanced-backbone and MSCA source files match byte for byte across the archive and current tree; the n/s configuration include names differ but their directly referenced data configuration bytes match. Selected code-tree trainer branches differ.',
             'limits': 'The archive lacks the searched historical logs and output checkpoints; its source and timestamps are not authenticated as the historical training version. Selected AST equality is not full-program or independent experimental equivalence.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
             'submission_ready': False, 'remaining': 'The paired BASE/FULL comparison remains incomplete; any new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十五轮\n当前稿件未改，SHA256：' + CURRENT_SHA + '\n'
        '迁移压缩档中的改进骨干与MSCA文件与现存文件逐字节一致；n/s配置的include名字不同但所比对应配置文件内容一致。\n'
        '所查训练器的评价数组保存和阶段二非最佳恢复分支不同；档案未包含所查历史日志/输出检查点，不认证历史训练源码。\n'
        '详情：' + str(explanation) + '\n机器可读索引：' + str(index_json) + '\n无模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第二十五轮旧档案源码比对】\n'
        '当前稿件未改：' + str(CURRENT) + '\nSHA256：' + CURRENT_SHA + '\n'
        '迁移档案与当前树的改进骨干和MSCA文件一致，但所查训练器分支不同；档案未证明历史源码身份。正文既有源码不确定性说明仍适用。\n'
        '详情见修订记录_20261103/第二十五轮旧档案源码比对_20261105/迁移档案与现存源码比对说明.md；最新证据索引为主要结论证据索引_第二十五轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。公平主对照仍不完整，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    addition = ('\n第二十五轮：迁移档案与当前源码比对（2026-11-05）\n'
        '旧迁移ZIP含2737个成员、无所查args.json/log.txt或outputs；改进骨干和MSCA源文件与现存树逐字节一致，选定SimAM、EMA及骨干分支的语法树一致。n/s配置文本的include名不同但对应include文件字节一致。档案和当前训练器的周期评价保存及非最佳恢复分支不同；档案没有绑定历史权重或日志，不能认证历史源码。详细比较和源码快照见第二十五轮旧档案源码比对_20261105。\n'
        '论文未改，SHA256：' + CURRENT_SHA + '。无模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + addition)
    lead = ('最新旧档案源码比对（2026-11-05）：改进骨干/MSCA文件与现存树字节一致，但训练器保存及恢复分支有差异；档案缺少所查历史日志与输出权重，不能认证历史源码。详见修订记录_20261103/第二十五轮旧档案源码比对_20261105。\n')
    replace_text(paths[3], lead + prior[3])
    proposal = ('\n## 旧迁移源码补核（不改变方案或授权状态）\n\n'
        '旧迁移ZIP的改进骨干及MSCA文件与当前树字节一致，但所查训练器在周期评价文件保存和阶段二非最佳恢复上与当前代码不同，且档案没有所查历史日志/输出检查点。未来若获授权重做，应固定并核验两臂实际执行的同一训练代码，不能把该档案或当前树单独称为已认证的历史源码。详情见修订记录_20261103/第二十五轮旧档案源码比对_20261105/迁移档案与现存源码比对说明.md。\n')
    replace_text(paths[4], prior[4] + proposal)
    write_new(receipt, {'status': 'SOURCE_COMPARISON_REGISTERED_MANUSCRIPT_UNCHANGED', 'revision': 25,
                        'manuscript': ref(CURRENT), 'index': ref(index_json), 'explanation': ref(explanation),
                        'original_preserved': True, 'submission_ready': False})
    print(json.dumps({'status': 'SOURCE_COMPARISON_REGISTERED_MANUSCRIPT_UNCHANGED',
                      'manuscript_sha256': digest(CURRENT), 'index_sha256': digest(index_json),
                      'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
