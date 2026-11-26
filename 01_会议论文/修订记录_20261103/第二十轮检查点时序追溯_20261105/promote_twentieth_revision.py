"""Promote the checked provenance wording and preserve an auditable revision chain."""
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
INDEX = REC / '主要结论证据索引'
REMOTE = PAPER / '远程公平对照_20261103'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十九轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十轮候选.docx'
OLD = '6c603477dd95225525e4505fbf8c126f9b349e1ba5fa6cffaf87df4672238969'
NEW = 'eae4bbc357b9355003ed6809fbba3077bcd5148c764b1238ffca13f5a21f6409'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'


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
    temp = path.with_name(path.name + '.rev20.tmp')
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
    pending = read(HERE / 'twentieth_revision_pending_qa.json')
    qa_path = HERE / 'twentieth_revision_visual_qa.json'
    qa = read(qa_path)
    trace_path = HERE / 'checkpoint_time_trace.json'
    trace = read(trace_path)
    timeline_path = HERE / 'n2_conditional_timeline.json'
    timeline = read(timeline_path)
    semantics_path = HERE / 'n2_source_semantics_verification.json'
    semantics = read(semantics_path)
    explanation = HERE / 'N2检查点条件追溯说明.md'
    assert pending['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert pending['timeline_sha256'] == digest(timeline_path)
    assert pending['source_semantics_sha256'] == digest(semantics_path)
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED'
    assert qa['inspected_pages'] == list(range(1, 11))
    assert qa['pdf_sha256'] == digest(HERE / 'rendered_word/manuscript.pdf')
    for name, expected in qa['page_png_sha256'].items():
        assert digest(HERE / 'rendered_word' / name) == expected
    assert trace['status'] == 'SAVED_CHECKPOINT_AND_EVALUATION_TIMESTAMPS_INSPECTED'
    assert timeline['status'] == 'N2_METADATA_CONSISTENT_WITH_CONDITIONAL_SOLVER_BOOKKEEPING'
    assert semantics['status'] == 'INSPECTED_CODE_AND_N2_COUNTER_ARITHMETIC_CONSISTENT'
    assert timeline['trace_sha256'] == semantics['source_trace_sha256'] == digest(trace_path)
    assert semantics['timeline_sha256'] == digest(timeline_path)
    assert all(semantics['static_source_checks'].values())
    for item in semantics['source_files']:
        assert digest(Path(item['path'])) == digest(Path(item['snapshot'])) == item['sha256']
    assert timeline['args_sha256'] == digest(REC / 'source_snapshot/N2/args.json')
    assert timeline['log_sha256'] == digest(REC / 'source_snapshot/N2/log.txt')
    assert timeline['training_manifest_sha256'] == digest(PAPER / '公平对照方案_20261103/training_image_manifest.json')
    n2 = next(item for item in trace['runs'] if item['run'] == 'N2')
    stage1 = trace['N2_additional_checkpoints']['best_stg1.pth']
    for item in [n2['checkpoint'], stage1, n2['best_log_evaluation_file']]:
        assert digest(Path(item['path'])) == item['sha256']
    assert (timeline['stage1_saved_epoch'], timeline['stage1_saved_ema_updates']) == (291, 472164)
    assert (timeline['saved_best_stage2_epoch_field'], timeline['saved_best_stage2_ema_updates']) == (296, 480249)
    assert timeline['assumptions']['assumed_steps_per_epoch'] == 1617
    assert all(item['is_new_global_best'] for item in timeline['conditional_trajectory'])
    assert len(timeline['conditional_trajectory']) == 5
    assert timeline['n0_saved_evaluation_embedded_date_span_seconds'] > 24 * 3600
    assert timeline['n0_saved_evaluation_filesystem_mtime_span_seconds'] < 10
    assert all(timeline[key] == 0 for key in ['models_run', 'forwards', 'evaluator_runs', 'optimizer_updates'])
    with ZipFile(CURRENT) as current_zip, ZipFile(CANDIDATE) as candidate_zip:
        assert candidate_zip.testzip() is None and current_zip.namelist() == candidate_zip.namelist()
        assert [name for name in current_zip.namelist() if current_zip.read(name) != candidate_zip.read(name)] == ['word/document.xml']

    previous = INDEX / '主要结论证据索引_第十九轮.json'
    previous_catalog = REC / '第十九轮历史评价追溯_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十轮.txt'
    locations = HERE / 'manuscript_version_locations.json'
    receipt = HERE / 'twentieth_revision_patch.json'
    assert not any(path.exists() for path in [index_json, index_txt, locations, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt',
             REC / '目标验收与阻塞记录.txt', REMOTE / '结果并入与验收清单.txt']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + OLD) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    temp = CURRENT.with_name(CURRENT.name + '.rev20.tmp')
    assert not temp.exists()
    shutil.copy2(CANDIDATE, temp)
    assert digest(temp) == NEW
    os.replace(temp, CURRENT)
    assert digest(CURRENT) == NEW and digest(original) == digest(original_backup) == ORIGINAL
    catalog = {'revision': 20, 'previous_catalog': ref(previous_catalog),
               'versions': [{'revision': 19, 'sha256': OLD, 'retained_file': ref(BACKUP)},
                            {'revision': 20, 'sha256': NEW, 'retained_file': ref(CURRENT)}],
               'note': 'The current-path entry in earlier catalogs described the then-current revision; use this catalog for revisions 19 and 20.'}
    write_new(locations, catalog)
    index = {'revision': 20, 'status': 'CONDITIONAL_N2_BOOKKEEPING_DOCUMENTED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(locations), 'previous_manuscript': ref(BACKUP),
             'checkpoint_time_trace': ref(trace_path), 'conditional_timeline': ref(timeline_path),
             'source_semantics_verification': ref(semantics_path), 'explanation': ref(explanation), 'visual_qa': ref(qa_path),
             'finding': 'Under the explicitly stated assumptions, the retained N2 stage-one metadata plus five epochs of 1617 EMA updates exactly matches the saved stage-two counters and the inspected restore logic.',
             'limits': 'Historical source, data length, batch updates and restore history are not authenticated. Neither the counter match nor current file timestamps bind the checkpoint hash to the historical evaluation. N0 timestamps demonstrate that current mtimes cannot be assumed to preserve chronology.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0, 'submission_ready': False,
             'remaining': 'A matched BASE/FULL comparison remains incomplete and new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十轮\n当前稿件SHA256：' + NEW + '\n'
        'N2阶段一计数291/472164，在所查恢复逻辑、每轮1617次EMA更新和无额外恢复等假设下，经过五轮得到296/480249，与留存阶段二计数一致。\n'
        '该一致性是条件性解释，历史代码、更新次数和权重与评价的哈希绑定未获认证。N0内部日期跨度和现有修改时间跨度明显不同，不能仅凭文件时间判定历史顺序。\n'
        '详情：' + str(explanation) + '\n机器可读索引：' + str(index_json) + '\n'
        '无新模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新稿件入口：2026-11-05，第二十轮检查点时序追溯】\n'
        '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
        '第4.6节澄清所查训练器的阶段恢复规则，记载N2计数在明确假设下与该规则一致；历史使用同一代码和历史权重归属均未认证。第十九轮稿件已备份，10页逐页检查。\n'
        '详情见修订记录_20261103/第二十轮检查点时序追溯_20261105/N2检查点条件追溯说明.md；最新证据索引为主要结论证据索引_第二十轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。公平主对照仍待授权与完成，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    note = ('\n第二十轮：N2检查点条件时序追溯（2026-11-05）\n'
        '所查训练器在阶段边界恢复阶段一状态，并可能在后续未刷新最佳值时再次恢复。假设每轮1617次EMA更新且无额外恢复，阶段一计数291/472164经过索引296–300五轮得到296/480249，恰与N2阶段二检查点一致。该计算不证明历史运行使用所查代码或该权重产生历史评价。N0文件内部日期与当前修改时间跨度明显不同，故不能仅凭修改时间认定历史顺序。\n'
        '第4.6节澄清恢复规则和条件性解释；第十九轮备份SHA256：' + OLD + '；当前稿件SHA256：' + NEW + '。6表、7图、6式和5组批注锚点保留，10页逐页检查。本轮无新模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + note)
    lead = ('最新N2检查点条件追溯（2026-11-05）：所查训练器恢复规则与假设的五轮EMA计数可解释留存元数据差异，但历史代码、数据和权重与评价的绑定未认证；时间戳不能独立证明归属。此线索不替代公平主对照。详见修订记录_20261103/第二十轮检查点时序追溯_20261105。\n')
    replace_text(paths[3], lead + prior[3].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW))
    write_new(receipt, {'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'revision': 20,
                        'manuscript': ref(CURRENT), 'previous_manuscript': ref(BACKUP),
                        'evidence_index': ref(index_json), 'visual_qa': ref(qa_path), 'original_preserved': True})
    print(json.dumps({'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'manuscript_sha256': digest(CURRENT),
                      'index_sha256': digest(index_json), 'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
