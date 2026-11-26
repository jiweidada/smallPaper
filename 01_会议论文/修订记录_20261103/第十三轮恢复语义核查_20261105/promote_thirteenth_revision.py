"""Promote the reviewed clarification and record the corrected evidence trail."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import copy
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDEX = REC / '主要结论证据索引'
S12 = REC / '第十二轮Stem尺寸核查_20261105'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十三轮候选.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十二轮备份.docx'
OLD = 'c49695664206778e6d0537b050194ddf1ab9a651021e8f64e8d4dacab7e22da9'
NEW = '3d6a72252567faebd7dd5bdb7d0f665d723f468edbe7e17f70caf08a76f9c1f9'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'

def sha(path):
    return sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())

def replace_text(path, text):
    temp = path.with_name(path.name + '.thirteenth-promotion.tmp')
    assert not temp.exists()
    with temp.open('x', encoding='utf-8', newline='\n') as out:
        out.write(text)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)

def ref(path):
    return {'path': str(path), 'sha256': sha(path)}

def main():
    assert sha(CURRENT) == sha(BACKUP) == OLD and sha(CANDIDATE) == NEW
    original = PAPER / '稿件/MPSMA-DEIM_conference_EN.docx'
    original_backup = PAPER / '稿件/备份_20261103/MPSMA-DEIM_conference_EN_original.docx'
    assert sha(original) == sha(original_backup) == ORIGINAL
    patch = read(HERE / 'thirteenth_revision_pending_qa.json')
    qa_path = HERE / 'thirteenth_revision_visual_qa.json'
    qa = read(qa_path)
    evidence_path = HERE / 'validation_restore_static_audit.json'
    evidence = read(evidence_path)
    assert patch['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert sha(evidence_path) == patch['static_evidence_sha256'] == qa['source_and_configuration_audit_sha256']
    assert evidence['status'] == 'RESTORE_SEMANTICS_AND_COLLATE_PATH_STATICALLY_CHECKED'
    assert not evidence['historical_solver_identity_certified']
    assert all(evidence[key] == 0 for key in ['model_executions', 'evaluator_executions', 'training_updates'])
    for source in evidence['sources']:
        assert sha(Path(source['path'])) == source['sha256']
    assert [item['run'] for item in evidence['configs']] == ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']
    for item in evidence['configs']:
        assert sha(REC / 'source_snapshot' / item['run'] / 'args.json') == item['args_sha256']
        assert item['train_base_size_repeat'] is None
    correction = evidence['prior_audit_correction']
    assert sha(Path(correction['previous_script'])) == correction['sha256']
    assert sha(Path(correction['previous_report'])) == correction['previous_report_sha256']
    assert correction['synthetic_counterexample_old_check_passes'] is True
    explanation = HERE / '恢复语义与配置核查说明.md'
    correction_note = S12 / 'Stem尺寸核查说明.md'
    assert '第十三轮核查更正' in correction_note.read_text(encoding='utf-8')
    prior_catalog_path = S12 / 'manuscript_version_locations.json'
    prior_catalog = read(prior_catalog_path)
    for item in prior_catalog['versions']:
        assert sha(Path(item['retained_file']['path'])) == item['sha256']
    prior_indices = [INDEX / ('主要结论证据索引_第十二轮' + suffix) for suffix in ['.txt', '.json']]
    assert read(prior_indices[1])['manuscript']['sha256'] == OLD
    diagnostic = REMOTE / 'diagnostic_completion_receipt_20261105.json'
    assert read(diagnostic)['status'] == 'ONE_AUTHORIZED_DIAGNOSTIC_COMPLETED_AND_EVIDENCE_VERIFIED'

    outputs = [HERE / 'thirteenth_revision_patch.json', HERE / 'manuscript_version_locations.json',
               INDEX / '主要结论证据索引_第十三轮.json', INDEX / '主要结论证据索引_第十三轮.txt']
    assert not any(path.exists() for path in outputs)
    text_paths = [PAPER / '当前修订入口.txt', REMOTE / '结果并入与验收清单.txt',
                  REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt']
    text_backups = [HERE / ('text_before_revision13_%d.txt' % i) for i in range(len(text_paths))]
    assert not any(path.exists() for path in text_backups)
    old_text = [path.read_text(encoding='utf-8-sig') for path in text_paths]
    assert old_text[1].count('当前稿件SHA256：' + OLD) == 1
    assert old_text[3].count('稿件现为第十二轮修订版') == 1
    for source, destination in zip(text_paths, text_backups):
        shutil.copy2(source, destination)

    temp = CURRENT.with_name(CURRENT.stem + '_revision13_promotion.tmp.docx')
    assert not temp.exists()
    with CANDIDATE.open('rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    assert sha(temp) == NEW and sha(CURRENT) == OLD
    os.replace(temp, CURRENT)
    assert sha(CURRENT) == NEW and sha(BACKUP) == OLD and sha(original) == sha(original_backup)
    now = datetime.now(timezone.utc).isoformat()
    promoted = dict(patch, status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA',
                    promoted_utc=now, current_manuscript_sha256=NEW, visual_qa_sha256=sha(qa_path))
    write_new(outputs[0], promoted)
    versions = copy.deepcopy(prior_catalog['versions'])
    for item in versions:
        if item['revision'] == 12:
            item['retained_file'] = ref(BACKUP)
        item['current_canonical_has_these_bytes'] = sha(Path(item['canonical_path_recorded_at_the_time'])) == item['sha256']
    versions.append({'revision': 13, 'sha256': NEW, 'canonical_path_recorded_at_the_time': str(CURRENT),
                     'retained_file': ref(CANDIDATE), 'current_canonical_has_these_bytes': True,
                     'reviewed_version_proof': ref(outputs[0])})
    catalog = {'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED', 'checked_utc': now,
               'current_manuscript_at_this_audit': ref(CURRENT), 'versions': versions,
               'previous_catalog': ref(prior_catalog_path),
               'rule': 'Resolve each version-specific hash to its retained_file; prior canonical paths are historical.',
               'scope': 'Original manuscript and revisions 7–13 in the inherited chain; file identity only.',
               'historical_indices_modified': False}
    write_new(outputs[1], catalog)
    index = {'revision': 13, 'status': 'VALIDATION_RESTORE_CLARIFIED_AND_PRIOR_COLLATE_AUDIT_CORRECTED',
             'created_utc': now, 'manuscript': ref(CURRENT), 'previous_indices': [ref(x) for x in prior_indices],
             'static_restore_and_collate_evidence': ref(evidence_path), 'plain_language_explanation': ref(explanation),
             'prior_audit_correction_note': ref(correction_note), 'visual_qa': ref(qa_path),
             'manuscript_version_locations': ref(outputs[1]), 'diagnostic_completion': ref(diagnostic),
             'historical_solver_identity_certified': False,
             'correction': 'The revision-12 base_size_repeat check read a missing top-level key and was not probative. Revision 13 checks the actual nested field in six saved training configurations and the inspected validation-collator default; the prior conclusion remains supported for these retained materials.',
             'description': 'The inspected solver selects a stage-one checkpoint using validation AP and may restore it during later training; its historical identity is unverified.',
             'model_executions': 0, 'evaluator_reruns': 0, 'training_updates': 0,
             'submission_ready': False,
             'remaining': 'The interrupted BASE/FULL fair pair is incomplete; the later BASE diagnostic does not replace it. Further experiments require explicit authorization.'}
    write_new(outputs[2], index)
    index_text = (
        '主要结论证据索引 第十三轮\n当前稿件SHA256：' + NEW + '\n'
        '留存训练器使用验证AP选择阶段一检查点，后期可能恢复；历史训练源码是否相同尚无认证，不能断言历史运行确实发生恢复。\n'
        '第十二轮base_size_repeat核查误读顶层缺失键，原项检查不成立；本轮改查六组训练配置的实际嵌套字段，'
        '结合验证collator在所查实现中的默认值，结论仍为所查配置未启用批次多尺度。旧JSON保留并附更正说明。\n'
        '详情、静态证据、候选、上一版备份及10页版式见第十三轮恢复语义核查_20261105。'
        '其余证据继承第十二轮索引及哈希链；历史训练源码版本不能确认。\n'
        '本轮未运行模型、重评或训练。BASE诊断不是公平主对照；该对照仍缺，尚不能投稿。\n'
        '机器可读索引：' + str(outputs[2]) + '\n')
    with outputs[3].open('x', encoding='utf-8', newline='\n') as out:
        out.write(index_text)

    header = ('【最新稿件入口：2026-11-05，第十三轮恢复语义核查】\n'
              '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
              '稿件说明所查训练器可能依据验证AP恢复阶段一检查点，并注明历史训练是否使用该版本尚未证实。'
              '第十二轮误查批次多尺度配置路径已更正，六组实际嵌套配置支持原结论。第十二轮稿件已备份；10页逐页核验通过。\n'
              '详情见修订记录_20261103/第十三轮恢复语义核查_20261105/恢复语义与配置核查说明.md；'
              '最新证据索引为主要结论证据索引_第十三轮.txt及JSON。\n'
              '本轮无新增实验。授权的BASE诊断已完成但不构成公平主对照；重做方案仍待明确授权，尚不能投稿。'
              '以下较早运行与版本状态均为历史。\n\n')
    note = ('\n第十三轮：验证驱动恢复及配置核查更正（2026-11-05）\n'
            '所查训练器按验证AP选择阶段一检查点，后期可能恢复并重置last_epoch；历史训练是否使用该版本未知，稿件已据此限定表述。\n'
            '更正第十二轮脚本误读yaml_cfg顶层缺失键的问题：重新核查六组训练配置的实际嵌套base_size_repeat均为null，'
            '验证项省略且所查实现默认None；对留存材料的批次多尺度结论不变。旧JSON保留，附独立更正说明。\n'
            '第十二轮备份SHA256：' + OLD + '；当前修订稿SHA256：' + NEW + '。'
            'Word只读导出10页并逐页检查；图、表、公式、引用及实验数值未改。本轮无模型执行、重评或训练。'
            '公平BASE/FULL对照仍缺，尚不能投稿。\n')
    goal_text = old_text[3].replace('稿件现为第十二轮修订版', '稿件现为第十三轮修订版', 1)
    replace_text(text_paths[0], header + old_text[0])
    replace_text(text_paths[1], old_text[1].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1))
    replace_text(text_paths[2], old_text[2] + note)
    replace_text(text_paths[3], goal_text + note)
    print(json.dumps({'status': promoted['status'], 'current_sha256': sha(CURRENT),
                      'original_sha256': sha(original), 'index': str(outputs[3])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
