"""Promote the reviewed optimizer correction and update traceable project records."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import copy
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
REC, PAPER = HERE.parent, HERE.parent.parent
REMOTE = PAPER / '远程公平对照_20261103'
INDEX = REC / '主要结论证据索引'
S13 = REC / '第十三轮恢复语义核查_20261105'
CURRENT = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第十四轮候选_v2.docx'
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第十三轮备份.docx'
OLD = '3d6a72252567faebd7dd5bdb7d0f665d723f468edbe7e17f70caf08a76f9c1f9'
NEW = '15c62d7f363775645d73e29710366b32309f2b500ec6a08bd524be826b9eb3f8'
ORIGINAL = 'ebd5159bd09d122367fd683c431c7b297800af7080504f4c5effeaaa4ef6c323'

def sha(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())

def replace_text(path, text):
    temp = path.with_name(path.name + '.fourteenth-promotion.tmp')
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
    patch = read(HERE / 'fourteenth_revision_v2_pending_qa.json')
    qa_path = HERE / 'fourteenth_revision_visual_qa.json'
    qa = read(qa_path)
    audit_path = HERE / 'optimizer_groups_audit.json'
    check_path = HERE / 'optimizer_semantic_verification.json'
    audit, check = read(audit_path), read(check_path)
    explanation = HERE / '优化器分组核查说明.md'
    assert patch['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert sha(audit_path) == patch['optimizer_audit_sha256'] == qa['optimizer_audit_sha256'] == check['audit_sha256']
    assert sha(check_path) == patch['semantic_check_sha256'] == qa['semantic_check_sha256']
    assert audit['status'] == 'OPTIMIZER_NAME_PATTERNS_AND_CHECKPOINT_GROUPS_INSPECTED'
    assert check['status'] == 'BN_LAYER_INDICES_REGEX_AND_SAVED_GROUP_VALUES_VERIFIED'
    assert not audit['historical_source_identity_certified'] and not check['historical_checkpoint_id_name_mapping_proven']
    assert all(audit[x] == 0 for x in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    for item in audit['sources']:
        assert sha(Path(item['path'])) == item['sha256']
    assert [item['run'] for item in audit['runs']] == ['N0', 'N1', 'N2', 'N3', 'N4', 'S1']
    for item in audit['runs']:
        for key in ['args', 'checkpoint_metadata']:
            assert sha(Path(item[key]['path'])) == item[key]['sha256']
        if item['saved_optimizer_summary']:
            saved = item['saved_optimizer_summary']['checkpoint']
            assert sha(Path(saved['path'])) == saved['sha256']
    prior_catalog_path = S13 / 'manuscript_version_locations.json'
    prior_catalog = read(prior_catalog_path)
    for item in prior_catalog['versions']:
        assert sha(Path(item['retained_file']['path'])) == item['sha256']
    prior_indices = [INDEX / ('主要结论证据索引_第十三轮' + suffix) for suffix in ['.txt', '.json']]
    assert read(prior_indices[1])['manuscript']['sha256'] == OLD
    diagnostic = REMOTE / 'diagnostic_completion_receipt_20261105.json'
    assert read(diagnostic)['status'] == 'ONE_AUTHORIZED_DIAGNOSTIC_COMPLETED_AND_EVIDENCE_VERIFIED'
    base_config_path = REMOTE / 'remote_evidence/execution_configs/BASE.yml'
    full_config_path = REMOTE / 'remote_evidence/execution_configs/FULL.yml'
    base, full = read(base_config_path), read(full_config_path)
    assert base['optimizer'] == full['optimizer']
    assert base['optimizer']['params'][0]['params'] == '^(?=.*backbone)(?!.*norm|bn).*$'
    assert all(base[key] == full[key] for key in ['lrsheduler', 'lr_gamma', 'warmup_iter', 'lr_warmup_scheduler'])
    assert base['lrsheduler'] == 'flatcosine' and base['lr_gamma'] == 1.0 and base['warmup_iter'] == 2000
    assert base['lr_warmup_scheduler']['warmup_duration'] == 500

    outputs = [HERE / 'fourteenth_revision_patch.json', HERE / 'manuscript_version_locations.json',
               INDEX / '主要结论证据索引_第十四轮.json', INDEX / '主要结论证据索引_第十四轮.txt']
    assert not any(path.exists() for path in outputs)
    text_paths = [PAPER / '当前修订入口.txt', REMOTE / '结果并入与验收清单.txt',
                  REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
                  REMOTE / '公平主对照重做提案_20261105.md']
    text_backups = [HERE / ('text_before_revision14_%d.txt' % i) for i in range(len(text_paths))]
    assert not any(path.exists() for path in text_backups)
    old_text = [path.read_text(encoding='utf-8-sig') for path in text_paths]
    assert old_text[1].count('当前稿件SHA256：' + OLD) == 1
    assert old_text[3].count('稿件现为第十三轮修订版') == 1
    assert '以上是供审阅的新实验提案，不是运行授权' in old_text[4]
    for source, destination in zip(text_paths, text_backups):
        shutil.copy2(source, destination)

    temp = CURRENT.with_name(CURRENT.stem + '_revision14_promotion.tmp.docx')
    assert not temp.exists()
    with CANDIDATE.open('rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    assert sha(temp) == NEW and sha(CURRENT) == OLD
    os.replace(temp, CURRENT)
    assert sha(CURRENT) == NEW and sha(BACKUP) == OLD and sha(original) == sha(original_backup)
    now = datetime.now(timezone.utc).isoformat()
    promoted = dict(patch, status='PROMOTED_AFTER_VISUAL_AND_STRUCTURAL_QA', promoted_utc=now,
                    current_manuscript_sha256=NEW, visual_qa_sha256=sha(qa_path))
    write_new(outputs[0], promoted)
    versions = copy.deepcopy(prior_catalog['versions'])
    for item in versions:
        if item['revision'] == 13:
            item['retained_file'] = ref(BACKUP)
        item['current_canonical_has_these_bytes'] = sha(Path(item['canonical_path_recorded_at_the_time'])) == item['sha256']
    versions.append({'revision': 14, 'sha256': NEW, 'canonical_path_recorded_at_the_time': str(CURRENT),
                     'retained_file': ref(CANDIDATE), 'current_canonical_has_these_bytes': True,
                     'reviewed_version_proof': ref(outputs[0])})
    catalog = {'status': 'CURRENT_AND_INHERITED_MANUSCRIPT_VERSIONS_LOCATED', 'checked_utc': now,
               'current_manuscript_at_this_audit': ref(CURRENT), 'versions': versions,
               'previous_catalog': ref(prior_catalog_path),
               'rule': 'Resolve each version-specific hash to its retained_file; prior canonical paths are historical.',
               'scope': 'Original manuscript and revisions 7–14 in the inherited chain; file identity only.',
               'historical_indices_modified': False}
    write_new(outputs[1], catalog)
    index = {'revision': 14, 'status': 'OPTIMIZER_NAME_BASED_GROUPING_CORRECTED',
             'created_utc': now, 'manuscript': ref(CURRENT), 'previous_indices': [ref(x) for x in prior_indices],
             'optimizer_config_and_saved_group_audit': ref(audit_path), 'semantic_verification': ref(check_path),
             'plain_language_explanation': ref(explanation), 'visual_qa': ref(qa_path),
             'manuscript_version_locations': ref(outputs[1]), 'diagnostic_completion': ref(diagnostic),
             'historical_training_source_identity_certified': False,
             'historical_optimizer_parameter_id_name_mapping_certified': False,
             'description': 'The inspected s configuration groups by parameter-name regex, not layer type; some BatchNorm parameters match the lower-rate default-decay group. The n patterns place backbone .bn names without norm in the default-decay group. The manuscript uses a conditional name-based description.',
             'frozen_fair_config_optimizer_identical': True,
             'frozen_fair_configs': [ref(base_config_path), ref(full_config_path)],
             'model_instantiations': 0, 'model_forwards': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
             'submission_ready': False,
             'remaining': 'The interrupted BASE/FULL fair pair is incomplete; the later BASE diagnostic does not replace it. Further experiments require explicit authorization.'}
    write_new(outputs[2], index)
    index_text = ('主要结论证据索引 第十四轮\n当前稿件SHA256：' + NEW + '\n'
                  '第4.1节更正s学习率分组：所查规则按参数名称而非层类型匹配；部分顺序容器BN名称不含bn，'
                  '属于较低学习率及默认衰减组。五组n的未分组括号正则也使常见骨干.bn名称进入默认衰减组。\n'
                  '六组保存配置和参数名、N0/N4/S1检查点中的优化器组、所查源码及独立核查见第十四轮优化器分组核查_20261105；'
                  '历史源码及优化器参数ID到名称的完整映射未认证。其余证据继承第十三轮索引及哈希链。\n'
                  '表值、图、公式、引用未改；10页已逐页检查。本轮无模型构建、推理、重评或训练。'
                  '公平BASE/FULL对照仍缺，尚不能投稿。\n机器可读索引：' + str(outputs[2]) + '\n')
    with outputs[3].open('x', encoding='utf-8', newline='\n') as out:
        out.write(index_text)

    header = ('【最新稿件入口：2026-11-05，第十四轮优化器分组核查】\n'
              '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
              '第4.1节按所查参数名称规则修正s学习率和权重衰减描述；部分BN参数位于较低学习率、默认衰减组。'
              '第十三轮稿件已备份；10页逐页检查，表注与表格同页。\n'
              '详情见修订记录_20261103/第十四轮优化器分组核查_20261105/优化器分组核查说明.md；'
              '最新证据索引为主要结论证据索引_第十四轮.txt及JSON。\n'
              '本轮无新增实验。授权的BASE诊断不构成公平主对照；重做方案仍待明确授权，尚不能投稿。'
              '以下较早运行与版本状态均为历史。\n\n')
    note = ('\n第十四轮：优化器名称分组核查（2026-11-05）\n'
            '所查s规则按名称中的bn/norm匹配而非按BN层类型匹配；某些顺序容器BN参数名称不含bn，'
            '会进入2e-4学习率、1e-4默认衰减组。第4.1节据此修正，默认衰减和按名称例外也已区分。'
            '所查n正则中的未分组bn分支使常见骨干.bn参数匹配默认衰减组；N0/N4保存检查点的骨干零衰减组为空。'
            '不能仅凭保存参数ID确定历史完整名称映射，历史源码身份亦未认证。\n'
            '第十三轮备份SHA256：' + OLD + '；当前修订稿SHA256：' + NEW + '。'
            '只改一个正文段落，10页逐页检查，表格、图片、公式、数值、引用与批注锚点保留。'
            '本轮没有构建模型、前向、评价或训练。公平BASE/FULL主对照仍缺，尚不能投稿。\n')
    proposal_note = ('\n## 优化器规则补充核查（不改变方案或授权状态）\n\n'
                     '已核查原冻结 BASE/FULL 配置：两组使用相同的 n 优化器名称正则和学习率设置。'
                     '该规则按名称匹配，所查实现中常见骨干 `.bn` 参数会落入默认权重衰减组；'
                     '结构不同的参数应在任何获授权的预检中逐组核对，不能只凭“同一配置”推断层类型分组。\n\n'
                     '所查实现先由 500 步 LinearWarmup 构造函数将初始学习率设为基础值的 1/500，'
                     '再在每次更新后应用 2000 步 flat-cosine 规则；gamma 为 1.0 时其预热后的调度值不衰减。'
                     '这只是对既定规则的说明，不改变两个方案、预算或端点；尚未启动任何新实验。'
                     '历史运行源码身份和完整参数 ID 到名称映射仍未认证。详细证据见修订记录_20261103/'
                     '第十四轮优化器分组核查_20261105/优化器分组核查说明.md。\n')
    goal_text = old_text[3].replace('稿件现为第十三轮修订版', '稿件现为第十四轮修订版', 1)
    replace_text(text_paths[0], header + old_text[0])
    replace_text(text_paths[1], old_text[1].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW, 1))
    replace_text(text_paths[2], old_text[2] + note)
    replace_text(text_paths[3], goal_text + note)
    replace_text(text_paths[4], old_text[4] + proposal_note)
    print(json.dumps({'status': promoted['status'], 'current_sha256': sha(CURRENT),
                      'original_sha256': sha(original), 'index': str(outputs[3])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
