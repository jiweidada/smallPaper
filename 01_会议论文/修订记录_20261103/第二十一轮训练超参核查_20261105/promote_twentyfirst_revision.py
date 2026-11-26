"""Promote the visually checked training-parameter clarification."""
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
BACKUP = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十轮备份.docx'
CANDIDATE = HERE / 'MPSMA-DEIM_conference_EN_revised_第二十一轮候选.docx'
OLD = 'eae4bbc357b9355003ed6809fbba3077bcd5148c764b1238ffca13f5a21f6409'
NEW = '9060aee70bae6804fbf61a4adbc76ce34a2d4be7e44d17dfa2500877300adbab'
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
    temp = path.with_name(path.name + '.rev21.tmp')
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
    pending = read(HERE / 'twentyfirst_revision_pending_qa.json')
    qa_path = HERE / 'twentyfirst_revision_visual_qa.json'
    qa = read(qa_path)
    audit_path = HERE / 'training_parameters_audit.json'
    audit = read(audit_path)
    explanation = HERE / '训练配置与调度语义说明.md'
    assert pending['candidate_sha256'] == qa['candidate_sha256'] == NEW
    assert pending['training_audit_sha256'] == digest(audit_path)
    assert qa['status'] == 'VISUAL_AND_STRUCTURAL_QA_PASSED' and qa['inspected_pages'] == list(range(1, 11))
    assert qa['pdf_sha256'] == digest(HERE / 'rendered_word/manuscript.pdf')
    for name, expected in qa['page_png_sha256'].items():
        assert digest(HERE / 'rendered_word' / name) == expected
    assert audit['status'] == 'SIX_CONFIGS_AND_INSPECTED_PARAMETER_RESOLUTION_CHECKED'
    assert len(audit['runs']) == 6 and all(audit['static_checks'].values())
    assert not audit['historical_source_identity_certified']
    assert all(audit[k] == 0 for k in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    for source in audit['sources'].values():
        assert digest(Path(source['path'])) == digest(Path(source['cpu_snapshot'])) == digest(Path(source['round_snapshot'])) == source['sha256']
    for run in audit['runs']:
        assert digest(Path(run['args_path'])) == run['args_sha256']
        assert run['retained']['ema_warmups'] == 1000 and run['retained']['outer_ema_warmups'] == 2000
    with ZipFile(CURRENT) as current_zip, ZipFile(CANDIDATE) as candidate_zip:
        assert candidate_zip.testzip() is None and current_zip.namelist() == candidate_zip.namelist()
        assert [name for name in current_zip.namelist() if current_zip.read(name) != candidate_zip.read(name)] == ['word/document.xml']
    previous = INDEX / '主要结论证据索引_第二十轮.json'
    previous_catalog = REC / '第二十轮检查点时序追溯_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十一轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十一轮.txt'
    locations = HERE / 'manuscript_version_locations.json'
    receipt = HERE / 'twentyfirst_revision_patch.json'
    assert not any(path.exists() for path in [index_json, index_txt, locations, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt', REMOTE / '公平主对照重做提案_20261105.md']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + OLD) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十一轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
    temp = CURRENT.with_name(CURRENT.name + '.rev21.tmp')
    assert not temp.exists()
    shutil.copy2(CANDIDATE, temp)
    assert digest(temp) == NEW
    os.replace(temp, CURRENT)
    assert digest(CURRENT) == NEW and digest(original) == digest(original_backup) == ORIGINAL
    catalog = {'revision': 21, 'previous_catalog': ref(previous_catalog),
               'versions': [{'revision': 20, 'sha256': OLD, 'retained_file': ref(BACKUP)},
                            {'revision': 21, 'sha256': NEW, 'retained_file': ref(CURRENT)}],
               'note': 'Earlier current-path catalog entries refer to their then-current versions; use this catalog for revisions 20 and 21.'}
    write_new(locations, catalog)
    index = {'revision': 21, 'status': 'SAVED_TRAINING_SETTINGS_CLARIFIED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(locations), 'previous_manuscript': ref(BACKUP),
             'training_parameter_audit': ref(audit_path), 'explanation': ref(explanation), 'visual_qa': ref(qa_path),
             'finding': 'All six retained configurations disable AMP, specify gradient clipping 0.1, EMA nested decay 0.9999 and warmups 1000, and a 2000-iteration flat-cosine warmup with gamma 1.0. In the inspected code the nested EMA configuration is used and a 500-step LinearWarmup is initialized separately.',
             'limits': 'The outer BaseConfig ema_warmups=2000 differs from the nested EMA configuration. The inspected EMA state saves no decay or warmup constructor value. Exact historical source identity and per-update trajectory remain unverified; settings do not remove confounds.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0, 'submission_ready': False,
             'remaining': 'A matched BASE/FULL comparison remains incomplete and new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十一轮\n当前稿件SHA256：' + NEW + '\n'
        '六组保存配置的内层ModelEMA warmups=1000，外层BaseConfig属性ema_warmups=2000；所查YAMLConfig构造使用内层值。EMA系数按指数式变化，历史代码身份尚未认证。\n'
        '留存配置还指定AMP关闭、梯度范数裁剪0.1、2000步flat-cosine预热和lr_gamma=1；所查实现另初始化500步LinearWarmup，预热后调度目标不衰减。\n'
        '详情：' + str(explanation) + '\n机器可读索引：' + str(index_json) + '\n'
        '本轮无新模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新稿件入口：2026-11-05，第二十一轮训练超参核查】\n'
        '当前修订稿：' + str(CURRENT) + '\nSHA256：' + NEW + '\n'
        '第4.1节补充六组保存配置的AMP、梯度裁剪、EMA和学习率设置，区分所查实现与未获认证的历史源码。第二十轮稿件已备份，10页逐页检查。\n'
        '详情见修订记录_20261103/第二十一轮训练超参核查_20261105/训练配置与调度语义说明.md；最新证据索引为主要结论证据索引_第二十一轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。公平主对照仍待授权与完成，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    note = ('\n第二十一轮：训练配置层级和调度语义（2026-11-05）\n'
        '六组保存配置的yaml_cfg.ema.warmups=1000，而外层ema_warmups=2000；所查YAMLConfig路径由内层构造EMA，其指数式时间常数为1000。六组均记录AMP关闭、范数裁剪0.1、flatcosine预热2000步、lr_gamma=1及另行初始化的500步LinearWarmup。历史源码及完整逐更新轨迹尚未认证。\n'
        '第4.1节补充设置与条件，第二十轮备份SHA256：' + OLD + '；当前稿件SHA256：' + NEW + '。6表、7图、6式和5组批注锚点保留，10页逐页检查。本轮无新模型、推理、评价或训练；公平BASE/FULL仍缺，尚不能投稿。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + note)
    lead = ('最新训练超参核查（2026-11-05）：六组留存配置的内层EMA warmups为1000，与外层2000不同；所查YAMLConfig使用内层值。论文已补充AMP、裁剪、EMA和调度设置，历史源码仍未认证。此核查不替代公平主对照。详见修订记录_20261103/第二十一轮训练超参核查_20261105。\n')
    replace_text(paths[3], lead + prior[3].replace('当前稿件SHA256：' + OLD, '当前稿件SHA256：' + NEW))
    proposal_note = ('\n## EMA 配置层级补充（不改变方案或授权状态）\n\n'
        '六组历史保存配置的外层 `ema_warmups=2000` 与内层 `yaml_cfg.ema.warmups=1000` 不同；所查 YAMLConfig 路径从内层构造 ModelEMA，指数系数的时间常数为 1000。历史代码身份未认证。若以后获授权重做，应在预检中核对两臂最终解析后的 EMA 配置和实际代码路径，而不能单看外层字段。此补充不启动、修改或扩大训练预算。详情见修订记录_20261103/第二十一轮训练超参核查_20261105/训练配置与调度语义说明.md。\n')
    replace_text(paths[4], prior[4] + proposal_note)
    write_new(receipt, {'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'revision': 21,
                        'manuscript': ref(CURRENT), 'previous_manuscript': ref(BACKUP),
                        'evidence_index': ref(index_json), 'visual_qa': ref(qa_path), 'original_preserved': True})
    print(json.dumps({'status': 'PROMOTED_AFTER_VISUAL_AND_EVIDENCE_QA', 'manuscript_sha256': digest(CURRENT),
                      'index_sha256': digest(index_json), 'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
