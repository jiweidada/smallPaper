"""Register the archive configuration audit without changing the manuscript."""
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
ARCHIVE_SHA = '4706b4e4d7032e5264e981c9d59fc9771807cb8a72c5d814363d8054ab29216e'
PREFIX = 'DEIM_1572862469/'


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
    temp = path.with_name(path.name + '.rev26.tmp')
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
    report_path = HERE / 'archived_config_resolution.json'
    proof_path = HERE / 'independent_field_provenance.json'
    explanation = HERE / '迁移档案基线与完整模型口径核查.md'
    report, proof = read(report_path), read(proof_path)
    assert report['status'] == 'ARCHIVED_INCLUDE_CHAINS_RESOLVED_AS_INERT_DATA'
    assert proof['status'] == 'FIELD_OVERLAY_AND_EVALUATOR_DEFAULTS_INDEPENDENTLY_VERIFIED'
    assert proof['resolution_report_sha256'] == digest(report_path)
    assert proof['archive_sha256'] == report['archive']['sha256'] == digest(Path(report['archive']['path'])) == ARCHIVE_SHA
    previous = INDEX / '主要结论证据索引_第二十五轮.json'
    old_index = read(previous)
    assert old_index['revision'] == 25 and old_index['manuscript']['sha256'] == CURRENT_SHA
    for value in old_index.values():
        if isinstance(value, dict) and set(value) >= {'path', 'sha256'}:
            assert digest(Path(value['path'])) == value['sha256']
    old_report = read(Path(old_index['archive_comparison']['path']))
    assert old_report['archive']['sha256'] == ARCHIVE_SHA
    assert not report['historical_execution_authenticated']
    assert all(report[k] == 0 for k in ['model_instantiations', 'model_forwards', 'evaluator_executions', 'optimizer_updates'])
    assert set(report['configurations']) == {'baseline_n', 'baseline_s', 'full_n', 'full_s'}
    for label, cfg in report['configurations'].items():
        fields = proof['fields'][label]['fields']
        assert len(cfg['include_visits']) == proof['fields'][label]['applied_document_count'] == 9
        for visit in cfg['include_visits']:
            snapshot = HERE / 'source_snapshot' / visit['member'].removeprefix(PREFIX)
            assert digest(snapshot) == visit['sha256']
        assert fields['val_dataloader.dataset.ann_file']['value'] == cfg['val_ann_file']
        assert cfg['evaluator']['type'] == 'CocoEvaluator'
        if label.startswith('baseline'):
            assert cfg['val_ann_file'].endswith('VisDrone2019-DET-test-dev/annotations/test.json')
            assert cfg['evaluator']['using_tiny_metrice'] is True and cfg['evaluator']['max_dets'] == 300
        else:
            assert cfg['val_ann_file'].endswith('VisDrone2019-DET-val/annotations/val.json')
            assert 'using_tiny_metrice' not in cfg['evaluator'] and 'max_dets' not in cfg['evaluator']
            assert fields['evaluator.using_tiny_metrice']['value'] == '<absent>'
            assert fields['evaluator.max_dets']['value'] == '<absent>'
    assert proof['evaluator_constructor_defaults']['using_tiny_metrice'] is False
    assert proof['evaluator_constructor_defaults']['max_dets'] == 100
    assert 'using_tiny_metrice' not in proof['evaluator_shared_fields']
    assert 'max_dets' not in proof['evaluator_shared_fields']
    for info in proof['migration_records'].values():
        assert digest(Path(info['path'])) == digest(Path(info['snapshot'])) == info['sha256']
    for name, filename in [('workspace', 'workspace.py'), ('yaml_config', 'yaml_config.py')]:
        assert digest(HERE / 'source_snapshot/engine/core' / filename) == proof['factory_and_config_source_sha256'][name]
    assert digest(HERE / 'source_snapshot/engine/data/dataset/coco_eval.py') == report['evaluator_constructor_source']['sha256']
    catalog = REC / '第二十四轮会议稿论述复核_20261105/manuscript_version_locations.json'
    index_json = INDEX / '主要结论证据索引_第二十六轮.json'
    index_txt = INDEX / '主要结论证据索引_第二十六轮.txt'
    receipt = HERE / 'config_resolution_receipt.json'
    assert not any(path.exists() for path in [index_json, index_txt, receipt])
    paths = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt', REC / '目标验收与阻塞记录.txt',
             REMOTE / '结果并入与验收清单.txt', REMOTE / '公平主对照重做提案_20261105.md']
    prior = [path.read_text(encoding='utf-8-sig') for path in paths]
    assert prior[3].count('当前稿件SHA256：' + CURRENT_SHA) == 1
    for i, path in enumerate(paths):
        saved = HERE / f'第二十六轮更新前文本_{i}.txt'
        assert not saved.exists()
        shutil.copy2(path, saved)
        assert digest(saved) == digest(path)
    index = {'revision': 26, 'status': 'ARCHIVED_EVALUATION_CONFIG_DIFFERENCES_DOCUMENTED',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'manuscript': ref(CURRENT),
             'previous_index': ref(previous), 'version_locations': ref(catalog),
             'config_resolution': ref(report_path), 'independent_verification': ref(proof_path),
             'explanation': ref(explanation),
             'finding': 'In the archived n/s baseline entries, static includes resolve to test-dev and explicit evaluator tiny=true, max_dets=300. The archived n/s enhanced entries resolve to val and omit those fields; the inspected constructor defaults are false and 100.',
             'limits': 'Without CLI/runtime overrides, direct use of the archived entries would have different evaluation conditions. This does not establish which configuration, source, or data generated any historical metric. The evaluator max_dets field is not evidence of the model postprocessor or official benchmark protocol.',
             'models_run': 0, 'forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
             'submission_ready': False, 'remaining': 'The paired BASE/FULL comparison remains incomplete; any new training requires explicit authorization.'}
    write_new(index_json, index)
    index_txt.write_text('主要结论证据索引 第二十六轮\n当前稿件未改，SHA256：' + CURRENT_SHA + '\n'
        '迁移档案的基线 n/s 入口静态解析为 test-dev、评价器 tiny=true 和 max_dets=300；完整模型 n/s 入口解析为 val，后两项未设置，所查构造函数默认 false 和 100。\n'
        '此结果假定无运行时覆盖，不能证明历史指标实际采用了这些入口或混用了数据划分。详见：' + str(explanation) + '\n'
        '机器可读索引：' + str(index_json) + '\n本轮无模型、推理、评价或训练；公平 BASE/FULL 仍不完整，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第二十六轮迁移配置口径核查】\n'
        '当前稿件未改：' + str(CURRENT) + '\nSHA256：' + CURRENT_SHA + '\n'
        '档案内基线与完整模型入口的静态评价划分和评价器字段不同；在无覆盖情况下直接比较不匹配，但历史指标是否采用这些入口未获证实。\n'
        '详见修订记录_20261103/第二十六轮迁移配置口径核查_20261105/迁移档案基线与完整模型口径核查.md；最新证据索引为主要结论证据索引_第二十六轮.txt及JSON。\n'
        '本轮未运行模型、评价或训练。公平主对照仍不完整，尚不能投稿。以下内容按时间保留，旧状态以本段及最新记录为准。\n\n')
    replace_text(paths[0], header + prior[0])
    addition = ('\n第二十六轮：迁移档案评价口径核查（2026-11-05）\n'
        '四个档案入口的 include 按顺序静态解析：基线 n/s 指向 test-dev，评价器 tiny=true、max_dets=300；完整模型 n/s 指向 val，两个字段未设置，所查构造函数默认 false、100。无覆盖时直接使用不构成口径匹配的比较；这些配置未绑定历史指标，不证明实际历史运行混用。迁移文档还记录了将自定义文件复制进新版代码的过程，不能认证历史训练源码。详情见第二十六轮迁移配置口径核查_20261105。\n'
        '论文未改，SHA256：' + CURRENT_SHA + '。无模型、推理、评价或训练；公平 BASE/FULL 仍不完整，尚不能投稿。\n')
    for i in [1, 2]:
        replace_text(paths[i], prior[i] + addition)
    lead = ('最新迁移配置核查（2026-11-05）：档案基线与完整模型入口分别静态指向 test-dev 与 val，评价器 tiny/max_dets 也有差异；无覆盖时不适合直接作匹配比较。未证实这些入口生成了历史指标。详见修订记录_20261103/第二十六轮迁移配置口径核查_20261105。\n')
    replace_text(paths[3], lead + prior[3])
    proposal = ('\n## 旧迁移配置的口径差异（不改变方案或授权状态）\n\n'
        '旧迁移档案的基线 n/s 入口静态指向 test-dev，评价器 tiny=true、max_dets=300；完整模型 n/s 入口指向 val，后两项未设置，所查评价器构造函数默认 false、100。无命令行或运行时覆盖时不能直接匹配比较；这些配置尚未与历史指标绑定。未来若另获授权重做，预检应核对两臂解析出的评价数据划分、标注与数据身份、评价器 tiny/max_dets 和模型后处理设置，并留存实际参数与代码版本；不据此改变既定方案或扩大实验预算。详见修订记录_20261103/第二十六轮迁移配置口径核查_20261105/迁移档案基线与完整模型口径核查.md。\n')
    replace_text(paths[4], prior[4] + proposal)
    write_new(receipt, {'status': 'CONFIG_DIFFERENCES_REGISTERED_MANUSCRIPT_UNCHANGED', 'revision': 26,
                        'manuscript': ref(CURRENT), 'index': ref(index_json), 'explanation': ref(explanation),
                        'original_preserved': True, 'submission_ready': False})
    print(json.dumps({'status': 'CONFIG_DIFFERENCES_REGISTERED_MANUSCRIPT_UNCHANGED',
                      'manuscript_sha256': digest(CURRENT), 'index_sha256': digest(index_json),
                      'submission_ready': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
