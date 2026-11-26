"""Record the local historical-run audit without editing the manuscript."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil

HERE = Path(__file__).resolve().parent
REC = HERE.parent
PAPER = REC.parent
INDEX = REC / '主要结论证据索引'
DOC = PAPER / '稿件/MPSMA-DEIM_conference_EN_revised.docx'
DOC_SHA = '617ffd2024d42affb7f6b66f88cdca67204a0de3fef97946ffebf0c90f896984'

def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def entry(path):
    return {'path': str(path), 'sha256': digest(path)}

def main():
    assert digest(DOC) == DOC_SHA
    output_json = INDEX / '主要结论证据索引_第十六轮.json'
    output_text = INDEX / '主要结论证据索引_第十六轮.txt'
    assert not output_json.exists() and not output_text.exists()
    files_to_update = [PAPER / '当前修订入口.txt', REC / '修订说明与未解决问题.txt',
                       REC / '目标验收与阻塞记录.txt',
                       PAPER / '远程公平对照_20261103/公平主对照重做提案_20261105.md']
    for path in files_to_update:
        assert path.is_file() and not (HERE / ('更新前_' + path.name)).exists()
    for path in files_to_update:
        shutil.copy2(path, HERE / ('更新前_' + path.name))
    evidence = {
        'revision': 16,
        'status': 'HISTORICAL_OUTPUTS_INVENTORIED_NO_MATCHED_BASELINE_FOUND',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'manuscript': entry(DOC),
        'manuscript_changed': False,
        'previous_indices': [entry(INDEX / '主要结论证据索引_第十五轮.json'),
                             entry(INDEX / '主要结论证据索引_第十五轮.txt')],
        'inventory': entry(HERE / 'existing_outputs_inventory.json'),
        'candidate_diff': entry(HERE / 'candidate_metadata_diff.txt'),
        'explanation': entry(HERE / '历史基线检索说明.md'),
        'scope': '23 immediate directories in the local DEIM outputs root, including .idea; missing args leave conditions unknown. No other roots or remote directories searched.',
        'finding': 'The same-class SimAM run declares simam aggregation and resume and has a gap at epoch index 266; the plain n batch-4 run has 160 configured epochs, whereas the 320-epoch plain n run has batch 16.',
        'limits': 'A config declaration does not authenticate historical code, data bytes, initialization or metric/checkpoint binding. A missing log row does not prove a missing training epoch.',
        'model_instantiations': 0, 'model_forwards': 0, 'evaluator_reruns': 0, 'optimizer_updates': 0,
        'submission_ready': False,
        'remaining': 'The BASE/FULL fair pair remains incomplete; the authorized diagnostic is not a substitute. New training requires explicit authorization.'}
    with output_json.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(evidence, out, ensure_ascii=False, indent=2)
        out.write('\n')
    output_text.write_text(
        '主要结论证据索引 第十六轮\n'
        '稿件未修改，SHA256：' + DOC_SHA + '\n'
        '扩展检索本机DEIM/outputs直属23个目录；缺少配置者条件未知。新增发现SimAM同类B0存档配置了simam及resume，日志缺零基266；批量4原始n存档仅160轮，320轮原始n存档批量16。\n'
        '仅在已查范围内未发现可认定的320轮同条件原始n基线；历史代码、数据及指标-权重绑定仍未完全认证。\n'
        '详情：' + str(HERE / '历史基线检索说明.md') + '\n'
        '机器可读索引：' + str(output_json) + '\n'
        '其余证据继承第十五轮。无新增实验；公平BASE/FULL对照仍缺，尚不能投稿。\n', encoding='utf-8')
    header = ('【最新证据入口：2026-11-05，第十六轮历史基线检索】\n'
              '当前修订稿未变：' + str(DOC) + '\nSHA256：' + DOC_SHA + '\n'
              '本机原项目outputs直属23个目录已做只读清点；新增SimAM存档配置指定simam和resume且日志缺零基266，原始n的批量4存档仅160轮，320轮存档批量16；未找到可认定的320轮同条件原始n基线。缺配置目录条件未知，范围不含其他目录及服务器。\n'
              '详见修订记录_20261103/第十六轮历史基线检索_20261105/历史基线检索说明.md；最新证据索引为主要结论证据索引_第十六轮.txt及JSON。\n'
              '无新增实验。已授权BASE诊断不构成公平主对照；重做方案仍待明确授权，尚不能投稿。以下版本和运行状态均为历史。\n\n')
    entry_path = files_to_update[0]
    old_text = entry_path.read_text(encoding='utf-8-sig')
    entry_path.write_text(header + old_text, encoding='utf-8')
    append = ('\n第十六轮：历史基线扩展检索（2026-11-05）\n'
              '只读枚举本机DEIM/outputs直属23目录，其中18个有args、20个有log；新增核实SimAM同类B0候选声明simam和resume，日志缺零基266；批量4原始n配置为160轮，320轮原始n配置批量16，未发现可认定的320轮同条件原始n基线。缺日志行不证明该轮未训练；其他根目录及服务器不在检索范围。\n'
              '清单、配置差异、哈希及限制见第十六轮历史基线检索_20261105；其余证据继承第十五轮。未构建模型、前向、评价或训练；稿件未改，SHA256：' + DOC_SHA + '。公平BASE/FULL仍缺，尚不能投稿。\n')
    for path in files_to_update[1:3]:
        with path.open('a', encoding='utf-8', newline='\n') as out:
            out.write(append)
    with files_to_update[3].open('a', encoding='utf-8', newline='\n') as out:
        out.write('\n## 历史存档扩展检索补充（未改变授权状态）\n\n'
                  '本机原项目 outputs 直属目录中，同类 B0 的 SimAM 候选配置包含 simam 和 resume、日志缺零基 266；原始 n 的 batch 4 存档只配置 160 轮，320 轮存档 batch 16。已查范围未找到可认定的 320 轮同条件原始 n 基线；缺少配置或其他目录的情况仍未知。详情见修订记录_20261103/第十六轮历史基线检索_20261105/历史基线检索说明.md。本补充没有新增实验，不改变重做方案或待明确授权状态。\n')
    assert digest(DOC) == DOC_SHA
    print(json.dumps({'index': str(output_json), 'manuscript_sha256': digest(DOC),
                      'inventory_sha256': digest(HERE / 'existing_outputs_inventory.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
