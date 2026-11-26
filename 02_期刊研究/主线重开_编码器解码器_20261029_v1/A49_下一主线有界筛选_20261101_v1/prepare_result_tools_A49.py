"""Prepare result archival/judgment tools only; never launch training or inference."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'A48_新机制候选与主线收敛_20261101_v1'
EXPECTED = 'd3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def replace_required(text, old, new):
    if old not in text:
        raise RuntimeError('Required adaptation missing: ' + old)
    return text.replace(old, new)

old_protocol = '2294d4e351e42e8760969a3ea06a3f6432591eeb70d17328b3cccec1cd5f4273'
assert sha(HERE / 'protocol_FROZEN.json') == EXPECTED
record = {'scope': 'Postprocessing tool preparation; original training protocol/code untouched. No GPU/model/optimizer work.',
          'sources': {}, 'outputs': {}, 'model_forwards_added': 0, 'optimizer_updates_added': 0}
for name in ['collect_completed_arm_A48.py', 'verify_completed_arm_local.py', 'judge_fixed_endpoint_A48.py']:
    src = OLD / name
    text = src.read_text(encoding='utf-8-sig')
    record['sources'][str(src)] = sha(src)
    dest_name = name.replace('A48', 'A49')
    if name == 'collect_completed_arm_A48.py':
        text = replace_required(text, '/root/uav_a48_20261101_v1', '/root/uav_a49_20261101_v1')
        text = replace_required(text, old_protocol, EXPECTED)
        text = replace_required(text,
            "assert result['full_model_forwards']==10159 and terminal['exit_code']==0",
            "assert result['full_model_forwards']==10159 and terminal['exit_code']==0\n"
            "    assert result['native_inference_no_mask_reads'] is True\n"
            "    assert result['extra_mask_core_reads']==(38832 if a=='Q' else 0)")
    if name == 'verify_completed_arm_local.py':
        text = replace_required(text,
            "assert json.loads((folder/'exit.json').read_text())['exit_code']==0",
            "assert json.loads((folder/'exit.json').read_text())['exit_code']==0\n"
            "    assert result['protocol_sha256']=='" + EXPECTED + "'\n"
            "    assert result['native_inference_no_mask_reads'] is True\n"
            "    assert result['extra_mask_core_reads']==(38832 if a=='Q' else 0)")
        text = replace_required(text,
            "'payload_files_verified':len(manifest['files'])",
            "'extra_mask_core_reads':result['extra_mask_core_reads'],'native_inference_no_mask_reads':True,\n"
            "      'payload_files_verified':len(manifest['files'])")
    if name == 'judge_fixed_endpoint_A48.py':
        text = replace_required(text, old_protocol, EXPECTED)
        text = text.replace('A48', 'A49')
        text = replace_required(text, "'all_batches_targets_sha256'", "'all_batches_targets_including_ignore_tiles_sha256'")
        text = replace_required(text, "protocol['practical_gates_pp']", "protocol['gates_pp']")
        text = replace_required(text,
            "assert result['full_model_forwards'] == 10159",
            "assert result['full_model_forwards'] == 10159\n"
            "        assert result['native_inference_no_mask_reads'] is True\n"
            "        assert result['extra_mask_core_reads'] == (38832 if arm == 'Q' else 0)")
        text = replace_required(text,
            "'capacity_model_forwards': 12, 'capacity_optimizer_updates': 9",
            "'capacity_model_forwards': 12, 'capacity_optimizer_updates': 9,\n"
            "                         'formal_extra_mask_core_reads': sum(r['extra_mask_core_reads'] for r in arms.values()),\n"
            "                         'capacity_extra_mask_core_reads': 12")
        text = replace_required(text,
            'Equal-group weighted normalization is prior art; gains alone do not establish novelty.',
            'Ordinary known-region ignoring and negative risk correction are prior art; read-support proxy benefit and novelty remain unproven.')
        text = replace_required(text, 'lambda/位置/seed', '阈值/位置/seed')
        text = replace_required(text,
            '已有加权归一化与等组贡献先例，最多主张有区别的任务适配；仍须评价与最近工作的差异。',
            '普通忽略区处理与背景风险校正已有先例；Q相对M的增量及最近工作的区别仍须单独判断。')
    ast.parse(text)
    dest = HERE / dest_name
    if dest.exists():
        if dest.read_text(encoding='utf-8') != text:
            raise RuntimeError('Do not overwrite changed tool: ' + str(dest))
    else:
        dest.write_text(text, encoding='utf-8')
    record['outputs'][str(dest)] = sha(dest)
record['frozen_training_protocol_sha256'] = sha(HERE / 'protocol_FROZEN.json')
out = HERE / 'result_tools_preparation_v1.json'
if out.exists():
    raise RuntimeError('Already prepared; preserve receipt.')
out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'scripts': len(record['outputs']),
                  'model_forwards_added': 0, 'optimizer_updates_added': 0, 'receipt': str(out)}, ensure_ascii=False))
