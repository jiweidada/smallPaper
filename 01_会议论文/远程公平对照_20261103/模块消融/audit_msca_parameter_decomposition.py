"""Count the removed gate's saved tensors without constructing a model."""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(1)
    root = HERE / 'NO_MSCA'
    protocol_path = root / 'execution_protocol.json'
    assert digest(protocol_path) == '3e5e8937e0889aa360ffe3f7acb722ad1ab5f8e6c66cd6b3991a71516212dcfc'
    protocol = read(protocol_path)
    full_path = root / 'reference/FULL.initial.pt'
    removed_path = root / 'inputs/initial.pt'
    assert digest(full_path) == protocol['initialization']['reference_initial_sha256']['FULL']
    assert digest(removed_path) == protocol['probe_files']['initial.pt']
    source_path = root / 'source_snapshot/engine/extre_module/custom_nn/attention/MSCA.py'
    source_sha = '04a3ebb82954c060843493a27a582694d9f056e7badc0629ffe0718674a6e888'
    assert digest(source_path) == source_sha
    source_tree = ast.parse(source_path.read_text(encoding='utf-8'))
    classes = {n.name: n for n in source_tree.body if isinstance(n, ast.ClassDef)}
    lite = classes['MSCA_Lite']
    lite_init = next(n for n in lite.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    layer_calls = [ast.unparse(n.func) for n in ast.walk(lite_init) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == 'nn']
    assert layer_calls == ['nn.Sigmoid']
    full = torch.load(full_path, map_location='cpu', weights_only=True)
    removed = torch.load(removed_path, map_location='cpu', weights_only=True)
    assert not removed.keys() - full.keys()
    assert all(torch.equal(v, full[k]) for k, v in removed.items())
    dropped = sorted(full.keys() - removed.keys())
    prefix = 'backbone.stages.3.blocks.0.aggregation.1.'
    assert dropped and all(k.startswith(prefix) for k in dropped)
    buffers = [k for k in dropped if k.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))]
    parameters = [k for k in dropped if k not in buffers]
    groups = {}
    for name in parameters:
        component = name[len(prefix):].split('.')[0]
        assert component in ['multi_scale_convs', 'context_conv', 'channel_att']
        groups[component] = groups.get(component, 0) + full[name].numel()
    assert groups == {'channel_att': 263296, 'context_conv': 3146752, 'multi_scale_convs': 431232}
    total = sum(groups.values())
    pair_path = HERE.parent / '断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
    full_count = read(pair_path)['shared_conditions']['trainable_parameters']['FULL']
    removed_count = protocol['initialization']['trainable_parameters']
    assert total == full_count - removed_count == 3841280
    shapes = {k[len(prefix):]: list(full[k].shape) for k in parameters}
    assert shapes['context_conv.weight'] == [1024, 3072, 1, 1]
    assert shapes['context_conv.bias'] == [1024]
    for i in range(3):
        assert shapes[f'multi_scale_convs.{i}.0.weight'] == [128, 1024, 1, 1]
        assert shapes[f'multi_scale_convs.{i}.3.weight'] == [1024, 1, 3, 3]
    c, r = 1024, 128
    analytical = {
        'multi_scale_convs': 3 * ((c*r+r) + 2*r + (9*c+c) + 2*c),
        'context_conv': 3*c*c+c,
        'channel_att': (c*r+r) + (r*c+c),
    }
    assert analytical == groups
    lite_paths = protocol['initialization']['identity_gate_paths'][:2]
    assert lite_paths == ['backbone.stages.2.blocks.0.aggregation.1', 'backbone.stages.2.blocks.1.aggregation.1']
    assert all(not any(k.startswith(path + '.') for k in full) for path in lite_paths)
    result = {
        'status': 'FROZEN_MSCA_GATE_PARAMETER_DECOMPOSITION_VERIFIED',
        'observed_local_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_sha,
        'source_class_line_ranges': {n: [classes[n].lineno, classes[n].end_lineno]
                                     for n in ['MSCA_Dilated', 'MSCA_Lite']},
        'protocol_sha256': digest(protocol_path),
        'input_checkpoint_sha256': {'FULL_initial': digest(full_path), 'NO_MSCA_initial': digest(removed_path)},
        'full_trainable_parameters': full_count, 'NO_MSCA_trainable_parameters': removed_count,
        'removed_gate_trainable_parameters': total,
        'trainable_parameters_by_component': groups,
        'analytical_counts_C1024_reduced128': analytical,
        'context_projection_fraction_of_removed_parameters': groups['context_conv'] / total,
        'removed_gate_fraction_of_full_parameters': total / full_count,
        'removed_parameter_tensor_shapes': shapes,
        'excluded_batchnorm_buffer_names': buffers,
        'lite_gate_paths': lite_paths, 'lite_gate_trainable_parameters': 0,
        'lite_evidence_scope': 'The saved states have no entries for the two Lite paths; its frozen constructor only creates Sigmoid. Frozen adapter and genuine preflight separately bind the actual gate replacements.',
        'interpretation': 'All additional learned parameters removed by NO_MSCA are in the stage-4 Dilated gate. Its dense 3072-to-1024 context projection dominates its trainable capacity. This is a parameter accounting result, not a performance, FLOP, latency or GPU-memory measurement.',
        'model_constructed': False, 'model_forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0,
        'NO_MSCA_outcome_pending': True}
    output = HERE / 'msca_parameter_decomposition_20261108.json'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'MSCA参数来源分解_20261108.txt').open('x', encoding='utf-8') as stream:
        stream.write('MSCA 冻结权重参数来源分解\n\n'
                     '读取 FULL/NO_MSCA 留存初始化张量并结合冻结源码静态计数；没有构建或运行模型，没有新实验。\n'
                     '两处 Lite 门没有可训练参数。NO_MSCA 移除的全部 3,841,280 个可训练参数来自第四阶段 Dilated 门，恰等于 7,540,942−3,699,662。\n'
                     '其中：三条卷积分支合计 431,232；3C→C 上下文投影 3,146,752；通道门 263,296。'
                     '投影占该门参数约 ' + f'{100 * groups["context_conv"] / total:.2f}' + '%。BN 运行均值、方差和计数器作为缓冲区排除。解析公式与逐张量计数完全一致。\n'
                     '此证据解释参数容量来源，不证明精度贡献，也不能代替 FLOPs、时延、显存或容量匹配对照。NO_MSCA 完整性能终点仍待训练完成后核验。\n')
    print(json.dumps({k: result[k] for k in ['status', 'removed_gate_trainable_parameters',
                                            'trainable_parameters_by_component',
                                            'context_projection_fraction_of_removed_parameters',
                                            'model_constructed', 'model_forwards']}))

if __name__ == '__main__':
    main()
