"""Account for the parameter changes between the actual frozen BASE and FULL."""
import ast
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
digest = lambda p: sha256(p.read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(1)
    module = HERE / 'NO_MSCA'
    parent = read(HERE.parent / 'execution_protocol.json')
    assert digest(HERE.parent / 'execution_protocol.json') == '8d495467c5ec74ac57f8e97f730cd345bf72360a2ee2ce30197880c05079902b'
    frozen = {f['path']: f['sha256'] for f in parent['frozen_files']}
    config_paths = {arm: HERE.parent / 'remote_evidence/execution_configs' / f'{arm}.yml' for arm in ['BASE', 'FULL']}
    configs = {}
    for arm, path in config_paths.items():
        assert digest(path) == frozen[f'execution_configs/{arm}.yml']
        configs[arm] = read(path)
    base_cfg = configs['BASE']['HGNetv2']
    full_cfg = configs['FULL']['HGNetv2_MSCA_EnhancedStem']
    assert all(c['freeze_at'] == -1 and c['freeze_norm'] is False and c['use_lab'] is True
               for c in [base_cfg, full_cfg])
    assert 'agg' not in base_cfg and 'hierarchical_attention' not in base_cfg
    assert full_cfg['agg'] == ['ese', 'simam', 'msca_lite', 'msca_dilated']
    sources = {name: module / 'source_snapshot/engine/backbone' / name for name in ['hgnetv2.py', 'hgnetv2_MSCA_EnhancedStem.py']}
    for name, path in sources.items():
        assert digest(path) == frozen[f'source_snapshot/engine/backbone/{name}']
    tree = ast.parse(sources['hgnetv2.py'].read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HGNetv2')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    defaults = {arg.arg: ast.literal_eval(default) for arg, default in zip(init.args.args[-len(init.args.defaults):], init.args.defaults)}
    assert defaults['agg'] == 'se' and defaults['hierarchical_attention'] is False
    paths = {arm: module / 'reference' / f'{arm}.initial.pt' for arm in ['BASE', 'FULL']}
    plan = read(module / 'execution_protocol.json')
    states = {}
    for arm, path in paths.items():
        assert digest(path) == plan['initialization']['reference_initial_sha256'][arm]
        states[arm] = torch.load(path, map_location='cpu', weights_only=True)
    outside = {arm: {k: list(t.shape) for k, t in state.items() if not k.startswith('backbone.')}
               for arm, state in states.items()}
    assert outside['BASE'] == outside['FULL']
    buffers = ('.running_mean', '.running_var', '.num_batches_tracked')
    categories = {}
    projection_blocks = {}
    tensor_counts = {}
    for arm, state in states.items():
        counts = {'stem': 0, 'aggregation_projections': 0, 'stage1_ESE': 0, 'stage4_MSCA_Dilated': 0, 'other_backbone': 0}
        blocks = {}
        assigned = []
        for key, tensor in state.items():
            if not key.startswith('backbone.') or key.endswith(buffers):
                continue
            if key.startswith('backbone.stem.'):
                category = 'stem'
            elif '.aggregation.' in key:
                match = re.match(r'backbone\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.', key)
                assert match, key
                stage, block, position = map(int, match.groups())
                if arm == 'BASE' or position == 0:
                    assert position in [0, 1]
                    category = 'aggregation_projections'
                    label = f'stage{stage+1}.block{block+1}'
                    blocks[label] = blocks.get(label, 0) + tensor.numel()
                else:
                    assert position == 1
                    assert stage in [0, 3], key
                    category = 'stage1_ESE' if stage == 0 else 'stage4_MSCA_Dilated'
            else:
                category = 'other_backbone'
            counts[category] += tensor.numel()
            assigned.append(key)
        categories[arm] = counts
        projection_blocks[arm] = blocks
        tensor_counts[arm] = len(assigned)
    assert categories['BASE']['stem'] == 6474 and categories['FULL']['stem'] == 11799
    assert categories['BASE']['stage1_ESE'] == categories['BASE']['stage4_MSCA_Dilated'] == 0
    assert categories['FULL']['stage1_ESE'] == 4160
    assert categories['FULL']['stage4_MSCA_Dilated'] == 3841280
    assert categories['BASE']['other_backbone'] == categories['FULL']['other_backbone']
    assert categories['BASE']['aggregation_projections'] == 1604564
    assert categories['FULL']['aggregation_projections'] == 1557130
    differences = {k: categories['FULL'][k] - categories['BASE'][k] for k in categories['BASE']}
    assert differences == {'stem': 5325, 'aggregation_projections': -47434,
                            'stage1_ESE': 4160, 'stage4_MSCA_Dilated': 3841280, 'other_backbone': 0}
    pair_path = HERE.parent / '断点续训_20261105/BASE_FULL_pair_verified_20261108.json'
    totals = read(pair_path)['shared_conditions']['trainable_parameters']
    assert sum(differences.values()) == totals['FULL'] - totals['BASE'] == 3803331
    by_block = {k: projection_blocks['FULL'][k] - projection_blocks['BASE'][k] for k in projection_blocks['BASE']}
    assert by_block == {'stage1.block1': -66, 'stage2.block1': -12546,
                        'stage3.block1': -16898, 'stage3.block2': 48638, 'stage4.block1': -66562}
    result = {
        'status': 'FROZEN_BASE_FULL_BACKBONE_CAPACITY_CHANGES_VERIFIED',
        'observed_local_utc': datetime.now(timezone.utc).isoformat(),
        'parent_protocol_sha256': digest(HERE.parent / 'execution_protocol.json'),
        'input_initial_sha256': {arm: digest(p) for arm, p in paths.items()},
        'source_sha256': {name: digest(p) for name, p in sources.items()},
        'config_sha256': {arm: digest(p) for arm, p in config_paths.items()},
        'BASE_actual_default_aggregation': 'se: two ConvBNAct projections; not an additional standalone ESE gate',
        'FULL_stage_aggregation': full_cfg['agg'],
        'trainable_backbone_counts_by_category': categories,
        'FULL_minus_BASE_parameters_by_category': differences,
        'aggregation_projection_counts_by_block': projection_blocks,
        'FULL_minus_BASE_projection_parameters_by_block': by_block,
        'total_trainable_parameters': totals, 'FULL_minus_BASE_total': sum(differences.values()),
        'unchanged_outside_backbone_state_names_and_shapes': True,
        'unchanged_other_backbone_parameter_count': True,
        'scope': 'Counts the changed, unfrozen backbone modules from saved initial tensors, excluding batch-normalization running buffers. Shapes outside the backbone match. The component differences reconcile exactly with genuine model preflight totals. No model was constructed or run.',
        'interpretation': 'Stem +5,325; projection paths -47,434 overall, with one block increasing; stage-1 ESE +4,160; stage-4 Dilated +3,841,280. SimAM and the two Lite gates add no learned parameters. Activation changes add none. These are structural counts, not a decomposition of AP gains or independent module effects.',
        'comparison_limit': 'BASE is not FULL with both MPS and MSCA removed: it also has different projections, stage-1/2 gates and activation choices. Existing conditional removals retain those other FULL settings.',
        'model_constructed': False, 'model_forwards': 0, 'evaluator_runs': 0, 'optimizer_updates': 0}
    output = HERE / 'base_full_capacity_changes_20261108.json'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'BASE_FULL参数变化核对_20261108.txt').open('x', encoding='utf-8') as stream:
        stream.write('BASE/FULL 实际结构参数变化核对\n\n'
                     '读取已留存初始化张量、冻结配置和源码，不构建或运行模型。BASE 默认 agg=se 在该代码中是两个 ConvBNAct 投影，不是独立 ESE 门；FULL 使用 ESE/SimAM/Lite/Dilated。\n'
                     'FULL−BASE 可训练参数变化：MPS stem +5,325；五个 HG 块聚合投影合计 −47,434；第一阶段 ESE +4,160；第四阶段 Dilated +3,841,280。'
                     '总和 +3,803,331，精确等于 7,540,942−3,737,611。其余骨干参数量不变，骨干之外状态名称与形状相同。\n'
                     '投影逐块差为 −66、−12,546、−16,898、+48,638、−66,562，不声称每一块都减参。两个 Lite、SimAM 与激活变化不增加可训练参数。\n'
                     '这是参数变化的来源核对，不是精度提升的可加分解，也不证明机制独立于容量。BASE 仍不等同于 MPS/MSCA 双移除组；两个条件移除保留 FULL 的其他设置。\n')
    print(json.dumps({k: result[k] for k in ['status', 'FULL_minus_BASE_parameters_by_category',
                                            'FULL_minus_BASE_total', 'model_constructed', 'model_forwards']}))

if __name__ == '__main__':
    main()
