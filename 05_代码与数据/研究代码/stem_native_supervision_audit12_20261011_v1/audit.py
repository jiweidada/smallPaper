"""Read pinned native sources and actual primary bundles; never import the model."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path.cwd()
OUT = ROOT / 'outputs/stem_native_supervision_audit12_20261011_v1'
NATIVE = ROOT / '英文小论文/code/p2_candidate_seed1_full132_20261007_v1'
USER = ROOT / '英文小论文/code/user_backbone12_20261011_v1'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def defaults(path, name):
    tree = ast.parse(path.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    values = {}
    for arg, value in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults):
        try:
            values[arg.arg] = ast.literal_eval(value)
        except (ValueError, TypeError):
            values[arg.arg] = {'source_expression': ast.unparse(value)}
    return values


def main():
    binding = read(USER / 'source_binding.json')['native_files_sha256']
    sources = ['engine/deim/deim_criterion.py', 'engine/deim/dfine_decoder.py',
               'engine/deim/matcher.py', 'engine/deim/postprocessor.py']
    assert len(binding) == 468
    for name in sources:
        assert sha(NATIVE / name) == binding[name], name
    rows = []
    for seed in (1, 2):
        for arm in ('full', 'RGB'):
            if arm == 'full':
                packet = ROOT / f'outputs/stem_primary_collection_20261011_v1/seed{seed}_original_stem_v1/extracted'
            else:
                packet = ROOT / f'outputs/stem_component_primary_collection_20261011_v2/seed{seed}_highfreq_rgb_v1/extracted'
            manifest = read(packet / 'primary_manifest.json')
            assert manifest['identity']['seed'] == seed and manifest['completed_epochs'] == 12
            assert manifest['epoch_zero_based'] == 11 and manifest['evaluated_weights'] == 'EMA'
            assert not manifest['saved_EMA_independent_replay_included']
            for name in ('bundle/plan.json', 'bundle/receipt.json', 'bundle/resolved.yml',
                         'code/source_binding.json', 'code/training_protocol.json'):
                entry = manifest['files'][name]
                assert (packet / name).stat().st_size == entry['bytes']
                assert sha(packet / name) == entry['sha256'], name
            assert read(packet / 'code/source_binding.json')['native_files_sha256'] == binding
            receipt = read(packet / 'bundle/receipt.json')
            plan = read(packet / 'bundle/plan.json')
            cfg = read(packet / 'bundle/resolved.yml')
            assert receipt['identity'] == manifest['identity']
            assert receipt['resolved_sha256'] == sha(packet / 'bundle/resolved.yml')
            assert manifest['identity']['plan_sha256'] == sha(packet / 'bundle/plan.json')
            assert cfg['criterion'] == 'DEIMCriterion' and cfg['use_focal_loss'] is True
            assert cfg['DEIMCriterion'] == plan['historical_resolved_config']['DEIMCriterion']
            decoder = defaults(NATIVE / sources[1], 'DFINETransformer') | plan['model_recipe']['DFINETransformer']
            criterion = defaults(NATIVE / sources[0], 'DEIMCriterion') | cfg['DEIMCriterion']
            matcher = defaults(NATIVE / sources[2], 'HungarianMatcher') | cfg['DEIMCriterion']['matcher']
            post = defaults(NATIVE / sources[3], 'PostProcessor') | cfg['PostProcessor']
            assert criterion['losses'] == ['mal', 'boxes', 'local'] and criterion['gamma'] == 1.5
            assert criterion['boxes_weight_format'] is None and criterion['mal_alpha'] is None
            assert criterion['no_weight_vfl_epoch'] == -1 and criterion['use_uni_set'] is True
            assert matcher['change_matcher'] is False
            assert decoder['query_select_method'] == 'default' and decoder['num_queries'] == 300
            assert decoder['mgsa_groups'] == 0 and decoder['o2m_t'] is False
            assert decoder['num_layers'] == 3 and decoder['aux_loss'] is True
            assert post['num_top_queries'] == 300 and post['score_threshold'] == .001
            rows.append(dict(seed=seed, arm=arm, primary_manifest_sha256=sha(packet / 'primary_manifest.json'),
                resolved_sha256=sha(packet / 'bundle/resolved.yml'), plan_sha256=sha(packet / 'bundle/plan.json'),
                criterion=cfg['DEIMCriterion'], native_defaults=dict(
                    boxes_weight_format=criterion['boxes_weight_format'], mal_alpha=criterion['mal_alpha'],
                    no_weight_vfl_epoch=criterion['no_weight_vfl_epoch'], use_uni_set=criterion['use_uni_set'],
                    change_matcher=matcher['change_matcher'], aux_loss=decoder['aux_loss'],
                    mgsa_groups=decoder['mgsa_groups'], o2m_t=decoder['o2m_t'],
                    learn_query_content=decoder['learn_query_content']),
                postprocessor=dict(num_top_queries=post['num_top_queries'], score_threshold=post['score_threshold']),
                evaluator=cfg['evaluator']))
    evidence = []
    spans = {'engine/deim/deim_criterion.py': [(202, 229), (274, 306), (619, 629), (680, 692)],
             'engine/deim/dfine_decoder.py': [(803, 853), (884, 907), (1040, 1058), (1069, 1092)],
             'engine/deim/matcher.py': [(177, 200)],
             'engine/deim/postprocessor.py': [(88, 101), (122, 130)]}
    for name, ranges in spans.items():
        lines = (NATIVE / name).read_text(encoding='utf8').splitlines()
        for first, last in ranges:
            evidence.append(dict(path=str(NATIVE / name), source_sha256=binding[name], first_line=first,
                last_line=last, text='\n'.join(f'{i+1}: {lines[i]}' for i in range(first - 1, last))))
    OUT.mkdir(exist_ok=False)
    report = dict(status='PINNED_NATIVE_SUPERVISION_AND_BOTH_PRIMARY12_BUNDLES_AUDITED',
        source_sha256=sha(__file__), source_files_sha256={n: binding[n] for n in sources},
        user_source_binding_sha256=sha(USER / 'source_binding.json'), actual_primary_bundles=rows,
        findings=[
            'Active classification loss is MAL, not VFL merely because loss_vfl exists in weight_dict. Positive target is detached matched-box IoU**1.5; positive weight is1, negative weight is detached sigmoid(score)**1.5.',
            'Default encoder selection is top300 of max class logit over spatial locations; enc_bbox_head is applied only AFTER gathering selected memory. Native8400 regressed candidate boxes do not exist in this path.',
            'Encoder MAL auxiliary loss uses selected300 logits/boxes and its own Hungarian matches. Available dense logits are not an extra dense per-location target in the active MAL implementation.',
            'Discrete index selection is not a differentiable ranking objective. This does not mean unselected memory has zero gradients: decoder cross-attention still receives full memory.',
            'Final postprocessor chooses300 query/class entries from flattened sigmoid logits and applies score>.001. One query may contribute multiple class entries; this is distinct from encoder spatial selection.',
            'Hungarian matcher combines focal classification cost2, L1 cost5 and GIoU cost2. Box/local losses use union matches by default, whereas MAL uses the branch-specific matches. Training assignment is distinct from native evaluator score-ordered IoU matching.',
        ],
        interpretation='Existing IoU-aware classification supervision rules out the unsupported claim that the model entirely lacks box-quality classification targets. Primary native PR prefix shifts do not identify a unique faulty head or Stem component.',
        changed_next_action='Use pending exact saved-EMA diagnostic bindings to distinguish dense location score/selection from selected-box regression; retain all10 classes and both seeds. Do not invent a dense8400 box oracle or add a redundant quality loss before that evidence. Existing original-user tests remain unchanged.',
        evidence=evidence, independent_saved_EMA_verification_pending=True, causal_root_proven=False,
        novelty_proven=False, baseline_gate_changed=False, training_code_changed=False,
        new_model_forwards=0, new_evaluator_calls=0, new_matching_calls=0, new_optimizer_updates=0,
        new_training_epochs=0)
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    note = '''原生监督和筛选代码审查（两种子、增强Stem/RGB对照，主记录；独立EMA复核待完成）

实际配置 losses=[mal, boxes, local]，gamma=1.5。weight_dict里的loss_vfl并不代表VFL正在执行。
MAL正样本分类目标已经是匹配框IoU的1.5次方；不能用“缺少质量监督”直接解释AP下降。

需要区分三处：
1. 编码器在8400个位置的类别分数上，按每个位置最高类别分数选300个位置；只对这300个位置预测初始框。不存在原生8400个回归候选框。
2. 解码器用完整memory进行交叉注意力并迭代更新这300个查询。离散TopK不可微，不等于未入选memory没有梯度。
3. 最终将300个查询×10类的分数展平，再选300项并过滤score<=0.001。同一查询可能输出多个类别。最终排序与最初位置选择不能混为一谈。

训练分类使用各分支自己的匹配；boxes/local默认使用union匹配。训练Hungarian匹配与验证器按分数进行IoU匹配也不同。
种子1汽车IoU0.75的前1000个预测少29个TP，其中新增/丢失GT净贡献+1，两组都检出的GT在前1000内的进出净贡献−30。这只能定位到输出排序现象，不能唯一指认分类头、Stem或注意力。

后续等已部署的精确EMA诊断产物，区分位置分数/选入变化与入选框的回归变化；保留两种子和全部10类。当前SiLU、历史注意力、P2三项实验不改代码、不改损失、不改变12轮判定。
本次只有离线代码与已有主记录审查，新增训练/优化/模型前向/评估器调用均为0。
'''
    (OUT / '监督与筛选审查.txt').write_text(note, encoding='utf8')
    manifest = dict(files={n: sha(OUT / n) for n in ('report.json', '监督与筛选审查.txt')})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status=report['status'], primary_bundles=len(rows), source_files=len(sources),
        active_loss='mal', gamma=1.5, new_forwards=0, saved_EMA_pending=True)))


if __name__ == '__main__':
    main()
