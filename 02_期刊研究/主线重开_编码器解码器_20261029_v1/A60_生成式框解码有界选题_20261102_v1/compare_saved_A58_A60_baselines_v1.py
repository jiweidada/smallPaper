"""Describe existing B8 pipeline outputs; no new matching, filtering or inference."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import math
import statistics

P = Path(__file__).resolve().parent
SCOPE = P / 'A58_A60_saved_baseline_comparison_scope_FROZEN_v1.json'
SUPPLEMENT = P / 'A58_A60_saved_baseline_comparison_scope_supplement_v1.json'
RESULT = P / 'A58_A60_saved_baseline_comparison_v1.json'
REPORT = P / '12_现有ECDet与当前生成式基线的可比性.txt'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not RESULT.exists() and not REPORT.exists()
    scope, supplement = read(SCOPE), read(SUPPLEMENT)
    assert supplement['parent_scope_sha256'] == sha(SCOPE)
    files = {}
    for mapping in [scope['data'], supplement['data']]:
        for name, item in mapping.items():
            path = Path(item['path'])
            assert path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
            files[name] = path
    expectations = read(files['A58_asset_expectations'])
    assert sha(files['A58_runtime_common']) == expectations['runtime_common_sha256']
    assert sha(files['A58_native_postprocessor']) == expectations['native_source_sha256']['engine/edgecrafter/postprocessor.py']
    tree = ast.parse(files['A58_native_postprocessor'].read_text(encoding='utf-8-sig'))
    post = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PostProcessor')
    forward = next(node for node in post.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    assert not any(isinstance(node, (ast.Name, ast.Attribute)) and
                   (getattr(node, 'id', '') in ['nms', 'batched_nms'] or getattr(node, 'attr', '') in ['nms', 'batched_nms'])
                   for node in ast.walk(forward))
    a60_source = files['A60_evaluator_source'].read_text(encoding='utf-8-sig')
    assert 'batched_nms(b,scores,labels,.6)' in a60_source

    gt58, gt60 = read(files['A58_B8_GT']), read(files['A60_B8_GT'])
    equality = {key: gt58.get(key) == gt60.get(key) for key in ['images', 'annotations', 'categories', 'ignored_valid_gt']}
    assert all(equality.values()) and len(gt58['images']) == 548 and len(gt58['annotations']) == 38602
    dimensions = {x['id']: (x['width'], x['height']) for x in gt58['images']}
    assert len(dimensions) == 548

    def describe(path, expected):
        predictions = read(path)
        assert len(predictions) == expected
        image_counts = Counter(); outside = 0
        for item in predictions:
            assert item['image_id'] in dimensions and 1 <= item['category_id'] <= 10
            x, y, w, h = item['bbox']; width, height = dimensions[item['image_id']]
            assert all(math.isfinite(v) for v in [x, y, w, h, item['score']]) and w > 0 and h > 0
            assert 0 <= item['score'] <= 1
            image_counts[item['image_id']] += 1
            outside += bool(x < 0 or y < 0 or x + w > width or y + h > height)
        per_image = [image_counts[i] for i in dimensions]
        assert max(per_image) <= 300
        return {'predictions': len(predictions), 'per_image_mean': statistics.mean(per_image),
                'per_image_median': statistics.median(per_image), 'per_image_min': min(per_image),
                'per_image_max': max(per_image), 'boxes_crossing_image_bounds': outside,
                'boxes_crossing_image_bounds_percent': 100 * outside / len(predictions) if predictions else None,
                'boundary_rule': 'Exact x<0 or y<0 or x+w>width or y+h>height; no tolerance.'}

    m58, m60, m58_12 = read(files['A58_B8_metrics']), read(files['A60_B8_metrics']), read(files['A58_B12_metrics'])
    predictions58 = describe(files['A58_B8_predictions'], m58['predictions_after_ignore'])
    predictions60 = describe(files['A60_B8_predictions'], m60['predictions_after_ignore'])
    p58, p60, constructor = read(files['A58_protocol']), read(files['A60_protocol']), read(files['A58_constructor'])
    a58_parameters = p58['operator']['registered_parameters_all_arms']
    assert a58_parameters == constructor['augmented_parameters'] == 9951721
    assert constructor['base_parameters'] == 9803112 and p58['operator']['B'] == 'Registered context disabled; original ECDet-S path'
    a60_parameters = p60['trainable_parameters']; assert a60_parameters == 110565396
    difference = {key: m60[key] - m58[key] for key in ['AP', 'AP50', 'AP75', 'APs', 'APs75', 'ARs']}
    result = {
        'status': 'COMPLETED_EXPLORATORY_SAVED_B8_PIPELINE_COMPARISON',
        'recorded_client_utc': datetime.now(timezone.utc).isoformat(),
        'scope_sha256': sha(SCOPE), 'supplement_sha256': sha(SUPPLEMENT),
        'validation_GT_fields_identical': equality,
        'A58_B8_EMA_metrics': m58, 'A60_B8_raw_metrics': m60,
        'A60_minus_A58_B8_pp_observed_pipeline_difference': difference,
        'A58_B12_EMA_metrics_for_context': m58_12,
        'A58_B8_predictions': predictions58, 'A60_B8_predictions': predictions60,
        'A58_registered_parameters_with_disabled_context': a58_parameters,
        'A58_original_unaugmented_parameters': constructor['base_parameters'],
        'A60_parameters': a60_parameters, 'A60_to_A58_registered_parameter_ratio': a60_parameters / a58_parameters,
        'postprocess_source_check': 'A58 pinned postprocessor forward has no NMS call; A60 applies classwise NMS at 0.6. Other pipeline differences remain.',
        'comparison_limits': ['A58 uses EMA; A60 uses raw weights', 'different architecture, initialization, seed and optimizer',
                              'A58 native batch8; A60 microbatch1 with accumulation8 and per-image loss reduction',
                              'different postprocessing and number of surviving predictions',
                              'B8 only for A60; Q and M still untested at the observed snapshot',
                              'saved outputs, not official VisDrone challenge scores or a causally matched model comparison'],
        'new_forwards': 0, 'new_updates': 0, 'new_COCO_evaluations': 0, 'new_matching_passes': 0,
        'inference_scope': 'Exploratory practical comparison, not a frozen A60 gate or a candidate efficacy result.',
    }
    report = (
        '已有 ECDet-S 基线与 A60 生成式基线：现有结果的可比性\n\n'
        '这是一项对已保存结果的探索性核对，不调整 A60 的冻结训练协议、停止规则或收益门槛。\n'
        '两个 B8 结果的 548 张验证图、38602 个计分框、类别和忽略框数量逐项相同；'
        '都使用该项目的 COCO-style 口径，非 VisDrone 官方挑战赛分数。\n\n'
        f"第8轮：A58 原 ECDet-S 路径（EMA）AP {m58['AP']:.4f}，APs {m58['APs']:.4f}；"
        f"A60 R50-FPN/DynamicHead（原始权重）AP {m60['AP']:.4f}，APs {m60['APs']:.4f}。\n"
        f"在各自已执行配置下，A60减A58分别为 {difference['AP']:.4f} 和 {difference['APs']:.4f} 个百分点。"
        '它是已观察到的整套流程差异，不是架构因果差异或候选Q收益。\n'
        f"A58第12轮EMA基线：AP {m58_12['AP']:.4f}，APs {m58_12['APs']:.4f}；A60第12轮在此记录时尚未有可核验结果。\n\n"
        f"参数：A58三组共同注册 {a58_parameters:,} 个参数（其中路由额外参数在B中关闭；原版 {constructor['base_parameters']:,}）；"
        f"A60共 {a60_parameters:,} 个，约为前者 {a60_parameters / a58_parameters:.2f} 倍。"
        '参数不等于算力、显存或时延；A60固定终点时延尚待实测。\n'
        f"保存预测：A58每图平均 {predictions58['per_image_mean']:.2f} 个，A60每图平均 {predictions60['per_image_mean']:.2f} 个。"
        'A58已绑定源码的后处理不调用NMS；A60先取全局top300，再做类内NMS（0.6）和裁剪。'
        '预测数量差异不能归因于某一个步骤。\n'
        f"按严格图像边界统计，A58有 {predictions58['boxes_crossing_image_bounds']:,} 个保存预测框越界，"
        f"A60有 {predictions60['boxes_crossing_image_bounds']:,} 个；计数只用于说明实现差异，不代表AP损失的因果分解。\n\n"
        '两者还存在架构、初始化、随机种子、优化器、EMA与原始权重、原生batch与梯度累积、后处理等差异。'
        '因此不能用该对照证明哪个单独设计更优，也不能直接认定Q失败。'
        '但它提示实用风险：即使Q通过内部B/M收益门槛，仍需结合完整12轮结果、相同条件下的可靠性及与强基线的差距判断是否值得作为论文主线。\n'
        '本次只读取保存的GT、预测、指标、配置和源码；无新模型前向、更新、COCO重评或GT匹配。\n'
    )
    with RESULT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(result, out, indent=2, ensure_ascii=False, allow_nan=False); out.write('\n')
    with REPORT.open('x', encoding='utf-8', newline='\n') as out:
        out.write(report)
    print(json.dumps({'status': result['status'], 'A58_B8_AP': m58['AP'], 'A60_B8_AP': m60['AP'],
                      'A58_B8_APs': m58['APs'], 'A60_B8_APs': m60['APs'],
                      'parameters_ratio': result['A60_to_A58_registered_parameter_ratio'],
                      'A58_outside': predictions58['boxes_crossing_image_bounds'],
                      'A60_outside': predictions60['boxes_crossing_image_bounds']}))


if __name__ == '__main__':
    main()
