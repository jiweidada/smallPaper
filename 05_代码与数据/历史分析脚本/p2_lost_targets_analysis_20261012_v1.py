#!/usr/bin/env python3
"""
P2损失目标的特征分析
分析P2新增TP但同时损失的593/492个原有成功目标的特征
"""
import json
import gzip
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("C:/Users/jiwei/Desktop/小论文投稿")

P2_SEED1 = BASE / "outputs/user_backbone_primary_collection12_20261011_v2/seed1_p2_skip_v1/extracted/run/endpoint_epoch12"
P2_SEED2 = BASE / "outputs/user_backbone_primary_collection12_20261011_v2/seed2_p2_skip_v1/extracted/run/endpoint_epoch12"
BASELINE_SEED1 = BASE / "outputs/stem_fixed12_seed1_formal_collection12_20261011_v1/seed1_native_stem_scratch/extracted/run/endpoint_epoch12"
BASELINE_SEED2 = BASE / "outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_native_stem_scratch/extracted/run/endpoint_epoch12"

ANNO_PATH = BASE / "数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-val/annotations/val.json"

def load_matching(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)

def load_annotations():
    with open(ANNO_PATH, 'r') as f:
        data = json.load(f)

    gt_dict = {}
    for ann in data['annotations']:
        gt_dict[ann['id']] = ann

    cat_names = {cat['id']: cat['name'] for cat in data['categories']}
    return gt_dict, cat_names

def extract_all_tp_by_category(matching_data, gt_dict):
    """提取每个类别的所有TP"""
    tp_by_cat = defaultdict(set)

    for eval_item in matching_data:
        cat_id = eval_item['category_id']
        gt_ids = eval_item['gtIds']
        gt_matches = eval_item['gtMatches']

        for i, gt_id in enumerate(gt_ids):
            if gt_id not in gt_dict:
                continue

            if i < len(gt_matches) and gt_matches[i]:
                if isinstance(gt_matches[i], list):
                    matched = gt_matches[i][0] if len(gt_matches[i]) > 0 else 0
                else:
                    matched = gt_matches[i]

                if matched > 0:
                    tp_by_cat[cat_id].add(gt_id)

    return tp_by_cat

def compute_gt_statistics(gt_ids, gt_dict):
    """计算一组GT的统计特征"""
    areas = []
    widths = []
    heights = []
    short_sides = []

    for gt_id in gt_ids:
        if gt_id not in gt_dict:
            continue
        ann = gt_dict[gt_id]
        x, y, w, h = ann['bbox']
        area = w * h
        short_side = min(w, h)

        areas.append(area)
        widths.append(w)
        heights.append(h)
        short_sides.append(short_side)

    if not areas:
        return None

    return {
        'count': len(areas),
        'area_mean': np.mean(areas),
        'area_std': np.std(areas),
        'area_median': np.median(areas),
        'short_side_mean': np.mean(short_sides),
        'short_side_median': np.median(short_sides),
        'width_mean': np.mean(widths),
        'height_mean': np.mean(heights),
    }

def analyze_lost_targets(baseline_tp, p2_tp, gt_dict, cat_names):
    """分析损失的目标"""
    results = {}

    for cat_id in sorted(set(baseline_tp.keys()) | set(p2_tp.keys())):
        baseline_set = baseline_tp.get(cat_id, set())
        p2_set = p2_tp.get(cat_id, set())

        lost = baseline_set - p2_set
        gained = p2_set - baseline_set
        retained = baseline_set & p2_set

        if lost:
            lost_stats = compute_gt_statistics(lost, gt_dict)
            retained_stats = compute_gt_statistics(retained, gt_dict)

            results[cat_id] = {
                'name': cat_names.get(cat_id, f'unknown_{cat_id}'),
                'baseline_tp': len(baseline_set),
                'p2_tp': len(p2_set),
                'lost_count': len(lost),
                'gained_count': len(gained),
                'retained_count': len(retained),
                'lost_stats': lost_stats,
                'retained_stats': retained_stats,
            }

    return results

def main():
    print("=== P2损失目标特征分析 ===\n")

    # 加载数据
    gt_dict, cat_names = load_annotations()

    # ===== Seed 1 =====
    print("=" * 70)
    print("Seed 1: 分析损失的593个目标")
    print("=" * 70)

    p2_match_s1 = load_matching(P2_SEED1 / "matching_all_area.json.gz")
    base_match_s1 = load_matching(BASELINE_SEED1 / "matching_all_area.json.gz")

    baseline_tp_s1 = extract_all_tp_by_category(base_match_s1, gt_dict)
    p2_tp_s1 = extract_all_tp_by_category(p2_match_s1, gt_dict)

    lost_analysis_s1 = analyze_lost_targets(baseline_tp_s1, p2_tp_s1, gt_dict, cat_names)

    print("\n按类别的损失目标特征:")
    print(f"{'类别':<15} {'损失数':>8} {'损失占比':>10} {'平均面积':>12} {'平均短边':>12}")
    print("-" * 70)

    total_lost_s1 = 0
    for cat_id, data in sorted(lost_analysis_s1.items(), key=lambda x: x[1]['lost_count'], reverse=True):
        if data['lost_count'] > 0:
            lost_ratio = data['lost_count'] / data['baseline_tp'] * 100
            lost_stats = data['lost_stats']

            print(f"{data['name']:<15} {data['lost_count']:>8} {lost_ratio:>9.1f}% "
                  f"{lost_stats['area_mean']:>11.1f} {lost_stats['short_side_mean']:>11.2f}px")

            total_lost_s1 += data['lost_count']

    print("-" * 70)
    print(f"{'总计':<15} {total_lost_s1:>8}")

    # 对比损失 vs 保留的目标尺寸
    print("\n\n损失目标 vs 保留目标的尺寸对比（主要类别）:")
    print(f"{'类别':<15} {'损失目标':<25} {'保留目标':<25} {'差异':<15}")
    print(f"{'':15} {'短边均值':<12} {'面积均值':<12} {'短边均值':<12} {'面积均值':<12} {'短边差':<12}")
    print("-" * 90)

    for cat_id, data in sorted(lost_analysis_s1.items(), key=lambda x: x[1]['lost_count'], reverse=True)[:5]:
        if data['lost_count'] > 0 and data['retained_stats']:
            lost_s = data['lost_stats']['short_side_mean']
            lost_a = data['lost_stats']['area_mean']
            kept_s = data['retained_stats']['short_side_mean']
            kept_a = data['retained_stats']['area_mean']
            diff_s = lost_s - kept_s

            print(f"{data['name']:<15} {lost_s:>11.2f}px {lost_a:>11.1f} "
                  f"{kept_s:>11.2f}px {kept_a:>11.1f} {diff_s:>+11.2f}px")

    # ===== Seed 2 =====
    print("\n\n" + "=" * 70)
    print("Seed 2: 分析损失的492个目标")
    print("=" * 70)

    p2_match_s2 = load_matching(P2_SEED2 / "matching_all_area.json.gz")
    base_match_s2 = load_matching(BASELINE_SEED2 / "matching_all_area.json.gz")

    baseline_tp_s2 = extract_all_tp_by_category(base_match_s2, gt_dict)
    p2_tp_s2 = extract_all_tp_by_category(p2_match_s2, gt_dict)

    lost_analysis_s2 = analyze_lost_targets(baseline_tp_s2, p2_tp_s2, gt_dict, cat_names)

    print("\n按类别的损失目标特征:")
    print(f"{'类别':<15} {'损失数':>8} {'损失占比':>10} {'平均面积':>12} {'平均短边':>12}")
    print("-" * 70)

    total_lost_s2 = 0
    for cat_id, data in sorted(lost_analysis_s2.items(), key=lambda x: x[1]['lost_count'], reverse=True):
        if data['lost_count'] > 0:
            lost_ratio = data['lost_count'] / data['baseline_tp'] * 100
            lost_stats = data['lost_stats']

            print(f"{data['name']:<15} {data['lost_count']:>8} {lost_ratio:>9.1f}% "
                  f"{lost_stats['area_mean']:>11.1f} {lost_stats['short_side_mean']:>11.2f}px")

            total_lost_s2 += data['lost_count']

    print("-" * 70)
    print(f"{'总计':<15} {total_lost_s2:>8}")

    # 对比损失 vs 保留
    print("\n\n损失目标 vs 保留目标的尺寸对比（主要类别）:")
    print(f"{'类别':<15} {'损失目标':<25} {'保留目标':<25} {'差异':<15}")
    print(f"{'':15} {'短边均值':<12} {'面积均值':<12} {'短边均值':<12} {'面积均值':<12} {'短边差':<12}")
    print("-" * 90)

    for cat_id, data in sorted(lost_analysis_s2.items(), key=lambda x: x[1]['lost_count'], reverse=True)[:5]:
        if data['lost_count'] > 0 and data['retained_stats']:
            lost_s = data['lost_stats']['short_side_mean']
            lost_a = data['lost_stats']['area_mean']
            kept_s = data['retained_stats']['short_side_mean']
            kept_a = data['retained_stats']['area_mean']
            diff_s = lost_s - kept_s

            print(f"{data['name']:<15} {lost_s:>11.2f}px {lost_a:>11.1f} "
                  f"{kept_s:>11.2f}px {kept_a:>11.1f} {diff_s:>+11.2f}px")

    # ===== 关键结论 =====
    print("\n\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)

    print(f"\n1. 总损失数量:")
    print(f"   Seed1: {total_lost_s1} 个TP（占新增3226的18.4%）")
    print(f"   Seed2: {total_lost_s2} 个TP（占新增3527的13.9%）")

    print(f"\n2. 损失目标的尺寸特征:")
    print(f"   如果损失目标显著小于保留目标 -> 可能是Top300竞争导致小目标被挤出")
    print(f"   如果损失目标显著大于保留目标 -> 可能是P2对某些大目标产生干扰")
    print(f"   如果尺寸相近 -> 可能是边界/遮挡等其他因素")

    print(f"\n3. 后续深入分析方向:")
    print(f"   - 检查损失目标的基线置信度分布（是否接近Top300边界）")
    print(f"   - 分析损失目标的遮挡/截断属性")
    print(f"   - 检查是否集中在图像边缘区域")

if __name__ == '__main__':
    main()
