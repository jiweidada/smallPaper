#!/usr/bin/env python3
"""
P2桥接的AP50 vs AP75细分对比
验证P2收益是否主要来自精细定位(IoU≥0.75)还是召回率(IoU≥0.5)
"""
import json
import gzip
from pathlib import Path
from collections import defaultdict

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

    return gt_dict

def compute_size_bins(gt_dict):
    """按缩放后尺寸分bin"""
    gt_bins = {}
    for ann_id, ann in gt_dict.items():
        x, y, w, h = ann['bbox']
        short_side = min(w, h)

        if short_side < 8:
            bin_name = '<8px'
        elif short_side < 16:
            bin_name = '8-16px'
        elif short_side < 32:
            bin_name = '16-32px'
        else:
            bin_name = '>=32px'

        gt_bins[ann_id] = bin_name

    return gt_bins

def analyze_recall_by_iou_threshold(matching_data, gt_dict, gt_bins, iou_threshold_idx):
    """
    按IoU阈值统计召回率
    iou_threshold_idx: 0 for IoU>=0.5, 5 for IoU>=0.75
    COCO评估IoU阈值: [0.50:0.05:0.95] 共10个
    索引5对应0.75
    """
    detected_by_bin = defaultdict(set)
    total_by_bin = defaultdict(set)

    for eval_item in matching_data:
        gt_ids = eval_item['gtIds']
        gt_matches = eval_item['gtMatches']

        for i, gt_id in enumerate(gt_ids):
            if gt_id not in gt_dict:
                continue

            bin_name = gt_bins.get(gt_id, 'unknown')
            total_by_bin[bin_name].add(gt_id)

            # gt_matches是二维数组 [num_gts, num_iou_thresholds]
            if i < len(gt_matches) and gt_matches[i]:
                if isinstance(gt_matches[i], list) and len(gt_matches[i]) > iou_threshold_idx:
                    matched = gt_matches[i][iou_threshold_idx]
                else:
                    matched = 0

                if matched > 0:
                    detected_by_bin[bin_name].add(gt_id)

    recall_by_bin = {}
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        det = len(detected_by_bin[bin_name])
        tot = len(total_by_bin[bin_name])
        recall = det / tot if tot > 0 else 0
        recall_by_bin[bin_name] = {
            'detected': det,
            'total': tot,
            'recall': recall
        }

    return recall_by_bin

def main():
    print("=== P2桥接：AP50 vs AP75 细分对比 ===\n")

    # 加载数据
    gt_dict = load_annotations()
    gt_bins = compute_size_bins(gt_dict)

    print(f"GT总数: {len(gt_dict)}\n")

    # 加载匹配数据
    p2_match_s1 = load_matching(P2_SEED1 / "matching_all_area.json.gz")
    base_match_s1 = load_matching(BASELINE_SEED1 / "matching_all_area.json.gz")
    p2_match_s2 = load_matching(P2_SEED2 / "matching_all_area.json.gz")
    base_match_s2 = load_matching(BASELINE_SEED2 / "matching_all_area.json.gz")

    # ===== Seed 1 分析 =====
    print("=" * 70)
    print("Seed 1: P2 vs 预训练基线")
    print("=" * 70)

    # IoU >= 0.5
    print("\n【IoU >= 0.5 召回率】")
    base_recall_50_s1 = analyze_recall_by_iou_threshold(base_match_s1, gt_dict, gt_bins, 0)
    p2_recall_50_s1 = analyze_recall_by_iou_threshold(p2_match_s1, gt_dict, gt_bins, 0)

    print(f"\n{'尺寸bin':<12} {'基线召回率':>12} {'P2召回率':>12} {'Δ召回率':>10}")
    print("-" * 50)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        base_r = base_recall_50_s1[bin_name]['recall'] * 100
        p2_r = p2_recall_50_s1[bin_name]['recall'] * 100
        delta = p2_r - base_r
        print(f"{bin_name:<12} {base_r:>11.2f}% {p2_r:>11.2f}% {delta:>+9.2f}pp")

    # IoU >= 0.75
    print("\n【IoU >= 0.75 召回率】")
    base_recall_75_s1 = analyze_recall_by_iou_threshold(base_match_s1, gt_dict, gt_bins, 5)
    p2_recall_75_s1 = analyze_recall_by_iou_threshold(p2_match_s1, gt_dict, gt_bins, 5)

    print(f"\n{'尺寸bin':<12} {'基线召回率':>12} {'P2召回率':>12} {'Δ召回率':>10}")
    print("-" * 50)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        base_r = base_recall_75_s1[bin_name]['recall'] * 100
        p2_r = p2_recall_75_s1[bin_name]['recall'] * 100
        delta = p2_r - base_r
        print(f"{bin_name:<12} {base_r:>11.2f}% {p2_r:>11.2f}% {delta:>+9.2f}pp")

    # 对比50 vs 75的提升幅度
    print("\n【Δ召回率对比：IoU≥0.75 相对于 IoU≥0.5】")
    print(f"{'尺寸bin':<12} {'Δ@0.5':>10} {'Δ@0.75':>10} {'差值':>10} {'比值':>8}")
    print("-" * 55)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta_50 = (p2_recall_50_s1[bin_name]['recall'] - base_recall_50_s1[bin_name]['recall']) * 100
        delta_75 = (p2_recall_75_s1[bin_name]['recall'] - base_recall_75_s1[bin_name]['recall']) * 100
        diff = delta_75 - delta_50
        ratio = delta_75 / delta_50 if delta_50 > 0 else 0
        print(f"{bin_name:<12} {delta_50:>+9.2f}pp {delta_75:>+9.2f}pp {diff:>+9.2f}pp {ratio:>7.2f}x")

    # ===== Seed 2 分析 =====
    print("\n\n" + "=" * 70)
    print("Seed 2: P2 vs 预训练基线")
    print("=" * 70)

    # IoU >= 0.5
    print("\n【IoU >= 0.5 召回率】")
    base_recall_50_s2 = analyze_recall_by_iou_threshold(base_match_s2, gt_dict, gt_bins, 0)
    p2_recall_50_s2 = analyze_recall_by_iou_threshold(p2_match_s2, gt_dict, gt_bins, 0)

    print(f"\n{'尺寸bin':<12} {'基线召回率':>12} {'P2召回率':>12} {'Δ召回率':>10}")
    print("-" * 50)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        base_r = base_recall_50_s2[bin_name]['recall'] * 100
        p2_r = p2_recall_50_s2[bin_name]['recall'] * 100
        delta = p2_r - base_r
        print(f"{bin_name:<12} {base_r:>11.2f}% {p2_r:>11.2f}% {delta:>+9.2f}pp")

    # IoU >= 0.75
    print("\n【IoU >= 0.75 召回率】")
    base_recall_75_s2 = analyze_recall_by_iou_threshold(base_match_s2, gt_dict, gt_bins, 5)
    p2_recall_75_s2 = analyze_recall_by_iou_threshold(p2_match_s2, gt_dict, gt_bins, 5)

    print(f"\n{'尺寸bin':<12} {'基线召回率':>12} {'P2召回率':>12} {'Δ召回率':>10}")
    print("-" * 50)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        base_r = base_recall_75_s2[bin_name]['recall'] * 100
        p2_r = p2_recall_75_s2[bin_name]['recall'] * 100
        delta = p2_r - base_r
        print(f"{bin_name:<12} {base_r:>11.2f}% {p2_r:>11.2f}% {delta:>+9.2f}pp")

    # 对比50 vs 75的提升幅度
    print("\n【Δ召回率对比：IoU≥0.75 相对于 IoU≥0.5】")
    print(f"{'尺寸bin':<12} {'Δ@0.5':>10} {'Δ@0.75':>10} {'差值':>10} {'比值':>8}")
    print("-" * 55)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta_50 = (p2_recall_50_s2[bin_name]['recall'] - base_recall_50_s2[bin_name]['recall']) * 100
        delta_75 = (p2_recall_75_s2[bin_name]['recall'] - base_recall_75_s2[bin_name]['recall']) * 100
        diff = delta_75 - delta_50
        ratio = delta_75 / delta_50 if delta_50 > 0 else 0
        print(f"{bin_name:<12} {delta_50:>+9.2f}pp {delta_75:>+9.2f}pp {diff:>+9.2f}pp {ratio:>7.2f}x")

    # ===== 关键结论 =====
    print("\n\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)

    print("\n如果P2主要改善精细定位，应该看到：")
    print("  ✓ Δ@0.75 > Δ@0.5（严格定位提升更大）")
    print("  ✓ 比值>1.2（显著更大）")
    print("  ✓ 小目标的比值最大（它们最需要精细定位）")

    print("\n如果P2主要改善召回率，应该看到：")
    print("  ✓ Δ@0.75 ≈ Δ@0.5（提升幅度相当）")
    print("  ✓ 比值接近1.0")
    print("  ✓ 所有尺度的比值相近")

    print("\n实际观察到的模式将揭示P2桥接的真实作用机制。")

if __name__ == '__main__':
    main()
