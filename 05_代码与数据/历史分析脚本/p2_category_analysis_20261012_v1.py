#!/usr/bin/env python3
"""
P2桥接收益的多维度分解分析
- 按类别分解
- 按IoU阈值（0.5/0.75）分解
- 计算新增检出 vs 原有成功目标的损害
"""
import json
import gzip
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("C:/Users/jiwei/Desktop/小论文投稿")

# P2和基线路径
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

    # 类别映射
    cat_names = {cat['id']: cat['name'] for cat in data['categories']}

    return gt_dict, cat_names

def extract_tp_by_category(matching_data, gt_dict):
    """
    从COCO评估格式提取每个类别的TP
    返回: {category_id: set(gt_ids)}
    """
    tp_by_cat = defaultdict(set)

    for eval_item in matching_data:
        cat_id = eval_item['category_id']
        gt_ids = eval_item['gtIds']
        gt_matches = eval_item['gtMatches']

        for i, gt_id in enumerate(gt_ids):
            if gt_id not in gt_dict:
                continue

            # 检查IoU>=0.5的匹配（第一列）
            if i < len(gt_matches) and gt_matches[i]:
                if isinstance(gt_matches[i], list):
                    matched = gt_matches[i][0] if len(gt_matches[i]) > 0 else 0
                else:
                    matched = gt_matches[i]

                if matched > 0:
                    tp_by_cat[cat_id].add(gt_id)

    return tp_by_cat

def extract_tp_by_category_and_iou(matching_data, gt_dict, iou_idx=0):
    """
    提取指定IoU阈值的TP
    iou_idx: 0 for IoU>=0.5, 7 for IoU>=0.75
    """
    tp_by_cat = defaultdict(set)

    for eval_item in matching_data:
        cat_id = eval_item['category_id']
        gt_ids = eval_item['gtIds']
        gt_matches = eval_item['gtMatches']

        for i, gt_id in enumerate(gt_ids):
            if gt_id not in gt_dict:
                continue

            if i < len(gt_matches) and gt_matches[i]:
                if isinstance(gt_matches[i], list) and len(gt_matches[i]) > iou_idx:
                    matched = gt_matches[i][iou_idx]
                else:
                    matched = 0

                if matched > 0:
                    tp_by_cat[cat_id].add(gt_id)

    return tp_by_cat

def compute_category_gains_losses(baseline_tp, p2_tp, gt_dict, cat_names):
    """
    计算每个类别的增益和损失
    """
    all_categories = set(baseline_tp.keys()) | set(p2_tp.keys())

    results = []
    for cat_id in sorted(all_categories):
        baseline_set = baseline_tp.get(cat_id, set())
        p2_set = p2_tp.get(cat_id, set())

        gained = p2_set - baseline_set
        lost = baseline_set - p2_set
        net = len(gained) - len(lost)

        results.append({
            'category_id': cat_id,
            'name': cat_names.get(cat_id, f'unknown_{cat_id}'),
            'baseline_tp': len(baseline_set),
            'p2_tp': len(p2_set),
            'gained': len(gained),
            'lost': len(lost),
            'net': net
        })

    return results

def main():
    print("=== P2桥接收益多维度分解 ===\n")

    # 加载GT和类别名称
    gt_dict, cat_names = load_annotations()
    print(f"GT总数: {len(gt_dict)}")
    print(f"类别数: {len(cat_names)}")
    print(f"类别列表: {cat_names}\n")

    # ===== Seed 1 分析 =====
    print("\n" + "="*60)
    print("Seed 1: P2 vs 预训练基线")
    print("="*60)

    p2_match_s1 = load_matching(P2_SEED1 / "matching_all_area.json.gz")
    base_match_s1 = load_matching(BASELINE_SEED1 / "matching_all_area.json.gz")

    # IoU >= 0.5
    print("\n--- IoU >= 0.5 ---")
    baseline_tp_s1 = extract_tp_by_category(base_match_s1, gt_dict)
    p2_tp_s1 = extract_tp_by_category(p2_match_s1, gt_dict)

    cat_results_s1 = compute_category_gains_losses(baseline_tp_s1, p2_tp_s1, gt_dict, cat_names)

    print("\n按类别的TP变化:")
    print(f"{'类别':<20} {'基线TP':>8} {'P2 TP':>8} {'新增':>6} {'损失':>6} {'净变化':>6}")
    print("-" * 70)

    total_gained = 0
    total_lost = 0
    for r in cat_results_s1:
        print(f"{r['name']:<20} {r['baseline_tp']:>8} {r['p2_tp']:>8} {r['gained']:>6} {r['lost']:>6} {r['net']:>+6}")
        total_gained += r['gained']
        total_lost += r['lost']

    print("-" * 70)
    print(f"{'总计':<20} {'':<8} {'':<8} {total_gained:>6} {total_lost:>6} {total_gained-total_lost:>+6}")
    print(f"\n新增检出被损失抵消比例: {total_lost/total_gained*100:.1f}%")

    # IoU >= 0.75 (注意：需要确认matching数据的IoU阈值索引)
    # COCO评估默认IoU阈值: [0.5:0.05:0.95]，共10个
    # 索引7对应0.75

    # ===== Seed 2 分析 =====
    print("\n\n" + "="*60)
    print("Seed 2: P2 vs 预训练基线")
    print("="*60)

    p2_match_s2 = load_matching(P2_SEED2 / "matching_all_area.json.gz")
    base_match_s2 = load_matching(BASELINE_SEED2 / "matching_all_area.json.gz")

    print("\n--- IoU >= 0.5 ---")
    baseline_tp_s2 = extract_tp_by_category(base_match_s2, gt_dict)
    p2_tp_s2 = extract_tp_by_category(p2_match_s2, gt_dict)

    cat_results_s2 = compute_category_gains_losses(baseline_tp_s2, p2_tp_s2, gt_dict, cat_names)

    print("\n按类别的TP变化:")
    print(f"{'类别':<20} {'基线TP':>8} {'P2 TP':>8} {'新增':>6} {'损失':>6} {'净变化':>6}")
    print("-" * 70)

    total_gained_s2 = 0
    total_lost_s2 = 0
    for r in cat_results_s2:
        print(f"{r['name']:<20} {r['baseline_tp']:>8} {r['p2_tp']:>8} {r['gained']:>6} {r['lost']:>6} {r['net']:>+6}")
        total_gained_s2 += r['gained']
        total_lost_s2 += r['lost']

    print("-" * 70)
    print(f"{'总计':<20} {'':<8} {'':<8} {total_gained_s2:>6} {total_lost_s2:>6} {total_gained_s2-total_lost_s2:>+6}")
    print(f"\n新增检出被损失抵消比例: {total_lost_s2/total_gained_s2*100:.1f}%")

    # ===== 双Seed汇总 =====
    print("\n\n" + "="*60)
    print("双Seed汇总")
    print("="*60)

    print("\n每个类别的平均净增益（两seed平均）:")
    print(f"{'类别':<20} {'Seed1净变化':>12} {'Seed2净变化':>12} {'平均':>8}")
    print("-" * 60)

    for r1, r2 in zip(cat_results_s1, cat_results_s2):
        avg = (r1['net'] + r2['net']) / 2
        print(f"{r1['name']:<20} {r1['net']:>+12} {r2['net']:>+12} {avg:>+8.1f}")

    print("\n关键发现:")
    print(f"1. Seed1新增检出抵消率: {total_lost/total_gained*100:.1f}%")
    print(f"2. Seed2新增检出抵消率: {total_lost_s2/total_gained_s2*100:.1f}%")
    print(f"3. P2桥接在救回目标的同时，也会损失原有的成功检测")
    print(f"4. 需要检查损失的目标是否因为Top300竞争被挤出")

if __name__ == '__main__':
    main()
