#!/usr/bin/env python3
"""
P2桥接效果的尺寸分组分析 v2
比较原P2接入 vs 预训练基线，按目标尺寸分bin计算召回率
使用COCO评估格式的匹配数据
"""
import json
import gzip
from pathlib import Path
from collections import defaultdict
import numpy as np

# 路径配置
BASE = Path("C:/Users/jiwei/Desktop/小论文投稿")
P2_SEED1 = BASE / "outputs/user_backbone_primary_collection12_20261011_v2/seed1_p2_skip_v1/extracted/run/endpoint_epoch12"
P2_SEED2 = BASE / "outputs/user_backbone_primary_collection12_20261011_v2/seed2_p2_skip_v1/extracted/run/endpoint_epoch12"
BASELINE_SEED1 = BASE / "outputs/stem_fixed12_seed1_formal_collection12_20261011_v1/seed1_native_stem_scratch/extracted/run/endpoint_epoch12"
BASELINE_SEED2 = BASE / "outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_native_stem_scratch/extracted/run/endpoint_epoch12"

# GT标注路径
ANNO_PATH = BASE / "数据/03_本轮实验记录_20260928/转换COCO标注原件_20260929/VisDrone2019-DET-val/annotations/val.json"

def load_matching(path):
    """加载匹配结果JSON"""
    with gzip.open(path, 'rt') as f:
        return json.load(f)

def load_annotations():
    """加载GT标注，返回按ID索引的字典"""
    with open(ANNO_PATH, 'r') as f:
        data = json.load(f)

    # 构建GT字典: ann_id -> annotation
    gt_dict = {}
    for ann in data['annotations']:
        gt_dict[ann['id']] = ann

    return gt_dict, data['images']

def compute_size_bins(gt_dict):
    """
    按缩放后尺寸分bin
    bins: <8px, 8-16px, 16-32px, 32+px（短边）
    """
    gt_bins = {}  # ann_id -> bin_name
    bin_counts = defaultdict(int)

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
        bin_counts[bin_name] += 1

    return gt_bins, bin_counts

def analyze_recall_by_bin(matching_data, gt_dict, gt_bins, iou_thresh=0.5):
    """
    从COCO格式匹配数据统计召回
    matching_data: list of {image_id, category_id, gtIds, gtMatches, ...}
    每个gtMatches[i]是匹配的检测ID，0表示未匹配
    """
    detected = set()
    total = set()

    detected_by_bin = defaultdict(set)
    total_by_bin = defaultdict(set)

    # 遍历所有image×category组合
    for eval_item in matching_data:
        gt_ids = eval_item['gtIds']
        gt_matches = eval_item['gtMatches']

        # gt_matches是一个数组，长度为len(gt_ids)×maxDet（这里是100）
        # 每行对应一个GT，如果匹配了检测则为检测ID，否则为0
        # 第一列是IoU阈值0.5的匹配，后面是更严格的阈值

        # 对于IoU>=0.5，我们只看第一列（索引0）
        num_gts = len(gt_ids)
        for i, gt_id in enumerate(gt_ids):
            if gt_id not in gt_dict:
                continue

            total.add(gt_id)
            bin_name = gt_bins.get(gt_id, 'unknown')
            total_by_bin[bin_name].add(gt_id)

            # 检查是否有匹配（gt_matches是二维数组，shape: [num_gts, num_iou_thresholds]）
            if i < len(gt_matches) and gt_matches[i]:
                # gt_matches[i]是一个列表，第0个元素是IoU>=0.5的匹配
                if isinstance(gt_matches[i], list):
                    matched = gt_matches[i][0] if len(gt_matches[i]) > 0 else 0
                else:
                    matched = gt_matches[i]

                if matched > 0:  # 有匹配的检测
                    detected.add(gt_id)
                    detected_by_bin[bin_name].add(gt_id)

    # 计算召回率
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
    print("=== P2桥接尺寸分组分析 ===\n")

    # 加载GT
    print("加载GT标注...")
    gt_dict, images = load_annotations()
    gt_bins, bin_counts = compute_size_bins(gt_dict)

    print(f"\n总GT数: {len(gt_dict)}")
    print("目标尺寸分布:")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        print(f"  {bin_name}: {bin_counts[bin_name]} 个 ({bin_counts[bin_name]/len(gt_dict)*100:.1f}%)")

    # 分析seed1
    print("\n\n--- Seed 1 ---")
    p2_match = load_matching(P2_SEED1 / "matching_all_area.json.gz")
    base_match = load_matching(BASELINE_SEED1 / "matching_all_area.json.gz")

    print("\n预训练基线 (IoU>=0.5):")
    base_recall = analyze_recall_by_bin(base_match, gt_dict, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = base_recall[bin_name]
        print(f"  {bin_name}: {r['detected']}/{r['total']} = {r['recall']*100:.2f}%")

    print("\nP2接入 (IoU>=0.5):")
    p2_recall = analyze_recall_by_bin(p2_match, gt_dict, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = p2_recall[bin_name]
        print(f"  {bin_name}: {r['detected']}/{r['total']} = {r['recall']*100:.2f}%")

    print("\nΔ召回率 (P2 - 基线):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta = p2_recall[bin_name]['recall'] - base_recall[bin_name]['recall']
        delta_count = p2_recall[bin_name]['detected'] - base_recall[bin_name]['detected']
        print(f"  {bin_name}: {delta*100:+.2f} pp ({delta_count:+d} 个)")

    # 分析seed2
    print("\n\n--- Seed 2 ---")
    p2_match_s2 = load_matching(P2_SEED2 / "matching_all_area.json.gz")
    base_match_s2 = load_matching(BASELINE_SEED2 / "matching_all_area.json.gz")

    print("\n预训练基线 (IoU>=0.5):")
    base_recall_s2 = analyze_recall_by_bin(base_match_s2, gt_dict, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = base_recall_s2[bin_name]
        print(f"  {bin_name}: {r['detected']}/{r['total']} = {r['recall']*100:.2f}%")

    print("\nP2接入 (IoU>=0.5):")
    p2_recall_s2 = analyze_recall_by_bin(p2_match_s2, gt_dict, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = p2_recall_s2[bin_name]
        print(f"  {bin_name}: {r['detected']}/{r['total']} = {r['recall']*100:.2f}%")

    print("\nΔ召回率 (P2 - 基线):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta = p2_recall_s2[bin_name]['recall'] - base_recall_s2[bin_name]['recall']
        delta_count = p2_recall_s2[bin_name]['detected'] - base_recall_s2[bin_name]['detected']
        print(f"  {bin_name}: {delta*100:+.2f} pp ({delta_count:+d} 个)")

    # 双seed平均
    print("\n\n=== 双Seed平均 ===")
    print("\nΔ召回率 (P2 - 基线, 两个种子平均):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta_s1 = p2_recall[bin_name]['recall'] - base_recall[bin_name]['recall']
        delta_s2 = p2_recall_s2[bin_name]['recall'] - base_recall_s2[bin_name]['recall']
        avg_delta = (delta_s1 + delta_s2) / 2
        print(f"  {bin_name}: {avg_delta*100:+.2f} pp (s1: {delta_s1*100:+.2f}, s2: {delta_s2*100:+.2f})")

if __name__ == '__main__':
    main()
