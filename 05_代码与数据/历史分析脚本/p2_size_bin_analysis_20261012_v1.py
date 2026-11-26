#!/usr/bin/env python3
"""
P2桥接效果的尺寸分组分析
比较原P2接入 vs 预训练基线，按目标尺寸分bin计算召回率和AP贡献
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
    """加载GT标注，构建image_id到GT的映射"""
    with open(ANNO_PATH, 'r') as f:
        data = json.load(f)

    # 按image_id组织GT
    gt_by_image = defaultdict(list)
    for ann in data['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    return gt_by_image

def compute_size_bins(gt_by_image):
    """
    按缩放后尺寸分bin（640x640缩放）
    bins: <8px, 8-16px, 16-32px, 32+px（短边）
    返回每个GT的bin标签
    """
    gt_bins = {}  # ann_id -> bin_name
    bin_counts = defaultdict(int)

    for img_id, anns in gt_by_image.items():
        for ann in anns:
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

            gt_bins[ann['id']] = bin_name
            bin_counts[bin_name] += 1

    return gt_bins, bin_counts

def analyze_recall_by_bin(matching_data, gt_bins, iou_thresh=0.5):
    """
    按尺寸bin统计召回率
    matching_data: 每张图的匹配结果
    gt_bins: GT的尺寸分bin
    """
    detected_by_bin = defaultdict(set)
    total_by_bin = defaultdict(int)

    for img_result in matching_data:
        for gt_id, matches in img_result['gt_matches'].items():
            gt_id_int = int(gt_id)
            bin_name = gt_bins.get(gt_id_int, 'unknown')
            total_by_bin[bin_name] += 1

            # 检查是否有IoU>=阈值的匹配
            if matches:
                best_iou = max(m['iou'] for m in matches)
                if best_iou >= iou_thresh:
                    detected_by_bin[bin_name].add(gt_id_int)

    # 计算召回率
    recall_by_bin = {}
    for bin_name in total_by_bin:
        recall = len(detected_by_bin[bin_name]) / total_by_bin[bin_name] if total_by_bin[bin_name] > 0 else 0
        recall_by_bin[bin_name] = {
            'detected': len(detected_by_bin[bin_name]),
            'total': total_by_bin[bin_name],
            'recall': recall
        }

    return recall_by_bin

def main():
    print("=== P2桥接尺寸分组分析 ===\n")

    # 加载GT
    print("加载GT标注...")
    gt_by_image = load_annotations()
    gt_bins, bin_counts = compute_size_bins(gt_by_image)

    print("\n目标尺寸分布:")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        print(f"  {bin_name}: {bin_counts[bin_name]} 个")

    # 分析seed1
    print("\n\n--- Seed 1 ---")
    p2_match = load_matching(P2_SEED1 / "matching_all_area.json.gz")
    base_match = load_matching(BASELINE_SEED1 / "matching_all_area.json.gz")

    print("\n预训练基线 (IoU>=0.5):")
    base_recall = analyze_recall_by_bin(base_match, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = base_recall.get(bin_name, {})
        print(f"  {bin_name}: {r.get('detected', 0)}/{r.get('total', 0)} = {r.get('recall', 0)*100:.2f}%")

    print("\nP2接入 (IoU>=0.5):")
    p2_recall = analyze_recall_by_bin(p2_match, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = p2_recall.get(bin_name, {})
        print(f"  {bin_name}: {r.get('detected', 0)}/{r.get('total', 0)} = {r.get('recall', 0)*100:.2f}%")

    print("\nΔ召回率 (P2 - 基线):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta = p2_recall.get(bin_name, {}).get('recall', 0) - base_recall.get(bin_name, {}).get('recall', 0)
        print(f"  {bin_name}: {delta*100:+.2f} pp")

    # 分析seed2
    print("\n\n--- Seed 2 ---")
    p2_match_s2 = load_matching(P2_SEED2 / "matching_all_area.json.gz")
    base_match_s2 = load_matching(BASELINE_SEED2 / "matching_all_area.json.gz")

    print("\n预训练基线 (IoU>=0.5):")
    base_recall_s2 = analyze_recall_by_bin(base_match_s2, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = base_recall_s2.get(bin_name, {})
        print(f"  {bin_name}: {r.get('detected', 0)}/{r.get('total', 0)} = {r.get('recall', 0)*100:.2f}%")

    print("\nP2接入 (IoU>=0.5):")
    p2_recall_s2 = analyze_recall_by_bin(p2_match_s2, gt_bins, 0.5)
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        r = p2_recall_s2.get(bin_name, {})
        print(f"  {bin_name}: {r.get('detected', 0)}/{r.get('total', 0)} = {r.get('recall', 0)*100:.2f}%")

    print("\nΔ召回率 (P2 - 基线):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta = p2_recall_s2.get(bin_name, {}).get('recall', 0) - base_recall_s2.get(bin_name, {}).get('recall', 0)
        print(f"  {bin_name}: {delta*100:+.2f} pp")

    # 同时检查IoU>=0.75的严格定位
    print("\n\n=== 严格定位 (IoU>=0.75) ===")
    print("\n--- Seed 1 ---")
    base_recall_75 = analyze_recall_by_bin(base_match, gt_bins, 0.75)
    p2_recall_75 = analyze_recall_by_bin(p2_match, gt_bins, 0.75)

    print("\nΔ召回率 (P2 - 基线, IoU>=0.75):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta = p2_recall_75.get(bin_name, {}).get('recall', 0) - base_recall_75.get(bin_name, {}).get('recall', 0)
        print(f"  {bin_name}: {delta*100:+.2f} pp")

    print("\n--- Seed 2 ---")
    base_recall_75_s2 = analyze_recall_by_bin(base_match_s2, gt_bins, 0.75)
    p2_recall_75_s2 = analyze_recall_by_bin(p2_match_s2, gt_bins, 0.75)

    print("\nΔ召回率 (P2 - 基线, IoU>=0.75):")
    for bin_name in ['<8px', '8-16px', '16-32px', '>=32px']:
        delta = p2_recall_75_s2.get(bin_name, {}).get('recall', 0) - base_recall_75_s2.get(bin_name, {}).get('recall', 0)
        print(f"  {bin_name}: {delta*100:+.2f} pp")

if __name__ == '__main__':
    main()
