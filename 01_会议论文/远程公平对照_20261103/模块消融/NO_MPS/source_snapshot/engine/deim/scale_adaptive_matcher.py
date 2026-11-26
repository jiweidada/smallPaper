"""
Scale-Adaptive Dynamic Matching for DEIM
核心贡献：针对航拍场景的尺度自适应匹配策略
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..core import register


@register()
class ScaleAdaptiveMatcher(nn.Module):
    """尺度自适应动态匹配器 - 论文核心贡献"""

    def __init__(self,
                 cost_class=1,
                 cost_bbox=5,
                 cost_giou=2,
                 scale_thresholds=[0.0005, 0.005, 0.05],  # 面积阈值：超小、小、中大目标
                 scale_weights=None,  # 各尺度的权重调整
                 alpha=0.25,
                 gamma=2.0):
        """
        Args:
            cost_class, cost_bbox, cost_giou: 基础代价权重
            scale_thresholds: 尺度分组阈值 [超小目标, 小目标, 中大目标]
            scale_weights: 各尺度的权重调整矩阵 [[cls, bbox, giou], ...]
            alpha, gamma: focal loss参数
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.scale_thresholds = scale_thresholds
        self.alpha = alpha
        self.gamma = gamma

        # 默认尺度自适应权重设计
        if scale_weights is None:
            self.scale_weights = [
                [1.5, 0.8, 1.2],  # 超小目标：重分类，轻定位
                [1.2, 1.0, 1.1],  # 小目标：平衡调整
                [1.0, 1.2, 1.0],  # 中大目标：重定位
            ]
        else:
            self.scale_weights = scale_weights

        assert len(self.scale_weights) == len(self.scale_thresholds), \
            "权重数量必须与尺度阈值数量一致"

    def forward(self, outputs, targets):
        """主匹配入口"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # 1. 将目标按尺度分组
            scale_groups = self._group_targets_by_scale(targets)

            # 2. 分层匹配策略
            all_indices = []
            for b in range(bs):
                batch_indices = self._hierarchical_matching_single_batch(
                    outputs, targets[b], scale_groups[b], b
                )
                all_indices.append(batch_indices)

            return {'indices': all_indices}

    def _group_targets_by_scale(self, targets):
        """将目标按面积大小分组"""
        batch_scale_groups = []

        for target in targets:
            if len(target['boxes']) == 0:
                # 空目标情况
                scale_groups = [[] for _ in range(len(self.scale_thresholds))]
                batch_scale_groups.append(scale_groups)
                continue

            # 计算目标面积
            boxes = target['boxes']  # [N, 4] (cx, cy, w, h)
            areas = boxes[:, 2] * boxes[:, 3]  # w * h

            # 按阈值分组
            scale_groups = [[] for _ in range(len(self.scale_thresholds))]

            for i, area in enumerate(areas):
                # 确定目标属于哪个尺度组
                scale_level = self._get_scale_level(area.item())
                scale_groups[scale_level].append(i)

            batch_scale_groups.append(scale_groups)

        return batch_scale_groups

    def _get_scale_level(self, area):
        """根据面积确定尺度级别"""
        for i, threshold in enumerate(self.scale_thresholds):
            if area <= threshold:
                return i
        return len(self.scale_thresholds) - 1  # 最大尺度组

    def _hierarchical_matching_single_batch(self, outputs, target, scale_groups, batch_idx):
        """单个batch的分层匹配"""
        if len(target['boxes']) == 0:
            # 空目标情况
            return (torch.as_tensor([], dtype=torch.int64, device=target['boxes'].device),
                    torch.as_tensor([], dtype=torch.int64, device=target['boxes'].device))

        num_queries = outputs["pred_logits"].shape[1]

        # 收集所有匹配结果
        all_query_indices = []
        all_target_indices = []

        # 对每个尺度组进行匹配
        for scale_level, target_indices_in_group in enumerate(scale_groups):
            if len(target_indices_in_group) == 0:
                continue

            # 获取该尺度的自适应权重
            scale_weight = self.scale_weights[scale_level]
            cost_class_adapt = self.cost_class * scale_weight[0]
            cost_bbox_adapt = self.cost_bbox * scale_weight[1]
            cost_giou_adapt = self.cost_giou * scale_weight[2]

            # 构建该尺度组的子目标
            sub_target = {
                'labels': target['labels'][target_indices_in_group],
                'boxes': target['boxes'][target_indices_in_group]
            }

            # 计算代价矩阵
            cost_matrix = self._compute_cost_matrix(
                outputs, [sub_target], batch_idx,
                cost_class_adapt, cost_bbox_adapt, cost_giou_adapt
            )

            # 匈牙利匹配
            query_indices, sub_target_indices = linear_sum_assignment(cost_matrix.cpu())

            # 将子目标索引映射回原始目标索引
            original_target_indices = [target_indices_in_group[i] for i in sub_target_indices]

            all_query_indices.extend(query_indices.tolist())
            all_target_indices.extend(original_target_indices)

        # 转换为张量
        query_indices = torch.as_tensor(all_query_indices, dtype=torch.int64, device=target['boxes'].device)
        target_indices = torch.as_tensor(all_target_indices, dtype=torch.int64, device=target['boxes'].device)

        return query_indices, target_indices

    def _compute_cost_matrix(self, outputs, targets, batch_idx, cost_class, cost_bbox, cost_giou):
        """计算代价矩阵"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 提取单个batch的预测 - FIXED: Removed unnecessary flatten
        out_prob = outputs["pred_logits"][batch_idx].sigmoid()  # [num_queries, num_classes]
        out_bbox = outputs["pred_boxes"][batch_idx]  # [num_queries, 4]

        # 目标
        tgt_ids = targets[0]["labels"]
        tgt_bbox = targets[0]["boxes"]

        # 分类代价 - 使用focal loss风格的代价
        alpha = self.alpha
        gamma = self.gamma
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        if tgt_ids.dim() > 1:
            tgt_ids = tgt_ids.flatten()
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # 回归代价 - L1 distance
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # IoU代价 - 使用GIoU
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # 总代价
        C = cost_bbox * cost_bbox + cost_class * cost_class + cost_giou * cost_giou
        C = C.view(num_queries, -1)

        return C