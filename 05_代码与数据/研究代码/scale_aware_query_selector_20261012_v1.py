"""
候选分配尺度感知调整实验代码
目标：通过保护不同尺度的候选数量，改善中大目标覆盖不足问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleAwareQuerySelector(nn.Module):
    """
    尺度感知的候选选择器
    在编码器输出选择Top300 query时，考虑P3/P4/P5的分配平衡
    """
    def __init__(self, mode='adaptive', p3_base=150, p4_base=100, p5_base=50,
                 flexibility=0.2, total_queries=300):
        """
        Args:
            mode: 'fixed' - 固定配额, 'adaptive' - 自适应配额, 'top300' - 原生Top300
            p3_base: P3基础配额
            p4_base: P4基础配额
            p5_base: P5基础配额
            flexibility: 自适应模式下的浮动比例（±20%）
            total_queries: 总query数量
        """
        super().__init__()
        self.mode = mode
        self.p3_base = p3_base
        self.p4_base = p4_base
        self.p5_base = p5_base
        self.flexibility = flexibility
        self.total_queries = total_queries

        assert p3_base + p4_base + p5_base == total_queries, \
            f"配额总和{p3_base+p4_base+p5_base}必须等于总query数{total_queries}"

    def forward(self, class_logits_p3, class_logits_p4, class_logits_p5):
        """
        Args:
            class_logits_p3: [B, H3*W3, num_classes] P3的类别logits
            class_logits_p4: [B, H4*W4, num_classes] P4的类别logits
            class_logits_p5: [B, H5*W5, num_classes] P5的类别logits

        Returns:
            selected_indices: [B, 300] 全局索引（跨P3/P4/P5）
            level_labels: [B, 300] 每个query来自哪个level (0=P3, 1=P4, 2=P5)
        """
        batch_size = class_logits_p3.shape[0]
        device = class_logits_p3.device

        # 每个位置的最大类别分数
        max_scores_p3, _ = class_logits_p3.max(dim=-1)  # [B, N3]
        max_scores_p4, _ = class_logits_p4.max(dim=-1)  # [B, N4]
        max_scores_p5, _ = class_logits_p5.max(dim=-1)  # [B, N5]

        n3 = max_scores_p3.shape[1]
        n4 = max_scores_p4.shape[1]
        n5 = max_scores_p5.shape[1]

        if self.mode == 'top300':
            # 原生Top300：直接全局排序
            all_scores = torch.cat([max_scores_p3, max_scores_p4, max_scores_p5], dim=1)
            _, global_indices = torch.topk(all_scores, self.total_queries, dim=1)

            # 计算level labels
            level_labels = torch.zeros_like(global_indices)
            level_labels[global_indices >= n3 + n4] = 2  # P5
            level_labels[(global_indices >= n3) & (global_indices < n3 + n4)] = 1  # P4

            return global_indices, level_labels

        elif self.mode == 'fixed':
            # 固定配额：每个level独立选Top-K
            selected_indices_list = []
            level_labels_list = []

            # P3
            _, p3_local_idx = torch.topk(max_scores_p3, self.p3_base, dim=1)
            selected_indices_list.append(p3_local_idx)
            level_labels_list.append(torch.zeros_like(p3_local_idx))

            # P4
            _, p4_local_idx = torch.topk(max_scores_p4, self.p4_base, dim=1)
            p4_global_idx = p4_local_idx + n3
            selected_indices_list.append(p4_global_idx)
            level_labels_list.append(torch.ones_like(p4_global_idx))

            # P5
            _, p5_local_idx = torch.topk(max_scores_p5, self.p5_base, dim=1)
            p5_global_idx = p5_local_idx + n3 + n4
            selected_indices_list.append(p5_global_idx)
            level_labels_list.append(torch.full_like(p5_global_idx, 2))

            selected_indices = torch.cat(selected_indices_list, dim=1)
            level_labels = torch.cat(level_labels_list, dim=1)

            return selected_indices, level_labels

        elif self.mode == 'adaptive':
            # 自适应配额：根据各level的分数分布动态调整
            selected_indices_list = []
            level_labels_list = []

            # 计算各level的平均分数（作为质量指标）
            avg_score_p3 = max_scores_p3.mean(dim=1)  # [B]
            avg_score_p4 = max_scores_p4.mean(dim=1)
            avg_score_p5 = max_scores_p5.mean(dim=1)

            # 归一化为分配权重
            total_avg = avg_score_p3 + avg_score_p4 + avg_score_p5
            weight_p3 = avg_score_p3 / (total_avg + 1e-8)
            weight_p4 = avg_score_p4 / (total_avg + 1e-8)
            weight_p5 = avg_score_p5 / (total_avg + 1e-8)

            for b in range(batch_size):
                # 基础配额 ± flexibility范围内调整
                flex_range = int(self.flexibility * self.p3_base)

                # 根据权重调整配额
                adjust_p3 = int((weight_p3[b] - 1/3) * flex_range * 3)
                adjust_p4 = int((weight_p4[b] - 1/3) * flex_range * 3)
                adjust_p5 = int((weight_p5[b] - 1/3) * flex_range * 3)

                # 归一化调整量，确保总和为300
                total_adjust = adjust_p3 + adjust_p4 + adjust_p5
                if total_adjust != 0:
                    adjust_p3 -= total_adjust // 3
                    adjust_p4 -= total_adjust // 3
                    adjust_p5 -= total_adjust - 2 * (total_adjust // 3)

                quota_p3 = max(50, min(250, self.p3_base + adjust_p3))  # 限制极端值
                quota_p4 = max(20, min(150, self.p4_base + adjust_p4))
                quota_p5 = max(10, min(100, self.p5_base + adjust_p5))

                # 精确调整为300
                total = quota_p3 + quota_p4 + quota_p5
                if total != self.total_queries:
                    quota_p3 += self.total_queries - total

                # 选择各level的Top-K
                _, p3_idx = torch.topk(max_scores_p3[b], quota_p3)
                _, p4_idx = torch.topk(max_scores_p4[b], quota_p4)
                _, p5_idx = torch.topk(max_scores_p5[b], quota_p5)

                p4_idx_global = p4_idx + n3
                p5_idx_global = p5_idx + n3 + n4

                batch_indices = torch.cat([p3_idx, p4_idx_global, p5_idx_global])
                batch_levels = torch.cat([
                    torch.zeros_like(p3_idx),
                    torch.ones_like(p4_idx),
                    torch.full_like(p5_idx, 2)
                ])

                selected_indices_list.append(batch_indices.unsqueeze(0))
                level_labels_list.append(batch_levels.unsqueeze(0))

            selected_indices = torch.cat(selected_indices_list, dim=0)
            level_labels = torch.cat(level_labels_list, dim=0)

            return selected_indices, level_labels

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============ 使用示例 ============

def integrate_into_encoder(encoder, selector):
    """
    将ScaleAwareQuerySelector集成到现有encoder中
    需要修改encoder的query选择部分
    """
    # 保存原始的query选择逻辑
    original_select_queries = encoder.select_queries

    def new_select_queries(class_logits):
        # 假设class_logits已经按P3/P4/P5拆分
        # 实际集成时需要根据encoder的具体结构调整

        # 拆分成三个level（需要知道各level的位置数量）
        n3 = 6400  # 80*80
        n4 = 1600  # 40*40
        n5 = 400   # 20*20

        logits_p3 = class_logits[:, :n3, :]
        logits_p4 = class_logits[:, n3:n3+n4, :]
        logits_p5 = class_logits[:, n3+n4:, :]

        selected_indices, level_labels = selector(logits_p3, logits_p4, logits_p5)

        return selected_indices, level_labels

    encoder.select_queries = new_select_queries


if __name__ == '__main__':
    # 测试代码
    batch_size = 2
    num_classes = 10

    # 模拟P3/P4/P5的logits
    logits_p3 = torch.randn(batch_size, 6400, num_classes)
    logits_p4 = torch.randn(batch_size, 1600, num_classes)
    logits_p5 = torch.randn(batch_size, 400, num_classes)

    # 测试三种模式
    for mode in ['top300', 'fixed', 'adaptive']:
        print(f"\n=== 测试模式: {mode} ===")
        selector = ScaleAwareQuerySelector(mode=mode)
        indices, levels = selector(logits_p3, logits_p4, logits_p5)

        print(f"输出shape: indices={indices.shape}, levels={levels.shape}")

        # 统计各level的数量
        for b in range(batch_size):
            counts = [(levels[b] == i).sum().item() for i in range(3)]
            print(f"Batch {b}: P3={counts[0]}, P4={counts[1]}, P5={counts[2]}")
