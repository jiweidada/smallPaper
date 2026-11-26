import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor
from typing import Optional, Tuple, Union
from mmdet.registry import MODELS
from .bbox_head import BBoxHead


class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=0, keepdim=True) + self.eps)
        x = self.gamma * (x * Nx) + self.beta
        return x


# class HybridVarianceSpatialGateTop2(nn.Module):
#     def __init__(self, num_experts=8, alpha=0.5):
#         super().__init__()
#         self.num_experts = num_experts
#         self.alpha = alpha

#     def forward(self, x):
#         B, C, H, W = x.shape
#         var_per_channel = x.var(dim=[2, 3])
#         var_score = var_per_channel.mean(dim=1)
#         spatial_map = x.mean(dim=1)
#         flat_map = spatial_map.view(B, -1)
#         attn = F.softmax(flat_map, dim=1)
#         indices = torch.arange(H * W, device=x.device).float()
#         spatial_score = (attn * indices).sum(dim=1) / (H * W)
        
        
        
#         var_score_norm = (var_score - var_score.min()) / (var_score.max() - var_score.min() + 1e-6)
#         hybrid_score = self.alpha * var_score_norm + (1 - self.alpha) * spatial_score
#         logits = hybrid_score.unsqueeze(1) + torch.randn(B, self.num_experts, device=x.device) * 0.01
        
#         top2 = torch.topk(logits, k=2, dim=1)
#         return top2.indices
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- 통계량 추출 함수들 ---------------
def multi_scale_stats(x, scales=[1, 2, 4]):
    B, C, H, W = x.shape
    stats_list = []
    for scale in scales:
        pooled = F.adaptive_avg_pool2d(x, (scale, scale))
        attn = F.softmax(pooled.mean(dim=1).view(B, -1), dim=1)
        entropy = -(attn * attn.clamp(min=1e-8).log()).sum(dim=1, keepdim=True)
        max_attn = attn.max(dim=1, keepdim=True).values
        variance = attn.var(dim=1, keepdim=True)
        stats_list.extend([entropy, max_attn, variance])
    return torch.cat(stats_list, dim=1)  # [B, 9]

def positional_embedding(H, W, device):
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    pos_embed = torch.stack([x_coords, y_coords], dim=0).view(2, -1)
    return pos_embed  # [2, H*W]

def dynamic_topk(logits, min_k=1, threshold=0.7):
    probs = logits.softmax(dim=1)
    cumulative_probs, indices = probs.sort(dim=1, descending=True)
    cumulative_sum = cumulative_probs.cumsum(dim=1)
    mask = (cumulative_sum <= threshold).float()
    mask[:, :min_k] = 1
    return mask, indices

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


# --------------- Small Object-Aware Gating Network ---------------

class TaskAwareGating(nn.Module):
    def __init__(self, num_experts=8, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk

        self.gate_cls = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
        )

        self.gate_reg = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def compute_cls_stats(self, x):
        B, C, H, W = x.shape
        attn = F.softmax(x.mean(dim=1).view(B, -1), dim=1)

        entropy = -(attn * attn.clamp(min=1e-8).log()).sum(dim=1, keepdim=True)
        max_attn = attn.max(dim=1, keepdim=True).values
        attn_variance = attn.var(dim=1, keepdim=True)

        mean_intensity = x.mean(dim=(1, 2, 3), keepdim=True).view(B, 1)
        contrast = x.std(dim=(1, 2, 3), keepdim=True).view(B, 1)

        attn_median = attn.median(dim=1, keepdim=True).values
        attn_skewness = ((attn - attn.mean(dim=1, keepdim=True)) ** 3).mean(dim=1, keepdim=True) / (attn.std(dim=1, keepdim=True)**3 + 1e-8)
        attn_kurtosis = ((attn - attn.mean(dim=1, keepdim=True)) ** 4).mean(dim=1, keepdim=True) / (attn.var(dim=1, keepdim=True)**2 + 1e-8)

        stats_cls = torch.cat([
            entropy, max_attn, attn_variance, mean_intensity,
            contrast, attn.mean(dim=1, keepdim=True), attn_median,
            attn_skewness, attn_kurtosis
        ], dim=1)

        return stats_cls


    def compute_reg_stats(self, x):
        B, C, H, W = x.shape
        attn = F.softmax(x.mean(dim=1).view(B, -1), dim=1)

        top_attn, _ = attn.topk(k=int(0.2 * H * W), dim=1)
        region_attn_mean = top_attn.mean(dim=1, keepdim=True)

        x_coords = torch.linspace(-1, 1, W, device=x.device).repeat(H, 1).view(-1)
        y_coords = torch.linspace(-1, 1, H, device=x.device).repeat_interleave(W)

        spatial_x = (attn * x_coords).sum(dim=1, keepdim=True)
        spatial_y = (attn * y_coords).sum(dim=1, keepdim=True)

        object_area_ratio = (attn > attn.mean(dim=1, keepdim=True)).float().sum(dim=1, keepdim=True) / (H * W)
        attn_variance = attn.var(dim=1, keepdim=True)

        attn_median = attn.median(dim=1, keepdim=True).values
        attn_skewness = ((attn - attn.mean(dim=1, keepdim=True)) ** 3).mean(dim=1, keepdim=True) / (attn.std(dim=1, keepdim=True)**3 + 1e-8)
        attn_kurtosis = ((attn - attn.mean(dim=1, keepdim=True)) ** 4).mean(dim=1, keepdim=True) / (attn.var(dim=1, keepdim=True)**2 + 1e-8)

        attn_mean = attn.mean(dim=1, keepdim=True)

        stats_reg = torch.cat([
            region_attn_mean, spatial_x, spatial_y, object_area_ratio,
            attn_variance, attn_mean, attn_median, attn_skewness, attn_kurtosis
        ], dim=1)

        return stats_reg

    def forward(self, x):
        stats_cls = self.compute_cls_stats(x)
        stats_reg = self.compute_reg_stats(x)

        logits_cls = self.gate_cls(stats_cls)
        logits_reg = self.gate_reg(stats_reg)

        topk_val_cls, topk_idx_cls = logits_cls.topk(self.topk, dim=1)
        topk_val_reg, topk_idx_reg = logits_reg.topk(self.topk, dim=1)

        mask_cls = torch.zeros_like(logits_cls).scatter(1, topk_idx_cls, 1.0)
        mask_reg = torch.zeros_like(logits_reg).scatter(1, topk_idx_reg, 1.0)

        routing_weight_cls = F.softmax(logits_cls * mask_cls, dim=1)
        routing_weight_reg = F.softmax(logits_reg * mask_reg, dim=1)

        return (routing_weight_cls, topk_idx_cls), (routing_weight_reg, topk_idx_reg)



@MODELS.register_module()
class ConvFCBBoxHead_MoE(BBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args, **kwargs):
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.num_experts = 8
        self.topk = 2
        self.input_dim = 12544
        self.hidden_dim = 1024
        self.reduced_dim = 512

        self.use_handcrafted_gate = True
        self.hybrid_gate = TaskAwareGating(self.num_experts, self.topk)

        # self.class_gate = ClassifierHead(256, 8)


        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ) for _ in range(self.num_experts)
        ])

        # 커스텀 weight 초기화
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.shard_experts_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.shard_experts_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.grn = GRN(self.num_experts)
        self.gelu = nn.GELU()

        # nn.init.xavier_uniform_(self.experts_fc1.weight); nn.init.zeros_(self.experts_fc1.bias)
        # nn.init.xavier_uniform_(self.experts_fc2.weight); nn.init.zeros_(self.experts_fc2.bias)
        nn.init.xavier_uniform_(self.shard_experts_fc1.weight); nn.init.zeros_(self.shard_experts_fc1.bias)
        nn.init.xavier_uniform_(self.shard_experts_fc2.weight); nn.init.zeros_(self.shard_experts_fc2.bias)

        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_shared_convs, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(
            self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(
            self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        if self.with_cls:
            cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            reg_predictor_cfg_.update(in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

    def _add_conv_fc_branch(self, num_branch_convs, num_branch_fcs, in_channels, is_shared=False):
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        for i in range(num_branch_convs):
            conv_in_channels = last_layer_dim if i == 0 else self.conv_out_channels
            branch_convs.append(ConvModule(conv_in_channels, self.conv_out_channels, 3, padding=1,
                                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        last_layer_dim = self.conv_out_channels if num_branch_convs > 0 else last_layer_dim

        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        # y = x.clone()  # [B, 256, 7, 7]
        # x = x.flatten(1)  # [B, 12544]
        # class_logits = self.class_gate(x)
        
        (routing_weight_cls, topk_idx_cls), (routing_weight_reg, topk_idx_reg) = self.hybrid_gate(x)

        x_flat = x.flatten(1)  # [B, C*H*W]
        B = x_flat.size(0)

        # 모든 전문가 출력
        all_expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B, E, D]

        # 분류 전문가
        selected_cls = torch.gather(all_expert_outputs, 1, topk_idx_cls.unsqueeze(-1).expand(-1, -1, all_expert_outputs.shape[-1]))
        out_cls = torch.einsum('bk,bkd->bd', routing_weight_cls.gather(1, topk_idx_cls), selected_cls)

        # 회귀 전문가
        selected_reg = torch.gather(all_expert_outputs, 1, topk_idx_reg.unsqueeze(-1).expand(-1, -1, all_expert_outputs.shape[-1]))
        out_reg = torch.einsum('bk,bkd->bd', routing_weight_reg.gather(1, topk_idx_reg), selected_reg)

        # Shared path
        shared = self.shard_experts_fc2(self.gelu(self.shard_experts_fc1(x_flat)))

        x_cls = out_cls + shared
        x_reg = out_reg + shared

        # if self.use_handcrafted_gate:
        #     topk_indices, gate_probs = self.hybrid_gate(y)  # ✅ 수정된 부분
        # else:
        #     gate_logits = self.gate(x)
        #     gate_logits = self.grn(gate_logits)
        #     gate_probs = F.softmax(gate_logits, dim=-1)
        #     gate_probs, topk_indices = torch.topk(gate_probs, self.topk, dim=-1)

        # expert_outputs = self.experts_fc1(x)
        # expert_outputs = self.gelu(expert_outputs)
        # expert_outputs = self.experts_fc2(expert_outputs)
        # expert_outputs = expert_outputs.view(-1, self.num_experts, self.hidden_dim)

        # selected_expert_outputs = torch.gather(
        #     expert_outputs, 1,
        #     topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))

        # output = torch.einsum('bk,bkd->bd', gate_probs, selected_expert_outputs)

        # shared_ex = self.shard_experts_fc1(x)
        # shared_ex = self.gelu(shared_ex)
        # shared_ex = self.shard_experts_fc2(shared_ex)
        # x = output + shared_ex

        # x_cls, x_reg = x.clone(), x.clone()
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
