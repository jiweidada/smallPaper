# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.registry import MODELS
from .bbox_head import BBoxHead
import torch

import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, dim]
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [batch_size, 1]
        Nx = Gx / (Gx.mean(dim=0, keepdim=True) + self.eps)  # [batch_size, 1]
        x = self.gamma * (x * Nx) + self.beta
        return x
    

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):

        return x.reshape(x.shape[0], x.shape[1], 1, 1)
    
#############
@MODELS.register_module()
class ConvFCBBoxHead_MoE(BBoxHead):

    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 2,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
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


        ######### 추가 
        self.num_experts = 8
        self.topk = 4
        self.noise_std = 0.1
        self.input_dim = 12544
        self.hidden_dim = 1024
        self.reduced_dim = 512
 
        self.gate_channel = 256
        self.gate = nn.Sequential( #classficiation
            nn.Linear(self.input_dim, self.reduced_dim),
            nn.SELU(),
            nn.Linear(self.reduced_dim, self.num_experts)  # top-1 gating 권장
        )

     
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Use Xavier uniform initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero
        
        
        self.experts_fc1 = nn.Linear(self.input_dim,self.hidden_dim * self.num_experts)
        self.experts_fc2 = nn.Linear(self.hidden_dim * self.num_experts, self.hidden_dim * self.num_experts )
        
        
        self.shard_experts_fc1 = nn.Linear(self.input_dim,self.hidden_dim )
        self.shard_experts_fc2 = nn.Linear(self.hidden_dim , self.hidden_dim)
        self.grn = GRN(self.num_experts)

        # self.grn = GRN(self.num_experts)
        nn.init.xavier_uniform_(self.experts_fc1.weight)
        nn.init.zeros_(self.experts_fc1.bias)
        
        nn.init.xavier_uniform_(self.experts_fc2.weight)
        nn.init.zeros_(self.experts_fc2.bias)
        
        
        nn.init.xavier_uniform_(self.shard_experts_fc1.weight)
        nn.init.zeros_(self.shard_experts_fc1.bias)
        
        nn.init.xavier_uniform_(self.shard_experts_fc2.weight)
        nn.init.zeros_(self.shard_experts_fc2.bias)
        #####
        self.gelu = nn.GELU()
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else \
                box_dim * self.num_classes
            reg_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_predictor_cfg_, (dict, ConfigDict)):
                reg_predictor_cfg_.update(
                    in_features=self.reg_last_dim, out_features=out_dim_reg)
            self.fc_reg = MODELS.build(reg_predictor_cfg_)

        if init_cfg is None:
   
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs: int,
                            num_branch_fcs: int,
                            in_channels: int,
                            is_shared: bool = False) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        
        # if isinstance(x, (tuple, list)):
        #     x = x[0]  #
        x = x.flatten(1)

        
        gate_logits = self.gate(x)  # [batch_size, num_experts]


        # if self.noise_std > 0:
        #     noise = torch.randn_like(gate_logits) * self.noise_std
        #     gate_logits = gate_logits + noise  # 노이즈 추가

        
        gate_logits = self.grn(gate_logits)

        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # Top-K 전문가 선택
        topk_probs, topk_indices = torch.topk(gate_probs, self.topk, dim=-1)  # [batch_size, topk]

        # 전문가 출력 계산
        expert_outputs = self.experts_fc1(x)  # [batch_size, num_experts * hidden_dim]
        expert_outputs = self.gelu(expert_outputs)  # [batch_size, num_experts * hidden_dim]
        expert_outputs = self.experts_fc2(expert_outputs)  # [batch_size, num_experts * output_dim]        
        expert_outputs = expert_outputs.view(-1, self.num_experts, self.hidden_dim)  # [batch_size, num_experts, output_dim]

        # 선택된 전문가의 출력 추출
        selected_expert_outputs = torch.gather(
            expert_outputs,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )  # [batch_size, topk, output_dim]

        # 최종 출력 계산
        output = torch.einsum('bk,bkd->bd', topk_probs, selected_expert_outputs)  # [batch_size, output_dim]


        shared_ex = self.shard_experts_fc1(x)
        shared_ex = self.gelu(shared_ex)
        shared_ex = self.shard_experts_fc2(shared_ex)
        
        x = output + shared_ex


        ##
        x_cls = x
        x_reg = x

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
