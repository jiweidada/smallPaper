import torch
import torch.nn as nn
import torch.nn.functional as F
import os, re
from .common import FrozenBatchNorm2d
from ..core import register

from .hgnetv2 import HGNetv2

from ..extre_module.custom_nn.stem.LoG import LoGStem
from ..extre_module.custom_nn.stem.SRFD import SRFD

__all__ = ['HGNetv2_SRFDStem']


@register()
class HGNetv2_SRFDStem(HGNetv2):
    """
    使用 SRFD Stem 的 HGNetv2 变体

    该类继承自 HGNetv2，但使用 SRFD (Spatial-Rich Feature Decomposition) 作为 stem 模块
    来替代原始的 StemBlock，以提供更好的初始特征提取能力。

    Args:
        name (str): 模型配置名称 (如 'B0', 'B1', 等)
        use_lab (bool): 是否使用可学习仿射变换模块
        return_idx (list): 返回特征的阶段索引
        freeze_stem_only (bool): 是否只冻结 stem 参数
        freeze_at (int): 冻结的阶段数量
        freeze_norm (bool): 是否冻结归一化层
        pretrained (bool): 是否加载预训练权重
        agg (str or list): 注意力聚合方式，支持字符串或列表
        hierarchical_attention (bool): 是否使用分层注意力策略
        local_model_dir (str): 本地模型权重目录
    """

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 agg='se',
                 hierarchical_attention=False,  # 新增：是否使用分层注意力策略
                 local_model_dir='weight/hgnetv2/'):
        # 调用父类初始化，传递所有参数包括新增的 hierarchical_attention
        super().__init__(
            name=name,
            use_lab=use_lab,
            return_idx=return_idx,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            pretrained=pretrained,
            agg=agg,
            hierarchical_attention=hierarchical_attention,  # 传递新参数
            local_model_dir=local_model_dir
        )

        # 使用 SRFD stem 替换原来的 StemBlock
        # SRFD 提供了更丰富的空间特征分解能力
        stem_channels = self.arch_configs[name]['stem_channels']
        self.stem = SRFD(stem_channels[0], stem_channels[2])

        # print(f"已将 StemBlock 替换为 SRFD，输入通道：{stem_channels[0]}，输出通道：{stem_channels[2]}")

# python tools/benchmark/get_info.py -c configs/test/deim_hgnetv2_n_MultiLoGStem_Simam_DilatedGCSA_visdrone.yml