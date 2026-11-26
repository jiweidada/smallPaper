"""
低通P2对照实验代码
目标：验证P2桥接的收益是否来自相位细节，还是P2层级的语义信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv_bn_relu(cin, cout, kernel=1, stride=1, groups=1, dilation=1):
    """基础卷积块"""
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(cin, cout, kernel, stride,
                           padding=dilation * (kernel // 2), dilation=dilation,
                           groups=groups, bias=False)),
        ('bn', nn.BatchNorm2d(cout)),
        ('act', nn.ReLU(inplace=False)),
    ]))


class P2ToP3Residual_Original(nn.Module):
    """原始P2桥接：使用pixel_unshuffle保留相位信息"""
    def __init__(self, p2_channels=64, p3_channels=256, width=16):
        super().__init__()
        self.reduce = conv_bn_relu(p2_channels, width)
        self.restore = nn.Conv2d(width * 4, p3_channels, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, p3):
        detail = self.reduce(p2)  # [B, width, H, W]
        h, w = detail.shape[2:]

        # pixel_unshuffle: 保留2×2内的相位信息
        # [B, width, H, W] -> [B, width*4, H/2, W/2]
        detail = F.pixel_unshuffle(detail, 2)

        residual = self.restore(detail)  # [B, p3_channels, H/2, W/2]
        return p3 + residual


class P2ToP3Residual_LowPass(nn.Module):
    """
    低通P2对照：消除相位信息，只保留P2层级的语义

    关键修改：用平均池化+上采样替换pixel_unshuffle
    - 平均池化破坏了2×2内的相位关系
    - 上采样恢复空间尺寸，但相位信息已丢失
    - 保持相同的参数量和计算流程
    """
    def __init__(self, p2_channels=64, p3_channels=256, width=16):
        super().__init__()
        self.reduce = conv_bn_relu(p2_channels, width)

        # 保持与原版相同的参数量：width*4 -> p3_channels
        self.restore = nn.Conv2d(width * 4, p3_channels, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, p3):
        detail = self.reduce(p2)  # [B, width, H, W]
        h, w = detail.shape[2:]

        # === 关键修改：低通滤波替代pixel_unshuffle ===
        # Step 1: 2×2平均池化 - 破坏sub-cell相位信息
        detail_lowpass = F.avg_pool2d(detail, kernel_size=2, stride=1, padding=0)
        # [B, width, H-1, W-1]

        # Step 2: 双线性上采样回原尺寸 - 空间对齐但相位已丢失
        detail_lowpass = F.interpolate(
            detail_lowpass,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        # [B, width, H, W]

        # Step 3: 复制4份模拟pixel_unshuffle的通道扩展
        # 这保证了restore层的输入通道数匹配
        detail_expanded = torch.cat([detail_lowpass] * 4, dim=1)
        # [B, width*4, H, W]

        # Step 4: 下采样到P3尺寸
        detail_expanded = F.avg_pool2d(detail_expanded, kernel_size=2, stride=2)
        # [B, width*4, H/2, W/2]

        residual = self.restore(detail_expanded)  # [B, p3_channels, H/2, W/2]
        return p3 + residual


class P2ToP3Residual_LowPass_V2(nn.Module):
    """
    低通P2对照 V2：更激进的相位破坏

    直接在P2上做低通滤波，然后走正常的pixel_unshuffle流程
    如果这个版本性能接近基线，说明相位确实重要
    如果接近原版，说明avg_pool没有充分破坏相位
    """
    def __init__(self, p2_channels=64, p3_channels=256, width=16):
        super().__init__()
        self.reduce = conv_bn_relu(p2_channels, width)
        self.restore = nn.Conv2d(width * 4, p3_channels, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, p2, p3):
        # === 在reduce之前就破坏P2的高频信息 ===
        # 3×3高斯模糊核
        p2_lowpass = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)

        detail = self.reduce(p2_lowpass)  # [B, width, H, W]
        h, w = detail.shape[2:]

        # 后续与原版相同
        detail = F.pixel_unshuffle(detail, 2)
        residual = self.restore(detail)
        return p3 + residual


# ============ 对比测试 ============

def test_three_versions():
    """对比三个版本的输出差异"""
    batch_size = 2
    p2_channels = 64
    p3_channels = 256
    h_p2, w_p2 = 160, 160  # P2特征图尺寸
    h_p3, w_p3 = 80, 80    # P3特征图尺寸

    # 模拟输入
    p2 = torch.randn(batch_size, p2_channels, h_p2, w_p2)
    p3 = torch.randn(batch_size, p3_channels, h_p3, w_p3)

    # 三个版本
    original = P2ToP3Residual_Original()
    lowpass_v1 = P2ToP3Residual_LowPass()
    lowpass_v2 = P2ToP3Residual_LowPass_V2()

    # 同步参数（让restore层权重相同）
    lowpass_v1.restore.weight.data.copy_(original.restore.weight.data)
    lowpass_v2.restore.weight.data.copy_(original.restore.weight.data)
    lowpass_v1.reduce[0].weight.data.copy_(original.reduce[0].weight.data)
    lowpass_v2.reduce[0].weight.data.copy_(original.reduce[0].weight.data)

    # 推理
    with torch.no_grad():
        out_original = original(p2, p3)
        out_lowpass_v1 = lowpass_v1(p2, p3)
        out_lowpass_v2 = lowpass_v2(p2, p3)

    print("=== 三版本输出对比 ===")
    print(f"原版输出范围: [{out_original.min().item():.4f}, {out_original.max().item():.4f}]")
    print(f"低通V1输出范围: [{out_lowpass_v1.min().item():.4f}, {out_lowpass_v1.max().item():.4f}]")
    print(f"低通V2输出范围: [{out_lowpass_v2.min().item():.4f}, {out_lowpass_v2.max().item():.4f}]")

    # 计算差异
    diff_v1 = (out_original - out_lowpass_v1).abs().mean()
    diff_v2 = (out_original - out_lowpass_v2).abs().mean()

    print(f"\n原版 vs 低通V1 平均绝对差: {diff_v1.item():.6f}")
    print(f"原版 vs 低通V2 平均绝对差: {diff_v2.item():.6f}")

    print("\n参数量对比:")
    for name, model in [("原版", original), ("低通V1", lowpass_v1), ("低通V2", lowpass_v2)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} 参数")


# ============ 实验判断标准 ============

def interpret_results():
    """
    实验结果判断标准

    假设12轮双seed结果如下：
    - 原P2: AP +0.54pp (seed均值)
    - 低通P2: AP +X pp
    - 基线: 0 (参照)

    判断：
    1. 如果 X ≈ 0.54 (差异<0.2pp):
       -> 相位信息不重要，收益来自P2语义/容量
       -> 论文不应声称"相位保留改善精细定位"
       -> 应诚实描述为"多尺度特征融合"

    2. 如果 X ≈ 0 (差异<0.1pp):
       -> 相位信息确实重要
       -> 但需解释为何大目标收益最大（与"精细定位"矛盾）
       -> 可能是pixel_unshuffle的其他特性

    3. 如果 0.1 < X < 0.44:
       -> 部分收益来自相位，部分来自语义
       -> 需更细致的机制分析
    """
    print("""
=== 实验结果判断标准 ===

设定：
- 原P2桥接: ΔAP = +0.54pp (12轮双seed平均)
- 预训练基线: ΔAP = 0 (参照)
- 低通P2对照: ΔAP = ??? (待测)

判断逻辑：

【情况1】低通P2 ≈ 原P2 (差异<0.2pp)
  结论: 相位信息不重要
  机制: P2层级的语义信息或额外容量
  论文: 不应声称"相位保留"或"精细定位"
  诚实描述: "通过P2特征增强检测召回率"

【情况2】低通P2 ≈ 基线 (ΔAP<0.1pp)
  结论: 相位信息确实重要
  问题: 为何大目标收益最大？(与"精细定位"矛盾)
  后续: 需解释pixel_unshuffle的其他作用

【情况3】介于两者之间 (0.1 < ΔAP < 0.44)
  结论: 相位和语义都有贡献
  后续: 需更细致的消融(如不同pooling kernel)

当前证据倾向【情况1】：
- 尺寸分组：大目标收益最大(+7.9pp > +5.5pp)
- AP50 vs AP75：提升主要在召回率(比值0.5-0.7)
- 这两点都不支持"精细定位"假设
""")


if __name__ == '__main__':
    print("=" * 70)
    print("低通P2对照实验代码")
    print("=" * 70)

    test_three_versions()

    print("\n")
    interpret_results()
