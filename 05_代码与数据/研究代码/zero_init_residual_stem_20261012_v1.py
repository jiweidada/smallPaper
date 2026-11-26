"""
零初始化残差Stem实验代码
目标：修复增强Stem的不稳定问题，通过零初始化残差分支实现暖启动
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv_bn_relu(cin, cout, kernel=3, stride=1, groups=1, padding=None):
    """基础卷积块"""
    if padding is None:
        padding = kernel // 2
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(cin, cout, kernel, stride, padding, groups=groups, bias=False)),
        ('bn', nn.BatchNorm2d(cout)),
        ('act', nn.ReLU(inplace=False)),
    ]))


class ZeroInitResidualStem(nn.Module):
    """
    零初始化残差Stem

    关键设计：
    1. 主路径：加载预训练权重，保持不变（或用很低学习率微调）
    2. 残差分支：随机初始化，学习补充信息
    3. 融合层：零初始化，确保初始时 output = 主路径

    优势：
    - 起点等价于预训练基线（消除初始退化风险）
    - 残差分支逐渐学习有用的补充特征
    - 如果12轮没收益，自然退化为基线（不会更差）
    """

    def __init__(self, in_channels=3, out_channels=32, pretrained_stem=None):
        """
        Args:
            in_channels: 输入通道数（RGB=3）
            out_channels: 输出通道数
            pretrained_stem: 预训练的原始Stem模块
        """
        super().__init__()

        # === 主路径：预训练Stem ===
        if pretrained_stem is not None:
            self.main_path = pretrained_stem
            # 可选：冻结主路径或使用低学习率
            # for param in self.main_path.parameters():
            #     param.requires_grad = False
        else:
            # 如果没有预训练，构建简单主路径
            self.main_path = nn.Sequential(
                conv_bn_relu(in_channels, out_channels // 2, kernel=3, stride=2),
                conv_bn_relu(out_channels // 2, out_channels, kernel=3, stride=2),
            )

        # === 残差分支：学习补充信息 ===
        # 分支1: 高频细节提取
        self.detail_branch = nn.Sequential(
            # 使用较小的卷积核捕捉细节
            conv_bn_relu(in_channels, 16, kernel=3, stride=1),
            conv_bn_relu(16, 16, kernel=3, stride=2),
            conv_bn_relu(16, 32, kernel=3, stride=2),
        )

        # 分支2: 通道注意力（可选）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 1),
            nn.Sigmoid()
        )

        # === 融合层：零初始化 ===
        # 将残差分支投影到主路径的输出空间
        self.fusion = nn.Conv2d(32, out_channels, kernel_size=1, bias=False)

        # 关键：零初始化融合层权重
        nn.init.zeros_(self.fusion.weight)

        # 可选：可学习的残差权重缩放因子
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播

        初始时：residual ≈ 0，输出 ≈ 主路径
        训练后：residual学到补充信息，输出 = 主路径 + α*残差
        """
        # 主路径输出
        main_out = self.main_path(x)  # [B, out_channels, H/4, W/4]

        # 残差分支
        detail = self.detail_branch(x)  # [B, 32, H/4, W/4]

        # 通道注意力（可选）
        attention_weights = self.attention(detail)
        detail = detail * attention_weights

        # 投影到主路径空间
        residual = self.fusion(detail)  # [B, out_channels, H/4, W/4]

        # 带缩放因子的残差连接
        # 初始时 residual_scale ≈ 0，输出 ≈ main_out
        output = main_out + self.residual_scale * residual

        return output


class ZeroInitResidualStem_Simple(nn.Module):
    """
    简化版：只有一个残差分支，无注意力

    用于快速验证零初始化策略是否有效
    """

    def __init__(self, in_channels=3, out_channels=32, pretrained_stem=None):
        super().__init__()

        # 主路径
        if pretrained_stem is not None:
            self.main_path = pretrained_stem
        else:
            self.main_path = nn.Sequential(
                conv_bn_relu(in_channels, out_channels // 2, kernel=3, stride=2),
                conv_bn_relu(out_channels // 2, out_channels, kernel=3, stride=2),
            )

        # 残差分支：与主路径平行的路径
        self.residual_branch = nn.Sequential(
            conv_bn_relu(in_channels, 16, kernel=3, stride=2),
            conv_bn_relu(16, out_channels, kernel=3, stride=2),
        )

        # 零初始化融合
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fusion.weight)

    def forward(self, x):
        main_out = self.main_path(x)
        residual = self.residual_branch(x)
        residual = self.fusion(residual)
        return main_out + residual


# ============ 验证零初始化效果 ============

def verify_zero_init():
    """验证初始时输出确实等于预训练基线"""
    batch_size = 2
    in_channels = 3
    out_channels = 32
    h, w = 640, 640

    # 模拟输入
    x = torch.randn(batch_size, in_channels, h, w)

    # 创建模拟的预训练stem
    pretrained_stem = nn.Sequential(
        conv_bn_relu(in_channels, out_channels // 2, kernel=3, stride=2),
        conv_bn_relu(out_channels // 2, out_channels, kernel=3, stride=2),
    )

    # 零初始化残差Stem
    zero_init_stem = ZeroInitResidualStem(in_channels, out_channels, pretrained_stem)

    # 前向传播
    with torch.no_grad():
        baseline_out = pretrained_stem(x)
        zero_init_out = zero_init_stem(x)

    # 计算差异
    diff = (baseline_out - zero_init_out).abs().max().item()

    print("=== 零初始化验证 ===")
    print(f"预训练基线输出范围: [{baseline_out.min().item():.4f}, {baseline_out.max().item():.4f}]")
    print(f"零初始化Stem输出范围: [{zero_init_out.min().item():.4f}, {zero_init_out.max().item():.4f}]")
    print(f"最大绝对差异: {diff:.8f}")

    if diff < 1e-6:
        print("✓ 验证通过：初始输出与基线完全一致")
    else:
        print(f"✗ 验证失败：存在{diff:.6f}的差异")

    # 检查梯度是否可以传播
    zero_init_stem.train()
    out = zero_init_stem(x)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in zero_init_stem.residual_branch.parameters())

    print(f"\n残差分支梯度传播: {'✓ 正常' if has_grad else '✗ 无梯度'}")


# ============ 与原增强Stem的对比 ============

def compare_with_enhanced_stem():
    """
    对比零初始化残差Stem vs 原增强Stem

    原增强Stem问题：
    1. 全部从零初始化，与预训练主干特征不匹配
    2. 12轮训练不足以稳定收敛
    3. 种子方差>1 AP

    零初始化残差Stem优势：
    1. 起点 = 预训练基线（保底性能）
    2. 残差分支逐渐学习，不会突然退化
    3. 如果12轮学不到有用特征，自然退化为基线
    """
    print("""
=== 零初始化残差Stem vs 原增强Stem ===

【原增强Stem】
结构: 主分支 + 细节 + 高频 + 注意力
初始化: 全部随机
问题:
  - Seed1: AP -1.17pp (中大目标退化)
  - Seed2: AP +1.80pp (全面改善)
  - 种子方差: 2.97pp (信号被噪声淹没)

【零初始化残差Stem】
结构: 预训练主路径 + 零初始化残差分支
初始化: 主路径=预训练，残差=零
优势:
  - 初始性能 = 预训练基线（保底）
  - 残差分支逐渐学习补充信息
  - 最坏情况：残差学不到东西 = 基线性能
  - 最好情况：残差学到有用特征 = 超过基线

预期12轮结果（双seed）:
  - 最坏: ΔAP ≈ 0 (等于基线，不退化)
  - 保守: ΔAP = +0.3 ~ +0.5pp
  - 乐观: ΔAP = +0.8 ~ +1.2pp

判断标准:
  - 如果两seed均非负 + 至少一个>+0.3pp -> 有效且稳定
  - 如果两seed均<+0.2pp -> 12轮学不到东西，放弃
  - 如果出现负值 -> 设计有误，需调试
""")


# ============ 训练建议 ============

def training_recommendations():
    """训练零初始化残差Stem的建议"""
    print("""
=== 训练建议 ===

1. 学习率设置（重要）:
   - 主路径: 1e-5 (极低，几乎冻结)
   - 残差分支: 1e-4 (正常)
   - 融合层: 1e-4 (正常)
   - 残差缩放因子: 1e-3 (略高，快速找到合适幅度)

2. 权重衰减:
   - 主路径: 0 (已预训练，不需要正则)
   - 残差分支: 1e-4 (标准)

3. Warm-up:
   - 前2轮：只训练残差分支，冻结主路径
   - 后10轮：全部训练，主路径低lr

4. 监控指标:
   - residual_scale的值（应逐渐增大，如0.1~0.5）
   - 残差分支的梯度范数（应稳定，不爆炸）
   - AP曲线（应单调上升或稳定，不震荡）

5. 消融对照:
   - 对照A: 完全冻结主路径（残差分支独立学习）
   - 对照B: 不用零初始化（随机初始化融合层）
   - 主实验: 零初始化 + 主路径低lr微调
""")


if __name__ == '__main__':
    print("=" * 70)
    print("零初始化残差Stem实验代码")
    print("=" * 70)

    print("\n1. 验证零初始化正确性\n")
    verify_zero_init()

    print("\n2. 与原增强Stem对比\n")
    compare_with_enhanced_stem()

    print("\n3. 训练建议\n")
    training_recommendations()
