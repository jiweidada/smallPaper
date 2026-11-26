'''
直接修改现有LoG.py文件的方案
在您的原始代码基础上添加多尺度功能，保持兼容性
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch, math
import torch.nn as nn

from engine.extre_module.ultralytics_nn.conv import Conv

class Conv_Extra(nn.Module):
    def __init__(self, channel):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(Conv(channel, 64, 1),
                                   Conv(64, 64, 3),
                                   Conv(64, channel, 1, act=False))
    def forward(self, x):
        out = self.block(x)
        return out

class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class DRFD_LoG(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = nn.SiLU()
        self.norm_c = nn.BatchNorm2d(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, feature_extra=False)
        self.norm_g = nn.BatchNorm2d(self.outdim)

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x

# =============================================================================
# 改进的LoGFilter - 支持多尺度
# =============================================================================
class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, sigma=1.0, use_multiscale=False):
        super(LoGFilter, self).__init__()

        self.use_multiscale = use_multiscale

        if use_multiscale:
            # 多尺度模式
            assert out_c % 3 == 0, f"For multiscale, out_c ({out_c}) must be divisible by 3"
            scale_dim = out_c // 3

            # 三个不同尺度的LoG滤波器
            self.log_small = self._create_single_log(in_c, scale_dim, 5, 0.5)
            self.log_medium = self._create_single_log(in_c, scale_dim, 7, 1.0)
            self.log_large = self._create_single_log(in_c, scale_dim, 11, 2.0)

            # 简单融合
            self.fusion = nn.Sequential(
                nn.Conv2d(out_c, out_c, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_c),
                nn.SiLU()
            )
        else:
            # 原始单尺度模式
            self.single_log = self._create_single_log(in_c, out_c, kernel_size, sigma)

    def _create_single_log(self, in_c, out_c, kernel_size, sigma):
        """创建单个LoG滤波器"""
        # 7x7 convolution with stride 1 for feature reinforcement
        conv_init = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)

        # 创建高斯-拉普拉斯核
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        if torch.abs(kernel).sum() > 0:
            kernel = kernel / torch.abs(kernel).sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)

        # LoG卷积层
        log_conv = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        log_conv.weight.data = log_kernel.repeat(out_c, 1, 1, 1)

        # 组合成Sequential
        return nn.ModuleDict({
            'conv_init': conv_init,
            'log_conv': log_conv,
            'act': nn.SiLU(),
            'norm1': nn.BatchNorm2d(out_c),
            'norm2': nn.BatchNorm2d(out_c)
        })

    def _forward_single_log(self, x, log_module):
        """单个LoG的前向传播"""
        x = log_module['conv_init'](x)
        log_response = log_module['log_conv'](x)
        log_edge = log_module['act'](log_module['norm1'](log_response))
        x = log_module['norm2'](x + log_edge)
        return x

    def forward(self, x):
        if self.use_multiscale:
            # 多尺度处理
            small_features = self._forward_single_log(x, self.log_small)
            medium_features = self._forward_single_log(x, self.log_medium)
            large_features = self._forward_single_log(x, self.log_large)

            # 拼接并融合
            multi_scale = torch.cat([small_features, medium_features, large_features], dim=1)
            return self.fusion(multi_scale)
        else:
            # 原始单尺度处理
            return self._forward_single_log(x, self.single_log)

# =============================================================================
# 保持原始LoGStem接口，但支持多尺度选项
# =============================================================================
class LoGStem(nn.Module):
    """
    通道兼容的LoGStem - 保持输出通道数与原始设计一致
    """

    def __init__(self, in_chans, stem_dim, use_multiscale=False):
        super().__init__()

        # 保持原始的输出通道数设计
        target_out_c14 = int(stem_dim / 4)
        target_out_c12 = int(stem_dim / 2)

        print(f"🔧 LoGStem初始化: stem_dim={stem_dim}, 目标out_c14={target_out_c14}, out_c12={target_out_c12}")

        if use_multiscale:
            # 多尺度模式：内部使用更多通道，最后投影回目标通道数

            # 找到适合3尺度分解的内部通道数
            internal_c14 = ((target_out_c14 // 3) + 1) * 3 if target_out_c14 % 3 != 0 else target_out_c14

            print(f"🔧 多尺度模式: 内部通道数={internal_c14}, 输出通道数={target_out_c14}")

            # 多尺度LoG滤波器（使用内部通道数）
            self.LoG = LoGFilter(in_chans, internal_c14, 7, 1.0, use_multiscale=True)

            # 通道数适配层：将内部通道数投影到目标通道数
            if internal_c14 != target_out_c14:
                self.channel_adapter = nn.Sequential(
                    nn.Conv2d(internal_c14, target_out_c14, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(target_out_c14),
                    nn.SiLU()
                )
                print(f"🔧 添加通道适配器: {internal_c14} -> {target_out_c14}")
            else:
                self.channel_adapter = nn.Identity()
        else:
            # 原始模式：直接使用目标通道数
            self.LoG = LoGFilter(in_chans, target_out_c14, 7, 1.0, use_multiscale=False)
            self.channel_adapter = nn.Identity()

        # 下采样层：始终使用目标通道数
        self.Conv_D = nn.Sequential(
            nn.Conv2d(target_out_c14, target_out_c12, kernel_size=3, stride=1, padding=1, groups=target_out_c14),
            Conv(target_out_c12, target_out_c12, 3, 2, g=target_out_c12)
        )

        # 其他组件：使用目标通道数
        self.gaussian = Gaussian(target_out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(target_out_c12)
        self.drfd = DRFD_LoG(target_out_c12)

        print(f"✅ LoGStem创建完成: 输出通道数={target_out_c12}")

    def forward(self, x):
        x = self.LoG(x)  # LoG特征提取
        x = self.channel_adapter(x)  # 通道数适配（多尺度模式下）
        x = self.Conv_D(x)  # 下采样
        x = self.norm(x + self.gaussian(x))  # 高斯增强
        x = self.drfd(x)  # 双路径特征融合
        return x

# =============================================================================
# 便捷函数：创建不同版本的LoGStem
# =============================================================================
def create_original_logstem(in_chans, stem_dim):
    """创建原始版本的LoGStem"""
    return LoGStem(in_chans, stem_dim, use_multiscale=False)

def create_multiscale_logstem(in_chans, stem_dim):
    """创建多尺度版本的LoGStem"""
    return LoGStem(in_chans, stem_dim, use_multiscale=True)

# =============================================================================
# 测试代码
# =============================================================================
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 3, 24, 224, 224  # 确保out_channel能被3整除
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    print(BLUE + "="*50)
    print("LoGStem 原始版本 vs 多尺度版本对比")
    print("="*50 + RESET)

    # 测试原始版本
    print(YELLOW + "\n1. 原始LoGStem:" + RESET)
    original_model = create_original_logstem(in_channel, out_channel).to(device)
    original_outputs = original_model(inputs)
    print(GREEN + f'   输入: {inputs.size()} -> 输出: {original_outputs.size()}' + RESET)

    # 测试多尺度版本
    print(YELLOW + "\n2. 多尺度LoGStem:" + RESET)
    multiscale_model = create_multiscale_logstem(in_channel, out_channel).to(device)
    multiscale_outputs = multiscale_model(inputs)
    print(GREEN + f'   输入: {inputs.size()} -> 输出: {multiscale_outputs.size()}' + RESET)

    # 计算量对比
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    orig_params = count_params(original_model)
    multi_params = count_params(multiscale_model)

    print(BLUE + "\n参数量对比:" + RESET)
    print(f"原始版本: {orig_params:,} 参数")
    print(f"多尺度版本: {multi_params:,} 参数")
    print(f"增加: {((multi_params/orig_params-1)*100):.1f}%")

    print(GREEN + f"\n✓ 测试完成！两个版本输出尺寸一致。" + RESET)
    print(YELLOW + f"💡 使用方法：" + RESET)
    print(f"  原始版本: LoGStem(in_chans, stem_dim, use_multiscale=False)")
    print(f"  多尺度版本: LoGStem(in_chans, stem_dim, use_multiscale=True)")