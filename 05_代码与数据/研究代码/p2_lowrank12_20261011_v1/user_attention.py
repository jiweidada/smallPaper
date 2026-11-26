import torch
from torch import nn
from torch.nn import functional as F

class MSCA_Dilated(nn.Module):
    """
    MSCA using dilated convolutions instead of different kernel sizes
    More efficient and avoids spatial dimension issues
    """
    def __init__(self, channels, dilations=[1, 2, 4], reduction=8, e_lambda=1e-4):
        super(MSCA_Dilated, self).__init__()
        self.channels = channels
        self.dilations = dilations
        self.e_lambda = e_lambda

        # Multi-scale dilated convolutions
        self.multi_scale_convs = nn.ModuleList()
        for dilation in dilations:
            self.multi_scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels//reduction, kernel_size=1),
                    nn.BatchNorm2d(channels//reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels//reduction, channels, kernel_size=3,
                             stride=1, padding=dilation, dilation=dilation,
                             groups=channels//reduction),
                    nn.BatchNorm2d(channels),
                )
            )

        # Channel attention branch
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Context aggregation
        self.context_conv = nn.Conv2d(channels * len(dilations), channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Multi-scale feature extraction using dilated convolutions
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))

        # Concatenate multi-scale features (same spatial dimensions guaranteed)
        ms_concat = torch.cat(multi_scale_features, dim=1)

        # Context aggregation
        context = self.context_conv(ms_concat)

        # Enhanced spatial attention
        mu = context.mean(dim=[2, 3], keepdim=True)
        variance = ((context - mu).pow(2)).mean(dim=[2, 3], keepdim=True)
        spatial_att = (context - mu).pow(2) / (4 * (variance + self.e_lambda)) + 0.5
        spatial_att = self.sigmoid(spatial_att)

        # Channel attention
        channel_att = self.channel_att(x)

        # Combine attentions
        attention = spatial_att * channel_att

        return x * attention

class MSCA_Lite(nn.Module):
    """
    Lightweight version of MSCA with fewer parameters
    Better for real-time drone applications
    """
    def __init__(self, channels, e_lambda=1e-4):
        super(MSCA_Lite, self).__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Multi-scale context using pooling (parameter-free)
        # Scale 1: Original
        scale1 = x

        # Scale 2: 2x2 pooling and upsample
        scale2 = F.avg_pool2d(x, kernel_size=2, stride=2)
        scale2 = F.interpolate(scale2, size=(h, w), mode='bilinear', align_corners=False)

        # Scale 3: 4x4 pooling and upsample
        scale3 = F.avg_pool2d(x, kernel_size=4, stride=4)
        scale3 = F.interpolate(scale3, size=(h, w), mode='bilinear', align_corners=False)

        # Aggregate multi-scale information
        ms_fusion = (scale1 + scale2 + scale3) / 3.0

        # Enhanced SimAM-style attention with multi-scale context
        mu = ms_fusion.mean(dim=[2, 3], keepdim=True)

        # Calculate attention weights
        diff_squared = (ms_fusion - mu).pow(2)
        sigma_squared = diff_squared.mean(dim=[2, 3], keepdim=True)

        # Attention calculation
        attention = diff_squared / (4 * (sigma_squared + self.e_lambda)) + 0.5
        attention = self.sigmoid(attention)

        return x * attention



class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()   
   
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):     
        s = self.__class__.__name__ + '('  
        s += ('lambda=%f)' % self.e_lambda)
        return s
  
    @staticmethod
    def get_module_name(): 
        return "simam"     

    def forward(self, x):  
        b, c, h, w = x.size()

        n = w * h - 1 

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)    
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5    
    
        return x * self.activaton(y)



def get_activation(act_type):
    """
    根据激活函数类型返回对应的激活函数

    Args:
        act_type (str): 激活函数类型，支持 'relu', 'silu', 'gelu', 'swish' 等

    Returns:
        nn.Module: 对应的激活函数模块
    """
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'silu' or act_type == 'swish':
        return nn.SiLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif act_type == 'hardswish':
        return nn.Hardswish()
    else:
        # 默认使用 ReLU
        return nn.ReLU()

class LearnableAffineBlock(nn.Module):
    """
    可学习的仿射变换模块 (Learnable Affine Block)

    该模块对输入 `x` 进行仿射变换：
        y = scale * x + bias
    其中 `scale` 和 `bias` 是可训练参数。

    适用于需要简单线性变换的场景，例如：
    - 归一化调整
    - 特征平移缩放
    - 作为更复杂模型的一部分
    """

    def __init__(
            self,
            scale_value=1.0,  # 初始化缩放因子，默认为 1.0（保持输入不变）
            bias_value=0.0  # 初始化偏移量，默认为 0.0（无偏移）
    ):
        super().__init__()
        # 定义可学习参数：缩放因子和偏移量
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        """
        前向传播：执行仿射变换

        参数:
        x (Tensor) - 输入张量

        返回:
        Tensor - 变换后的输出张量
        """
        return self.scale * x + self.bias

class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            act_type='relu'  # 新增激活函数类型参数
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                # nn.ZeroPad2d([0, 1, 0, 1]) 手动填充 右侧 1 个像素 和 底部 1 个像素，而左侧和顶部不填充。
                # 这种方式适用于 kernel_size=2 的情况，使得卷积输出的尺寸与输入相同（在 stride=1 时）。
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,  # 表示 PyTorch 默认的 SAME 填充，即对 左右、上下 进行均匀填充。
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = get_activation(act_type)  # 使用可配置的激活函数
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x

class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            act_type='relu'  # 新增激活函数类型参数
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            act_type=act_type,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            act_type=act_type,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)

class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,  # 输入通道数
            mid_chs,  # 中间通道数（隐含层）
            out_chs,  # 输出通道数
            layer_num,  # 卷积层的数量
            kernel_size=3,  # 卷积核大小
            residual=False,  # 是否使用残差连接
            light_block=False,  # 是否使用轻量级卷积块（深度可分离卷积）
            use_lab=False,  # 是否使用可学习的仿射变换模块
            agg='ese',  # 选择特征聚合方式（'se' 或 'ese' 或 'ema' 或 'simam' 或 'lsk' 或 'deformable_lka' 或 'msca' 或 'msca_dilated' 或 'msca_lite' 或 'msca_enhanced'）
            drop_path=0.,  # DropPath 比例（用于正则化）
            act_type='relu',  # 新增激活函数类型参数
    ):
        super().__init__()
        self.residual = residual

        # 构建卷积层序列
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                # 轻量级卷积块（深度可分离卷积）
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        act_type=act_type,
                    )
                )
            else:
                # 标准卷积块
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        act_type=act_type,
                    )
                )

        # 计算聚合层的输入通道数
        total_chs = in_chs + layer_num * mid_chs

        # 特征聚合方式
        if agg == 'se':  # SE注意力机制
            aggregation_squeeze_conv = ConvBNAct(
                total_chs, out_chs // 2, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        elif agg == 'ese':  # 默认ESE注意力机制
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = EseModule(out_chs)  # ESE注意力模块
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'ema':  # ema注意力
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = EMA(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'simam':  # simam注意力
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = SimAM()
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'lsk':  # LSKBlock注意力
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = LSKBlock(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'deformable_lka':  # DeformableLKA注意力
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = DeformableLKA(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'msca':  # MSCA完整版本注意力
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = MSCA(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'msca_dilated':  # MSCA膨胀卷积版本注意力（推荐用于精度）
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = MSCA_Dilated(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'msca_lite':  # MSCA轻量级版本注意力（推荐用于实时应用）
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = MSCA_Lite(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        elif agg == 'msca_enhanced':  # MSCA增强版本注意力（最高精度但更多计算量）
            aggregation_conv = ConvBNAct(
                total_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act_type=act_type
            )
            att = MSCA_Enhanced(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        else:
            raise Exception(f"param agg '{agg}' Illegal.")

        # DropPath 处理（防止过拟合）
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x  # 保留输入用于残差连接
        output = [x]  # 存储各层输出
        for layer in self.layers:
            x = layer(x)
            output.append(x)  # 逐层添加到列表

        # x1 = layers_1(x)
        # x2 = layers_2(x1)
        # x3 = layers_3(x2)
        # output = [x, x1, x2, x3]

        # 进行通道拼接
        x = torch.cat(output, dim=1)

        # 进行聚合处理
        x = self.aggregation(x)

        # 若使用残差连接，则进行残差加和
        if self.residual:
            x = self.drop_path(x) + identity

        return x
