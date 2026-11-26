"""Unmodified native Stem definitions; no pretrained Stem state is loaded."""
import torch
from torch import nn
from torch.nn import functional as F

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
            bias_value=0.0    # 初始化偏移量，默认为 0.0（无偏移）   
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
            use_lab=False
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
                padding=(kernel_size - 1) // 2, # 表示 PyTorch 默认的 SAME 填充，即对 左右、上下 进行均匀填充。   
                groups=groups,
                bias=False
            ) 
        self.bn = nn.BatchNorm2d(out_chs)    
        if self.use_act:    
            self.act = nn.ReLU()
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

class StemBlock(nn.Module):
    """  
    HGNetv2 的 Stem Block，作为网络的输入处理模块。
    
    主要作用：  
    1. 进行初步特征提取（stem1）。  
    2. 通过两条不同路径（stem2a 和 stem2b）处理特征，并进行特征融合。
    3. 采用池化（pool）和卷积（stem3、stem4）进一步提取特征。
    4. 适用于深度神经网络的前几层，以减少计算量并提高模型稳定性。
  
    Args:
        in_chs (int): 输入通道数。    
        mid_chs (int): 中间层通道数。
        out_chs (int): 输出通道数。 
        use_lab (bool): 是否使用 Lab 颜色空间的特殊处理（默认为 False)。
    """
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):  
        super().__init__()    
        # 初始卷积层，进行基本特征提取，步长为2，实现下采样
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,   
            stride=2,
            use_lab=use_lab,   
        )
        
        # 分支 1：stem2a 和 stem2b 组成一个小的子路径   
        # stem2a：将 mid_chs 降维到 mid_chs//2
        self.stem2a = ConvBNAct(
            mid_chs,    
            mid_chs // 2,   
            kernel_size=2,
            stride=1,   
            use_lab=use_lab,
        ) 
        # stem2b：再将 mid_chs//2 提升回 mid_chs  
        self.stem2b = ConvBNAct(    
            mid_chs // 2,     
            mid_chs,
            kernel_size=2,
            stride=1, 
            use_lab=use_lab,   
        )  
   
        # 分支 2：最大池化层，对 stem1 输出进行下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
   
        # 合并两个分支后的卷积层     
        # stem3：对拼接后的特征进行卷积，步长为 2 进行下采样   
        self.stem3 = ConvBNAct(
            mid_chs * 2,  # 两个分支的特征图通道数加起来
            mid_chs,     
            kernel_size=3,  
            stride=2,
            use_lab=use_lab,    
        ) 
 
        # stem4：最后一个 1x1 卷积层，调整通道数为 out_chs 
        self.stem4 = ConvBNAct(
            mid_chs,    
            out_chs,
            kernel_size=1, 
            stride=1,
            use_lab=use_lab,    
        )     

    def forward(self, x):   
        """ 
        前向传播过程： 
        1. 通过 stem1 进行特征提取。
        2. 通过 stem2a 和 stem2b 进行一个额外的特征变换路径（右分支）。
        3. 通过最大池化（左分支）。
        4. 拼接两个分支的特征（cat）。     
        5. 通过 stem3 和 stem4 进行进一步特征提取。
     
        Args:    
            x (torch.Tensor): 输入特征图 (batch_size, in_chs, H, W)   
 
        Returns:  
            torch.Tensor: 处理后的特征图 (batch_size, out_chs, H // 4, W // 4)
        """   
        # Step 1: 初步特征提取
        x = self.stem1(x) 
        
        # Step 2: 右分支，使用额外的 2x2 卷积进行特征处理
        x = F.pad(x, (0, 1, 0, 1))  # 右侧和底部填充 1 个像素 
        x2 = self.stem2a(x)   
        x2 = F.pad(x2, (0, 1, 0, 1))  # 右侧和底部再次填充
        x2 = self.stem2b(x2)  # 经过第二个小卷积
    
        # Step 3: 左分支，使用最大池化进行特征下采样
        x1 = self.pool(x)     

        # Step 4: 组合左右分支的特征    
        x = torch.cat([x1, x2], dim=1)  # 在通道维度（dim=1）拼接  
        
        # Step 5: 进一步处理
        x = self.stem3(x)  # 进行 3x3 卷积
        x = self.stem4(x)  # 进行 1x1 卷积 
        
        return x
