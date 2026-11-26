import warnings
# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.logging import MMLogger
from mmengine.model import Sequential, constant_init, kaiming_init
from mmengine.runner.checkpoint import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
import torch

from .resnet import BasicBlock
from .resnet import Bottleneck as _Bottleneck

from mmdet.registry import MODELS
from .resnet import BasicBlock
# from .resnet import Bottleneck
from mmengine.model import BaseModule
import torch.nn.init as init
import segmentation_models_pytorch as smp
from ..layers import ResLayer
from .unetplusplus import Unetplusplus


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        if x.dim() == 1:
            # 예: (C,) -> (1, C, 1, 1)
            return x.view(1, x.shape[0], 1, 1)
        elif x.dim() == 2:
            # 예: (B, C) -> (B, C, 1, 1)
            return x.view(x.shape[0], x.shape[1], 1, 1)
        else:
            return x.view(x.shape[0], x.shape[1], 1, 1)



class Bottleneck(_Bottleneck):
    r"""Bottleneck for the ResNet backbone in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_.

    This bottleneck allows the users to specify whether to use
    SAC (Switchable Atrous Convolution) and RFP (Recursive Feature Pyramid).

    Args:
         inplanes (int): The number of input channels.
         planes (int): The number of output channels before expansion.
         rfp_inplanes (int, optional): The number of channels from RFP.
             Default: None. If specified, an additional conv layer will be
             added for ``rfp_feat``. Otherwise, the structure is the same as
             base class.
         sac (dict, optional): Dictionary to construct SAC. Default: None.
         init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 rfp_inplanes=None,
                 init_cfg=None,
                 **kwargs):
        super(Bottleneck, self).__init__(
            inplanes, planes, init_cfg=init_cfg, **kwargs)

        self.rfp_inplanes = rfp_inplanes
        if self.rfp_inplanes:
            self.rfp_conv = build_conv_layer(
                None,
                self.rfp_inplanes,
                planes * self.expansion,
                1,
                stride=1,
                bias=True)
            if init_cfg is None:
                self.init_cfg = dict(
                    type='Constant', val=0, override=dict(name='rfp_conv'))

    def rfp_forward(self, x, rfp_feat):
        """The forward function that also takes the RFP features as input."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.rfp_inplanes:
            rfp_feat = self.rfp_conv(rfp_feat)
            out = out + rfp_feat

        out = self.relu(out)

        return out



class ResLayer(Sequential):


    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 rfp_inplanes=None,
                 **kwargs):
        self.block = block
        assert downsample_first, f'downsample_first={downsample_first} is ' \
                                 'not supported in DetectoRS'

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                rfp_inplanes=rfp_inplanes,
                **kwargs))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super(ResLayer, self).__init__(*layers)

class Branch3(BaseModule):
    def __init__(self,
                 in_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 block_init_cfg=None,
                 init_cfg=None,
                **kwargs) -> None:

        super(Branch3, self).__init__()
        self.in_channel = in_channel
        self.conv1_1 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel//2, kernel_size=1)

        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg


        self.spatial_attention =   nn.Sequential(
                 build_conv_layer(
                                dict(type='DCN', deform_groups=1),
                                self.in_channel,
                                self.in_channel,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),    
                build_norm_layer(self.norm_cfg, self.in_channel)[1],
                nn.ReLU(),
                build_conv_layer(
                            self.conv_cfg,
                            self.in_channel,
                            self.in_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),  
                nn.Sigmoid())
        self.channel_attention =   nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Linear(self.in_channel, self.in_channel*2),
            # nn.BatchNorm1d(self.in_channel*2),
            nn.ReLU(),
            nn.Linear(self.in_channel*2, self.in_channel),
            nn.Sigmoid(),
            Reshape()
        )

        self._init_weights()

    def _init_weights(self):

        for m in self.channel_attention:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
        for m in self.spatial_attention:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                                
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x,y):
        processed_x = self.conv1_1(x)
        processed_y = self.conv1_1(y)
        processed_x = torch.cat((processed_x, processed_y), dim=1)

        
        return self.spatial_attention(processed_x)*self.channel_attention(processed_x)*x
    
@MODELS.register_module()
class CAHF_ResNet(BaseModule):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    
    def __init__(self,
                depth,
                in_channels=3,
                stem_channels=None,
                base_channels=64,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                style='pytorch',
                deep_stem=False,
                avg_down=False,
                frozen_stages=-1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                dcn=None,
                stage_with_sac=(False, False, False, False),
                sac = None,
                stage_with_dcn=(False, False, False, False),
                plugins=None,
                with_cp=False,
                zero_init_residual=True,
                pretrained=None,
                rfp_inplanes=None,
                output_img=True,

                init_cfg=None):
        self.sac = sac
        self.stage_with_sac = stage_with_sac
        self.rfp_inplanes = rfp_inplanes
        self.output_img = output_img
        super(CAHF_ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                        'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='ConstaBranch3nt',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        # self.attention_layer_name = ["attention_name1","attention_name2","attention_name3", "attention_name4"]
        self._make_stem_layer(in_channels, stem_channels)
        encoder = Unetplusplus().encoder.cuda()

        self.encoder_layer = self.extract_layer(encoder)
    
        self.attention_layers = ['attention_layer1', 'attention_layer2', 'attention_layer3', 'attention_layer4']
        
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            sac = self.sac if self.stage_with_sac[i] else None
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,

                rfp_inplanes=rfp_inplanes if i > 0 else None,                
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion

            layer_name = f'layer{i + 1}'
            attention_name = f'attention_name{i+1}'
            
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)



        # self.attention_layer1 = Branch3(
        #     in_channels=256,
        #     out_channels=256,
        #     kernel_size=3
        # )
        # self.attention_layer2 = Branch3(
        #     in_channels=512,
        #     out_channels=512,
        #     kernel_size=3
        # )
        # self.attention_layer3 = Branch3(
        #     in_channels=1024,
        #     out_channels=1024,
        #     kernel_size=3
        # )
        # self.attention_layer4 = Branch3(
        #     in_channels=2048,
        #     out_channels=2048,
        #     kernel_size=3
        # )
        self.attention_layer1 = Branch3(256,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg,
                )
        self.attention_layer2 = Branch3(512,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg,
                )        
        self.attention_layer3 = Branch3(1024,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg,
                )
        self.attention_layer4 = Branch3(2048,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg,
                )    

    def extract_layer(self, backbone) :
        bakcbone_children = list(backbone.children())
        layer1 = nn.Sequential(*bakcbone_children[:5])
        layer2 = bakcbone_children[5]
        layer3 = bakcbone_children[6]
        layer4 = bakcbone_children[7]

        return [layer1, layer2, layer3, layer4]
    def make_stage_plugins(self, plugins, stage_idx):

        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)



    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
                



    def forward(self, x):


        """Forward function."""
        outs = []

        y = x.clone()
        if self.output_img:
            outs.insert(0, x)
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            y = self.encoder_layer[i](y)
            # del self.encoder_layer[i]
            x = res_layer(x)
            if i > 0 :
                atten_layer = getattr(self, self.attention_layers[i])
                x = atten_layer(x, y)
                
            if i in self.out_indices:
                outs.append(x),
        del y
        return tuple(outs)


    def rfp_forward(self, x, rfp_feats):
        """Forward function for RFP."""
        y = x.clone()

        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            # atten_layer = getattr(self, self.attention_layers[i])
            # self.encoder_layer[i] = self.encoder_layer[i].cuda()

            y = self.encoder_layer[i](y)
            # del self.encoder_layer[i]
            # self.encoder_layer[i] = self.encoder_layer[i].cpu()            
            # del self.encoder_layer[i]
            rfp_feat = rfp_feats[i] if i > 0 else None
            for layer in res_layer:
                x = layer.rfp_forward(x, rfp_feat)
            # x = atten_layer(x, y)
                # if i > 0 :
                #     atten_layer = getattr(self, self.attention_layers[i])
                #     x = atten_layer(x, y)
            
            # atten_layer = getattr(self, self.attention_layers[i])
            # x = atten_layer(x, y)
            if i in self.out_indices:
                outs.append(x)
        del y
        return tuple(outs)


    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(CAHF_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
