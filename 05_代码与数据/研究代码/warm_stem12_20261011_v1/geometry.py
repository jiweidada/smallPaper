"""Isolated local geometry residual experiment. Native encoder path retained.
Original encoder/decoder interface copyright: DEIM and D-FINE Authors, 2024.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from residual import LocalGeometryResidual
ARMS = ('baseline','fused_residual','fine_residual')

class FeatureBundleEncoder(HybridEncoder):
    def forward(self, feats):
        if self.use_encoder_idx != [2] or len(feats) != 3:
            raise ValueError('Only the bound three-level, P5-transformer encoder is supported')
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        fine_p3 = proj_feats[0]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh, feat_low = inner_outs[0], proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low, feat_height = outs[-1], inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            outs.append(self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1)))
        return {'features': outs, 'fine_p3': fine_p3}

class ResidualDecoder(DFINETransformer):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        if source not in ('fine', 'fused'):
            raise ValueError(source)
        if self.query_select_method != 'default' or self.mgsa_groups or self.o2m_t:
            raise ValueError('Only the native default single query group is supported')
        if self.feat_strides != [8,16,32] or any(not isinstance(m, nn.Identity) for m in self.input_proj):
            raise ValueError('The bound three-level identity decoder projections are required')
        self.geometry_source = source
        self.geometry_residual = LocalGeometryResidual(self.hidden_dim)
        self._geometry_map = None
        self.audit_trace = False
        self.last_trace = None

    def forward(self, feats, targets=None):
        self.last_trace = None
        if not isinstance(feats, dict) or set(feats) != {'features','fine_p3'}:
            raise ValueError('Expected the explicit encoder feature bundle')
        self._geometry_map = feats['fine_p3'] if self.geometry_source == 'fine' else feats['features'][0]
        try:
            return super().forward(feats['features'], targets)
        finally:
            self._geometry_map = None

    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        output_memory = self.enc_output(valid_mask.to(memory.dtype) * memory)
        logits = self.enc_score_head(output_memory)
        _, ids = torch.topk(logits.max(-1).values, self.num_queries, dim=-1)
        selected_anchors = anchors.gather(1, ids[...,None].repeat(1,1,4))
        selected_logits = logits.gather(1,ids[...,None].repeat(1,1,logits.shape[-1])) if self.training else None
        semantic = output_memory.gather(1,ids[...,None].repeat(1,1,output_memory.shape[-1]))
        if self._geometry_map is None or list(self._geometry_map.shape[-2:]) != list(spatial_shapes[0]):
            raise ValueError('Geometry feature grid differs from the native P3 grid')
        native_raw = self.enc_bbox_head(semantic)
        delta = self.geometry_residual(semantic, self._geometry_map, ids, valid_mask)
        raw = native_raw + delta.to(native_raw.dtype)
        reference = raw + selected_anchors
        enc_boxes = [reference.sigmoid()] if self.training else []
        enc_logits = [selected_logits] if self.training else []
        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0],1,1]) if self.learn_query_content else semantic.detach()
        if self.audit_trace:
            self.last_trace = {name:value.detach().clone() for name,value in dict(
                ids=ids,semantic=semantic,native_raw=native_raw,delta=delta,
                reference=reference,all_logits=logits,content=content).items()}
        reference = reference.detach()
        if denoising_bbox_unact is not None:
            reference = torch.concat([denoising_bbox_unact,reference],dim=1)
            content = torch.concat([denoising_logits,content],dim=1)
        return content,reference,enc_boxes,enc_logits,logits


class ResidualGeometryDEIM(DEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ARMS:
            raise ValueError(arm)
        config = deepcopy(recipe)
        backbone = HGNetv2(**config['HGNetv2'])
        encoder_type = HybridEncoder if arm == 'baseline' else FeatureBundleEncoder
        encoder = encoder_type(**config['HybridEncoder'], eval_spatial_size=[640,640])
        kwargs = dict(config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640])
        decoder = DFINETransformer(**kwargs) if arm == 'baseline' else ResidualDecoder(
            source='fine' if arm == 'fine_residual' else 'fused', **kwargs)
        super().__init__(backbone,encoder,decoder,num_classes=10)
        self.geometry_arm = arm
