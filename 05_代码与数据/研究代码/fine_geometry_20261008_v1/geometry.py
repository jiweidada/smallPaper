"""Isolated 2 x 2 geometry readout experiment; no added parameters or losses.

The encoder forward and decoder entrance follow the hash-bound DEIM/D-FINE
implementation. Original copyright: DEIM and D-FINE Authors, 2024.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from engine.deim.utils import inverse_sigmoid

ARMS = ('baseline', 'fine_only', 'grid_only', 'fine_grid')


class FineGeometryEncoder(HybridEncoder):
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


def selected_geometry_memory(semantic_selected, ids, fine_p3, valid_mask, projection):
    """Only selected P3 tokens receive the shared Linear+LayerNorm readout."""
    fine = fine_p3.flatten(2).transpose(1, 2)
    if fine.shape[0] != ids.shape[0] or fine.shape[-1] != semantic_selected.shape[-1]:
        raise ValueError('Fine and semantic features must have the same batch/channel dimensions')
    bi, qi = torch.where(ids < fine.shape[1])
    if not len(bi):
        return semantic_selected
    local_ids = ids[bi, qi]
    valid = valid_mask.expand(fine.shape[0], -1, -1)[bi, local_ids]
    selected = fine[bi, local_ids] * valid.to(fine.dtype)
    geometry = semantic_selected.clone()
    geometry[bi, qi] = projection(selected)
    return geometry


def grid_reference(raw, anchors_unact, ids, p3_shape, eps=1e-5):
    """P3 grid-unit centers, native width/height and invalid-anchor behavior.

1/W and 1/H support the original multi-scale training. In float32, the inverse
sigmoid round trip can introduce small numerical differences even when raw=0.
"""
    h, w = p3_shape
    if h <= 0 or w <= 0 or raw.shape != anchors_unact.shape:
        raise ValueError('Invalid grid or reference shape')
    native = raw + anchors_unact
    mask = (ids < h * w) & torch.isfinite(anchors_unact).all(-1)
    if not mask.any():
        return native
    anchor_xy = anchors_unact[mask][:, :2].float().sigmoid()
    step = anchor_xy.new_tensor([1. / w, 1. / h])
    center = anchor_xy + raw[mask][:, :2].float() * step
    center = center.clamp(eps, 1 - eps)
    selected = native[mask].clone()
    selected[:, :2] = inverse_sigmoid(center, eps=eps).to(selected.dtype)
    result = native.clone()
    result[mask] = selected
    return result


class GeometryDecoder(DFINETransformer):
    def __init__(self, read_fine=False, grid_center=False, **kwargs):
        super().__init__(**kwargs)
        if self.query_select_method != 'default' or self.mgsa_groups or self.o2m_t:
            raise ValueError('Only native default, fixed single query group is supported')
        if self.feat_strides != [8, 16, 32] or any(not isinstance(m, nn.Identity) for m in self.input_proj):
            raise ValueError('The bound three-level identity decoder projections are required')
        self.read_fine, self.grid_center = read_fine, grid_center
        self._fine_p3 = None
        # Disabled in experiments; engineering tests opt in to detached trace values.
        self.audit_trace = False
        self.last_trace = None

    def forward(self, feats, targets=None):
        self.last_trace = None
        if self.read_fine:
            if not isinstance(feats, dict) or set(feats) != {'features', 'fine_p3'}:
                raise ValueError('Fine readout requires the explicit encoder feature bundle')
            self._fine_p3 = feats['fine_p3']
            feats = feats['features']
        try:
            return super().forward(feats, targets)
        finally:
            self._fine_p3 = None

    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        masked_memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(masked_memory)
        logits = self.enc_score_head(output_memory)
        _, ids = torch.topk(logits.max(-1).values, self.num_queries, dim=-1)
        selected_anchors = anchors.gather(1, ids[..., None].repeat(1, 1, 4))
        selected_logits = logits.gather(1, ids[..., None].repeat(1, 1, logits.shape[-1])) if self.training else None
        semantic = output_memory.gather(1, ids[..., None].repeat(1, 1, output_memory.shape[-1]))
        geometry = semantic
        if self.read_fine:
            if self._fine_p3 is None or list(self._fine_p3.shape[-2:]) != list(spatial_shapes[0]):
                raise ValueError('Fine P3 dimensions do not match native memory')
            geometry = selected_geometry_memory(semantic, ids, self._fine_p3, valid_mask, self.enc_output)
        raw = self.enc_bbox_head(geometry)
        reference = grid_reference(raw, selected_anchors, ids, spatial_shapes[0]) if self.grid_center else raw + selected_anchors
        enc_boxes = [reference.sigmoid()] if self.training else []
        enc_logits = [selected_logits] if self.training else []
        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1]) if self.learn_query_content else semantic.detach()
        if self.audit_trace:
            self.last_trace = {name: value.detach().clone() for name, value in
                               dict(ids=ids, semantic=semantic, geometry=geometry,
                                    selected_anchors=selected_anchors, reference=reference,
                                    content=content, all_logits=logits).items()}
        reference = reference.detach()
        if denoising_bbox_unact is not None:
            reference = torch.concat([denoising_bbox_unact, reference], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        return content, reference, enc_boxes, enc_logits, logits


class GeometryDEIM(DEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ARMS:
            raise ValueError(arm)
        config = deepcopy(recipe)
        read_fine = arm in ('fine_only', 'fine_grid')
        grid_center = arm in ('grid_only', 'fine_grid')
        backbone = HGNetv2(**config['HGNetv2'])
        encoder_type = FineGeometryEncoder if read_fine else HybridEncoder
        encoder = encoder_type(**config['HybridEncoder'], eval_spatial_size=[640, 640])
        kwargs = dict(config['DFINETransformer'], num_classes=10, eval_spatial_size=[640, 640])
        decoder = DFINETransformer(**kwargs) if arm == 'baseline' else GeometryDecoder(
            read_fine=read_fine, grid_center=grid_center, **kwargs)
        super().__init__(backbone, encoder, decoder, num_classes=10)
        self.geometry_arm = arm
