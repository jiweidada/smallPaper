"""Independent DEIM adapter; native path adapted from DEIM/D-FINE Authors 2024.

No edits to the frozen native source. Only the single-group default selector is
supported. BASE instantiates the unmodified native decoder.
"""
from copy import deepcopy
import torch
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from readout import P3LocalReadout


class P3ReadoutDecoder(DFINETransformer):
    def __init__(self, arm, **kwargs):
        super().__init__(**kwargs)
        if self.query_select_method != 'default' or self.mgsa_groups or self.o2m_t:
            raise ValueError('Only native default single-group TopK is supported')
        if self.hidden_dim != 256 or self.feat_strides != [8, 16, 32]:
            raise ValueError('Expected 256-channel P3/P4/P5')
        self.local_readout = P3LocalReadout(arm)

    def _get_decoder_input(self, memory, spatial_shapes,
                           denoising_logits=None, denoising_bbox_unact=None):
        # memory is already input-projected; take raw P3 BEFORE valid-mask and
        # enc_output, without intercepting or recomputing the encoder forward.
        h, w = spatial_shapes[0]
        p3 = memory[:, :h*w].transpose(1, 2).reshape(memory.shape[0], 256, h, w)
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        scores = self.enc_score_head(output_memory)
        semantic, selected_scores, selected_anchors = self._select_topk(
            output_memory, scores, anchors, self.num_queries)
        z0 = self.enc_bbox_head(semantic) + selected_anchors
        z1 = z0 + self.local_readout(semantic, p3, z0)
        enc_boxes = [z1.sigmoid()] if self.training else []
        enc_scores = [selected_scores] if self.training else []
        content = (self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
                   if self.learn_query_content else semantic.detach())
        reference = z1.detach()
        if denoising_bbox_unact is not None:
            reference = torch.concat([denoising_bbox_unact, reference], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        return content, reference, enc_boxes, enc_scores, scores


class P3ReadoutDEIM(DEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ('baseline', 'center', 'spatial'):
            raise ValueError(arm)
        config = deepcopy(recipe)
        backbone = HGNetv2(**config['HGNetv2'])
        encoder = HybridEncoder(**config['HybridEncoder'], eval_spatial_size=[640, 640])
        kwargs = dict(config['DFINETransformer'], num_classes=10, eval_spatial_size=[640, 640])
        decoder = (DFINETransformer(**kwargs) if arm == 'baseline'
                   else P3ReadoutDecoder(arm, **kwargs))
        super().__init__(backbone, encoder, decoder, num_classes=10)
        self.readout_arm = arm
