"""Isolated native DEIM candidate-only P2 experiment. No decoder-memory change."""
import types
import torch
from torch import nn
from torch.nn import functional as F
from engine.core import register
from engine.deim.deim import DEIM


class P2Recruitment(nn.Module):
    def __init__(self, c2_channels, hidden_dim, p2_budget, anchor_wh_px):
        super().__init__()
        # A zero module must not advance the RNG used by the paired BASE arm.
        with torch.random.fork_rng(devices=[]):
            self.project = nn.Conv2d(c2_channels, hidden_dim, 1, bias=False)
            nn.init.zeros_(self.project.weight)
        self.p2_budget = int(p2_budget)
        self.anchor_wh_px = float(anchor_wh_px)

    def p2_anchors(self, shape, eps, device):
        h, w = shape
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device), indexing='ij')
        centers = (torch.stack((x, y), -1).float() + .5) / torch.tensor([w, h], device=device)
        # Nominal pixels at the 640 reference resolution, matching native priors'
        # normalized-size behavior during multiscale augmentation.
        wh = torch.full_like(centers, self.anchor_wh_px / 640.)
        anchors = torch.cat((centers, wh), -1).reshape(1, h*w, 4)
        valid = ((anchors > eps) & (anchors < 1-eps)).all(-1, keepdim=True)
        unact = torch.log(anchors / (1-anchors))
        return torch.where(valid, unact, torch.inf), valid


def candidate_decoder_input(self, memory, spatial_shapes, denoising_logits=None,
                            denoising_bbox_unact=None):
    c2, p3 = self._p2_context
    branch = self.query_recruitment
    fine = branch.project(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
    p2_memory = fine.flatten(2).transpose(1, 2)
    p2_anchors, p2_valid = branch.p2_anchors(c2.shape[-2:], self.eps, memory.device)
    if self.training or self.eval_spatial_size is None:
        old_anchors, old_valid = self._generate_anchors(spatial_shapes, device=memory.device)
    else:
        old_anchors, old_valid = self.anchors, self.valid_mask
    # Native three-level scores and invalid-anchor handling remain unchanged.
    old_output = self.enc_output(old_valid.to(memory.dtype) * memory)
    old_logits = self.enc_score_head(old_output)
    p2_output = self.enc_output(p2_valid.to(p2_memory.dtype) * p2_memory)
    p2_logits = self.enc_score_head(p2_output)
    p2_scores = p2_logits.max(-1).values.masked_fill(~p2_valid.squeeze(-1), -torch.inf)
    old_k = self.num_queries - branch.p2_budget
    old_idx = old_logits.max(-1).values.topk(old_k, dim=-1).indices
    p2_idx = p2_scores.topk(branch.p2_budget, dim=-1).indices
    # Keep original queries first; quota is fixed, not tuned against validation.
    def gather(v, idx):
        return v.expand(memory.shape[0], -1, -1).gather(1, idx[..., None].expand(-1, -1, v.shape[-1]))
    selected_memory = torch.cat((gather(old_output, old_idx), gather(p2_output, p2_idx)), 1)
    selected_anchors = torch.cat((gather(old_anchors, old_idx), gather(p2_anchors, p2_idx)), 1)
    selected_logits = torch.cat((gather(old_logits, old_idx), gather(p2_logits, p2_idx)), 1)
    selected_boxes = self.enc_bbox_head(selected_memory) + selected_anchors
    enc_boxes = [selected_boxes.sigmoid()] if self.training else []
    enc_logits = [selected_logits] if self.training else []
    content = (self.tgt_embed.weight[None].expand(memory.shape[0], -1, -1)
               if self.learn_query_content else selected_memory.detach())
    references = selected_boxes.detach()
    if denoising_bbox_unact is not None:
        references = torch.cat((denoising_bbox_unact, references), 1)
        content = torch.cat((denoising_logits, content), 1)
    self._p2_last_info = {'old_count': old_k, 'p2_count': branch.p2_budget,
                          'memory_tokens': memory.shape[1],
                          'candidate_tokens': memory.shape[1]+p2_memory.shape[1],
                          'spatial_shapes': [list(s) for s in spatial_shapes],
                          'selected_boxes_require_grad': selected_boxes.requires_grad,
                          'content_requires_grad': content.requires_grad,
                          'references_require_grad': references.requires_grad}
    return content, references, enc_boxes, enc_logits, torch.cat((old_logits, p2_logits), 1)


@register()
class P2CandidateDEIM(DEIM):
    __inject__ = ['backbone', 'encoder', 'decoder']
    __share__ = ['num_classes']

    def __init__(self, backbone, encoder, decoder, o2m_c=False, num_classes=80,
                 c2_channels=64, p2_budget=60, anchor_wh_px=8.2):
        super().__init__(backbone, encoder, decoder, o2m_c, num_classes)
        if (decoder.num_levels != 3 or decoder.num_queries != 300 or
                decoder.query_select_method != 'default' or decoder.mgsa_groups or
                decoder.o2m_t or o2m_c or not 0 < p2_budget < decoder.num_queries):
            raise ValueError('P2 experiment requires the native 3-level/300-query default DEIM-S configuration')
        decoder.query_recruitment = P2Recruitment(c2_channels, decoder.hidden_dim, p2_budget, anchor_wh_px)
        decoder._p2_context = None
        decoder._get_decoder_input = types.MethodType(candidate_decoder_input, decoder)

    def forward(self, x, targets=None):
        captured = []
        hook = self.backbone.stages[0].register_forward_hook(lambda _m, _i, out: captured.append(out))
        try:
            backbone_features = self.backbone(x)
        finally:
            hook.remove()
        features = self.encoder(backbone_features)
        if len(captured) != 1:
            raise RuntimeError('C2 capture must execute exactly once')
        self.decoder._p2_context = (captured[0], features[0])
        try:
            return self.decoder(features, targets)
        finally:
            self.decoder._p2_context = None
