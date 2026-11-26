"""Shared backbone/encoder and detector, native second-layer callback only."""
import torch
from dual_view import DualViewDEIM
from range_context import RangeContext, VARIANTS


class DualRangeDEIM(DualViewDEIM):
    def __init__(self, backbone, encoder, decoder, variant, adaptive_multiplier=1.):
        super().__init__(backbone, encoder, decoder, mode='independent', width=64)
        self.range_interaction = RangeContext(variant, adaptive_multiplier=adaptive_multiplier)

    def callback(self, global_encoded, local_encoded, crop, projected_context=None):
        _, context_source = VARIANTS[self.range_interaction.variant]
        if context_source == 'disabled':
            return None
        module = self.range_interaction
        if context_source == 'local':
            context = module.prepare_context(local_encoded[1])
        else:
            context = projected_context if projected_context is not None else module.prepare_context(global_encoded[1])
        local = module.prepare_local(local_encoded[0])
        hw = global_encoded[1].shape[-2:]
        return lambda q, p, ref, support, scale: module(q, p, ref, support, scale, crop, context, local, hw)

    def forward(self, samples, targets=None):
        full, local, crop = samples['global'], samples['local'], samples['crop_norm']
        if full.shape != local.shape or full.ndim != 4 or crop.shape != (full.shape[0], 4):
            raise ValueError('Expected paired [B,3,640,640] and [B,4] crop coordinates')
        batch = full.shape[0]
        encoded = self.encoder(self.backbone(torch.cat((full, local), 0)))
        global_encoded, local_encoded = [x[:batch] for x in encoded], [x[batch:] for x in encoded]
        return {
            'global': self.decoder(global_encoded, targets['global'] if targets else None),
            'local': self.decoder(local_encoded, targets['local'] if targets else None,
                                  distribution_update=self.callback(global_encoded, local_encoded, crop)),
        }

    def cache_global(self, image):
        if self.training:
            raise RuntimeError('Caching is evaluation-only')
        encoded = self.encoder(self.backbone(image))
        source = VARIANTS[self.range_interaction.variant][1]
        context = self.range_interaction.prepare_context(encoded[1]) if source == 'global' else None
        return {'encoded': encoded, 'context': context, 'variant': self.range_interaction.variant}, self.decoder(encoded)

    def predict_local(self, image, crop_norm, cache):
        if self.training:
            raise RuntimeError('Use paired forward for training')
        if cache['variant'] != self.range_interaction.variant:
            raise ValueError('Context cache belongs to a different variant')
        encoded = self.encoder(self.backbone(image))
        return self.decoder(encoded, distribution_update=self.callback(cache['encoded'], encoded, crop_norm, cache['context']))
