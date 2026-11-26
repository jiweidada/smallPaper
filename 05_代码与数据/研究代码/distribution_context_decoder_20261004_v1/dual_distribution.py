"""Shared detector, asymmetric update of local queries only."""
import torch
from dual_view import DualViewDEIM
from distribution_context import DistributionContext


class DualDistributionDEIM(DualViewDEIM):
    def __init__(self, backbone, encoder, decoder, arm):
        super().__init__(backbone, encoder, decoder, 'independent', 64)
        self.distribution_interaction = DistributionContext(arm)

    def callback(self, global_encoded, local_encoded, crop, projected_context=None):
        module = self.distribution_interaction
        if module.arm == 'A':
            return None
        context = (module.prepare_context(local_encoded[1]) if module.arm == 'C' else
                   projected_context if projected_context is not None else module.prepare_context(global_encoded[1]))
        local = module.prepare_local(local_encoded[0])
        return lambda q, p, ref, support, scale: module(q, p, ref, support, scale, crop, context, local)

    def forward(self, samples, targets=None):
        full, local, crop = samples['global'], samples['local'], samples['crop_norm']
        if full.shape != local.shape or crop.shape != (len(full), 4):
            raise ValueError('Expected paired global/crop batch')
        b = len(full)
        encoded = self.encoder(self.backbone(torch.cat((full, local), 0)))
        g, l = [x[:b] for x in encoded], [x[b:] for x in encoded]
        return {'global': self.decoder(g, targets['global'] if targets else None),
                'local': self.decoder(l, targets['local'] if targets else None,
                                      distribution_update=self.callback(g, l, crop))}

    def cache_global(self, image):
        if self.training:
            raise RuntimeError('Caching is evaluation-only')
        encoded = self.encoder(self.backbone(image))
        module = self.distribution_interaction
        context = module.prepare_context(encoded[1]) if module.arm not in ('A', 'C') else None
        return {'encoded': encoded, 'context': context}, self.decoder(encoded)

    def predict_local(self, image, crop_norm, global_features):
        if self.training:
            raise RuntimeError('Caching is evaluation-only')
        local = self.encoder(self.backbone(image))
        cache = global_features
        return self.decoder(local, distribution_update=self.callback(cache['encoded'], local, crop_norm, cache['context']))
