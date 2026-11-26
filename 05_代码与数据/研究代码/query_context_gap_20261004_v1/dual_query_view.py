"""Isolated two-view query interaction; shared backbone/encoder/native detector."""
import torch
from dual_view import DualViewDEIM
from query_context import QueryContext


class DualQueryDEIM(DualViewDEIM):
    def __init__(self, backbone, encoder, decoder, mode, context_config):
        if mode not in ('independent', 'global', 'gap', 'local', 'support', 'gap_unit', 'aligned'):
            raise ValueError(mode)
        super().__init__(backbone, encoder, decoder,
                         mode='aligned' if mode == 'aligned' else 'independent', width=64)
        self.context_mode = mode
        # All arms consume identical construction RNG and carry identical state
        # keys; dormant tensors are excluded from active-parameter counts.
        self.context_interaction = QueryContext(**context_config)
        if mode in ('independent', 'aligned'):
            self.context_interaction.requires_grad_(False)

    def context_update(self, global_encoded, local_encoded, crop):
        if self.context_mode in ('independent', 'aligned'):
            return None
        block = self.context_interaction
        memory = block.prepare_memory(local_encoded[1], crop) if self.context_mode == 'local' else block.prepare_memory(global_encoded[1])
        return lambda query, reference: block(query, reference, crop, memory, self.context_mode)

    def forward(self, samples, targets=None):
        full, local, crop = samples['global'], samples['local'], samples['crop_norm']
        if full.shape != local.shape or full.ndim != 4 or full.shape[1:] != (3, 640, 640) or crop.shape != (full.shape[0], 4):
            raise ValueError('Requires native 640 crops and global resize640; batch is original-image pairs')
        b = full.shape[0]
        features = self.backbone(torch.cat((full, local), 0))
        global_features = [x[:b] for x in features]
        local_features = self.interact([x[b:] for x in features], global_features, crop)
        encoded = self.encoder([torch.cat((g, l), 0) for g, l in zip(global_features, local_features)])
        global_encoded, local_encoded = [x[:b] for x in encoded], [x[b:] for x in encoded]
        global_output = self.decoder(global_encoded, targets['global'] if targets else None)
        local_output = self.decoder(local_encoded, targets['local'] if targets else None,
                                    query_context_update=self.context_update(global_encoded, local_encoded, crop))
        return {'global': global_output, 'local': local_output}

    def cache_global(self, image):
        if self.training:
            raise RuntimeError('Caching is evaluation-only')
        features = self.backbone(image)
        encoded = self.encoder(features)
        cache = {'features': features}
        if self.context_mode not in ('independent', 'aligned', 'local'):
            cache['memory'] = self.context_interaction.prepare_memory(encoded[1])
        return cache, self.decoder(encoded)

    def predict_local(self, image, crop_norm, global_features):
        if self.training:
            raise RuntimeError('Use paired forward for training')
        if image.shape[1:] != (3, 640, 640):
            raise ValueError('Only native 640 crops are supported')
        features = self.interact(self.backbone(image), global_features['features'], crop_norm)
        encoded = self.encoder(features)
        update = None
        if self.context_mode not in ('independent', 'aligned'):
            block = self.context_interaction
            memory = block.prepare_memory(encoded[1], crop_norm) if self.context_mode == 'local' else global_features['memory']
            update = lambda q, r: block(q, r, crop_norm, memory, self.context_mode)
        return self.decoder(encoded, query_context_update=update)
