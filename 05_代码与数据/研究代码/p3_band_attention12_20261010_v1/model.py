"""Native DEIM construction, with one optional output-P3 attention adapter."""
from copy import deepcopy
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from band_attention import P3BandAttention


class P3BandAttentionDEIM(DEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ('baseline', 'wavelet', 'spatial'):
            raise ValueError(arm)
        config = deepcopy(recipe)
        # Preserve the construction order and kwargs of the frozen
        # fine_residual_screen24_20261009_v2 baseline geometry.py.
        backbone = HGNetv2(**config['HGNetv2'])
        encoder = HybridEncoder(**config['HybridEncoder'], eval_spatial_size=[640, 640])
        kwargs = dict(config['DFINETransformer'], num_classes=10, eval_spatial_size=[640, 640])
        decoder = DFINETransformer(**kwargs)
        super().__init__(backbone, encoder, decoder, num_classes=10)
        if config['HybridEncoder']['hidden_dim'] != 256 or config['DFINETransformer']['feat_strides'] != [8, 16, 32]:
            raise ValueError('Expected C256 P3/P4/P5 native interface')
        self.attention_arm = arm
        self.band_attention = None if arm == 'baseline' else P3BandAttention(arm)

    def forward(self, x, targets=None):
        features = self.encoder(self.backbone(x))
        if self.band_attention is not None:
            # A fresh list preserves the original P4/P5 tensors unchanged.
            features = [self.band_attention(features[0]), *features[1:]]
        outputs = self.decoder(features, targets)
        if self.training and self.o2m_c is not None:
            outputs['o2m_c'] = self.o2m_c(features)
        return outputs
