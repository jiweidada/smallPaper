"""Keep the native detector; supplement its loaded native Stem instead of replacing it."""
from geometry import ResidualGeometryDEIM
from warm_stem import WarmStartEnhancedStem
import torch

class WarmStartEnhancedDEIM(ResidualGeometryDEIM):
    __inject__=[]
    __share__=[]
    def __init__(self,recipe,arm='baseline'):
        if arm!='baseline':
            raise ValueError('Only the bound native encoder/decoder path is supported')
        super().__init__(recipe,arm=arm)
        cfg=recipe['HGNetv2']
        if cfg['name']!='B0' or cfg.get('freeze_at',-1)>=0 or cfg.get('freeze_norm') is not False:
            raise ValueError('Unexpected backbone configuration')
        channels=self.backbone.arch_configs['B0']['stem_channels']
        if channels!=[3,16,16]:
            raise ValueError('Unexpected native Stem channels')
        self.old_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
        self.backbone.stem=WarmStartEnhancedStem(self.backbone.stem,channels,cfg['use_lab'],torch.initial_seed())
        self.new_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())

