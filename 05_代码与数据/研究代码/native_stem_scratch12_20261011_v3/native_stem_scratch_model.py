"""Native Stem from scratch; no outside-Stem tensor or data RNG change."""
import torch
from geometry import ResidualGeometryDEIM
from native_stem_scratch import StemBlock

class NativeStemScratchDEIM(ResidualGeometryDEIM):
    __inject__=[]
    __share__=[]
    def __init__(self,recipe,arm='baseline'):
        assert arm=='baseline'
        super().__init__(recipe,arm=arm)
        cfg=recipe['HGNetv2']
        assert cfg['name']=='B0' and cfg.get('freeze_at',-1)<0 and cfg.get('freeze_norm') is False
        channels=self.backbone.arch_configs['B0']['stem_channels']
        assert channels==[3,16,16]
        self.old_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.backbone.stem=StemBlock(*channels,use_lab=cfg['use_lab'])
        self.new_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
        assert self.old_stem_parameter_count==self.new_stem_parameter_count==6474
