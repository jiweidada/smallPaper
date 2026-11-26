"""Replace only the stem after the historical native constructor."""
import torch
from geometry import ResidualGeometryDEIM
from stem_controls import ControlledStemBlock

class OriginalStemDEIM(ResidualGeometryDEIM):
    __inject__=[]
    __share__=[]
    def __init__(self,recipe,arm='baseline',control=None):
        assert arm=='baseline'
        super().__init__(recipe,arm=arm)
        cfg=recipe['HGNetv2']
        assert cfg['name']=='B0' and cfg.get('freeze_at',-1)<0 and cfg.get('freeze_norm') is False
        channels=self.backbone.arch_configs['B0']['stem_channels']
        assert channels==[3,16,16]
        self.old_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
        # Isolate new-module initialization from the established data/DN RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.backbone.stem=ControlledStemBlock(*channels,use_lab=cfg['use_lab'],act_type='relu')
            self.backbone.stem.set_control(control)
        self.new_stem_parameter_count=sum(p.numel() for p in self.backbone.stem.parameters())
