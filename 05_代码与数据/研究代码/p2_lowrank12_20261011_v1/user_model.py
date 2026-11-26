"""One learned rank16 P2 bridge on the common pretrained native detector."""
import hashlib
import torch
from geometry import ResidualGeometryDEIM
from user_p2 import P2SkipBackbone
from lowrank_branch import LowRankP2Residual

class LowRankBackboneDEIM(ResidualGeometryDEIM):
    __inject__=[]
    __share__=[]
    def __init__(self,recipe,arm='baseline',user_arm=None):
        assert arm=='baseline' and user_arm=='p2_lowrank16'
        super().__init__(recipe,arm=arm)
        assert recipe['HGNetv2']['name']=='B0'
        assert recipe['HGNetv2']['freeze_at']<0 and not recipe['HGNetv2']['freeze_norm']
        self.user_arm=user_arm
        self.removed_common_prefixes=[]
        self.new_state_prefixes=['backbone.p2_skip.']
        self.changed_paths=['backbone.p2_skip','backbone.forward:stage0_to_stage1']
        rng=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed())
            self.backbone.p2_skip=LowRankP2Residual()
        self.backbone.__class__=P2SkipBackbone
        assert torch.equal(rng,torch.get_rng_state())
        self.added_construction_rng_unchanged=True

def load_lowrank_common(model,path):
    assert hashlib.sha256(open(path,'rb').read()).hexdigest()=='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    original=payload['ema']['module'] if 'ema' in payload else payload['model']
    assert len(original)==794
    current=model.state_dict()
    added={n:v.detach().clone() for n,v in current.items() if n.startswith('backbone.p2_skip.')}
    assert len(added)==8 and len(current)==802 and set(original).isdisjoint(added)
    assert set(current)==set(original)|set(added)
    assert all(current[n].dtype==v.dtype and current[n].shape==v.shape for n,v in original.items())
    model.load_state_dict(dict(original,**added),strict=True)
    loaded=model.state_dict()
    assert all(torch.equal(loaded[n],v) for n,v in original.items())
    assert all(torch.equal(loaded[n],v) for n,v in added.items())
    assert torch.count_nonzero(model.backbone.p2_skip.restore.weight)==0
    count=sum(p.numel() for p in model.parameters())
    assert count==10238173 and count-10231997==6176
    return dict(common_init_sha256='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91',user_arm=model.user_arm,exact_shared_tensors=794,
        new_state_tensors=8,removed_native_tensors=0,state_tensors=802,parameters=count,extra_parameters=6176,changed_paths=model.changed_paths,
        removed_common_prefixes=[],new_state_prefixes=model.new_state_prefixes,strict_load=True,all_shared_tensors_exact=True,new_state_unchanged=True,
        initialization_scope='All794 common pretrained native states exact. Reduce initialized from isolated same seed; mix random orthonormal rows; restore zero. No factor initialized from a trained/validation-selected checkpoint. Training/AP efficacy untested.')
