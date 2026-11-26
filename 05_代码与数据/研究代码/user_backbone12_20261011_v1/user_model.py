"""Three independent user changes on the historical native detector constructor."""
import torch
from torch import nn
from geometry import ResidualGeometryDEIM
from user_attention import HG_Block
from user_p2 import P2ToP3Residual, P2SkipBackbone

ARMS = ('backbone_silu', 'backbone_attention', 'p2_skip')

class UserBackboneDEIM(ResidualGeometryDEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline', user_arm=None):
        assert arm == 'baseline' and user_arm in ARMS
        super().__init__(recipe, arm=arm)
        self.user_arm = user_arm
        self.changed_paths = []
        self.removed_common_prefixes = []
        self.new_state_prefixes = []
        assert recipe['HGNetv2']['name'] == 'B0'
        assert recipe['HGNetv2']['freeze_at'] < 0 and recipe['HGNetv2']['freeze_norm'] is False
        native_rng = torch.get_rng_state().clone()
        if user_arm == 'backbone_silu':
            for name, module in list(self.backbone.named_modules()):
                for child_name, child in list(module.named_children()):
                    if isinstance(child, nn.ReLU):
                        setattr(module, child_name, nn.SiLU(inplace=True))
                        self.changed_paths.append('backbone.' + (name + '.' if name else '') + child_name)
            assert self.changed_paths
        elif user_arm == 'backbone_attention':
            configs = self.backbone.arch_configs['B0']['stage_config']
            assert len(self.backbone.stages) == len(configs) == 4
            methods = ('ese', 'simam', 'msca_lite', 'msca_dilated')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed())
                for i, values in enumerate(configs.values()):
                    cin, mid, cout, blocks, down, light, kernel, layers = values
                    assert len(self.backbone.stages[i].blocks) == blocks
                    for j, native in enumerate(self.backbone.stages[i].blocks):
                        replacement = HG_Block(cin if j == 0 else cout, mid, cout, layers,
                            kernel_size=kernel, residual=j > 0, light_block=light,
                            use_lab=recipe['HGNetv2']['use_lab'], agg=methods[i], act_type='relu')
                        native.aggregation = replacement.aggregation
                        prefix = f'backbone.stages.{i}.blocks.{j}.aggregation.'
                        self.changed_paths.append(prefix[:-1])
                        self.removed_common_prefixes.append(prefix)
                        self.new_state_prefixes.append(prefix)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed())
                self.backbone.p2_skip = P2ToP3Residual(64, 256, 16)
            self.backbone.__class__ = P2SkipBackbone
            self.changed_paths = ['backbone.p2_skip', 'backbone.forward:stage0_to_stage1']
            self.new_state_prefixes = ['backbone.p2_skip.']
        assert torch.equal(native_rng, torch.get_rng_state()), 'Added construction changed native data/DN RNG stream'
        self.added_construction_rng_unchanged = True

def load_user_common(model, path):
    import hashlib
    assert hashlib.sha256(open(path, 'rb').read()).hexdigest() == '85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    original = payload['ema']['module'] if 'ema' in payload else payload['model']
    assert len(original) == 794
    current = model.state_dict()
    kept = {n:v for n,v in original.items() if not any(n.startswith(p) for p in model.removed_common_prefixes)}
    added = {n:v.detach().clone() for n,v in current.items() if any(n.startswith(p) for p in model.new_state_prefixes)}
    assert set(kept).isdisjoint(added) and set(current) == set(kept) | set(added)
    assert all(n in current and v.shape == current[n].shape and v.dtype == current[n].dtype for n,v in kept.items())
    model.load_state_dict(dict(kept, **added), strict=True)
    loaded = model.state_dict()
    assert all(torch.equal(loaded[n],v) for n,v in kept.items())
    assert all(torch.equal(loaded[n],v) for n,v in added.items())
    if model.user_arm in ('backbone_silu','p2_skip'):
        assert len(kept) == 794
    if model.user_arm == 'p2_skip':
        assert torch.count_nonzero(model.backbone.p2_skip.restore.weight) == 0
    count = sum(p.numel() for p in model.parameters())
    return dict(common_init_sha256='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91',
        user_arm=model.user_arm, exact_shared_tensors=len(kept), new_state_tensors=len(added),
        removed_native_tensors=len(original)-len(kept), state_tensors=len(current), parameters=count,
        extra_parameters=count-10231997, changed_paths=model.changed_paths,
        removed_common_prefixes=model.removed_common_prefixes, new_state_prefixes=model.new_state_prefixes,
        strict_load=True, all_shared_tensors_exact=True, new_state_unchanged=True,
        initialization_scope='New attention aggregations use forked seed initialization; attention arm changes aggregation architecture and its initialization. SiLU retains all794 original states; P2 adds only the historical width16 bridge.')
