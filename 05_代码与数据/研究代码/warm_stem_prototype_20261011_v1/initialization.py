"""Load all original common-init tensors, remapping only the retained native Stem prefix."""
import hashlib
from pathlib import Path
import torch

COMMON_INIT_SHA='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'

def load_warm_common(model,path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=COMMON_INIT_SHA:
        raise ValueError('Unexpected common initial checkpoint')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    state=payload['ema']['module'] if 'ema' in payload else payload['model']
    old_prefix='backbone.stem.'
    mapped={('backbone.stem.native.'+n[len(old_prefix):] if n.startswith(old_prefix) else n):v
            for n,v in state.items()}
    current=model.state_dict()
    extra=set(current)-set(mapped)
    expected_extra={n for n in current if n.startswith('backbone.stem.enhanced.')}|{'backbone.stem.residual_gain'}
    if extra!=expected_extra or set(mapped)-set(current):
        raise ValueError('Unexpected common-init tensor layout')
    if not all(current[n].shape==v.shape and current[n].dtype==v.dtype for n,v in mapped.items()):
        raise ValueError('Common-init tensor interface mismatch')
    loaded=model.load_state_dict(mapped,strict=False)
    if set(loaded.missing_keys)!=extra or loaded.unexpected_keys:
        raise ValueError('Common-init loading omitted original tensors')
    after=model.state_dict()
    if not all(torch.equal(after[n],v) for n,v in mapped.items()):
        raise ValueError('Original common-init tensor changed')
    if not torch.equal(model.backbone.stem.residual_gain,torch.zeros_like(model.backbone.stem.residual_gain)):
        raise ValueError('Residual does not start at zero contribution')
    outside=sum(not n.startswith(old_prefix) for n in state)
    retained=len(state)-outside
    if (outside,retained)!=(754,40):
        raise ValueError('Unexpected retained native tensor counts')
    return dict(common_init_sha256=COMMON_INIT_SHA,exact_original_tensors=len(mapped),
                outside_Stem_tensors=outside,retained_native_Stem_tensors=retained,
                enhanced_state_tensors=len(extra)-1,gain_parameters=model.backbone.stem.residual_gain.numel(),
                native_initialization_retained=True)

