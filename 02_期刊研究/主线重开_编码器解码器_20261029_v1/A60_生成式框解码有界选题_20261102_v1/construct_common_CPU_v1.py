"""One real CPU detector constructor, no complete detector forwards."""
from pathlib import Path
import hashlib
import json
import re
import sys
import time

P=Path(__file__).resolve().parent
if sys.platform=='win32':
    sys.path.insert(0,str(P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.misc import FrozenBatchNorm2d
from model_A60_v1 import EndpointDetector

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(4<<20),b''):h.update(v)
    return h.hexdigest()

def backbone_key(k):
    if k=='stem.conv1.weight':return 'conv1.weight'
    if k.startswith('stem.conv1.norm.'):return 'bn1.'+k.split('stem.conv1.norm.',1)[1]
    m=re.fullmatch(r'res([2-5])\.(\d+)\.(.+)',k);assert m,k
    level,index,tail=m.groups();prefix=f'layer{int(level)-1}.{index}.'
    if tail.startswith('shortcut.norm.'):return prefix+'downsample.1.'+tail.split('shortcut.norm.',1)[1]
    if tail.startswith('shortcut.'):return prefix+'downsample.0.'+tail.split('shortcut.',1)[1]
    tail=re.sub(r'conv([1-3])\.norm\.',r'bn\1.',tail)
    return prefix+tail

def main():
    out=P/'CPU_constructor_stage_v1';out.mkdir(exist_ok=False)
    marker=dict(CPU_detector_constructors_allocated=1,full_detector_forwards_allocated=0,
        reference_backbone_constructors_allocated=1,backbone_module_forwards_allocated=2,
        head_module_forwards_allocated=2,optimizer_updates_allocated=0)
    with (out/'run_once.json').open('x') as f:json.dump(marker,f,indent=2)
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261102)
    model=EndpointDetector().cpu().eval()
    target=model.state_dict()
    warm_path=P/'assets_local_v1/official_diffdet_coco_res50_300boxes.pth'
    assert sha(warm_path)=='31c31e39f7b44bd9735c3181fdb0ea8bd092aa64229067866e4f56480efa6f49'
    warm=torch.load(warm_path,map_location='cpu',weights_only=True)['model']
    sampling={'betas','alphas_cumprod','alphas_cumprod_prev','sqrt_alphas_cumprod',
        'sqrt_one_minus_alphas_cumprod','log_one_minus_alphas_cumprod','sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod','posterior_variance','posterior_log_variance_clipped',
        'posterior_mean_coef1','posterior_mean_coef2'}
    class_keys={f'head.head_series.{n}.class_logits.{s}' for n in range(6) for s in ['weight','bias']}
    assert set(warm)-set(target)==sampling
    assert set(target)-set(warm)==set()
    assert {k for k in target if warm[k].shape!=target[k].shape}==class_keys
    kept={k:v for k,v in warm.items() if k in target and k not in class_keys}
    missing,unexpected=model.load_state_dict(kept,strict=False)
    assert set(missing)==class_keys and not unexpected
    assert all(torch.equal(model.state_dict()[k],v) for k,v in kept.items())
    del warm,kept
    # Independent TorchVision standard R50/FPN reference checks the reconstruction.
    ref=resnet50(weights=None,norm_layer=FrozenBatchNorm2d).eval()
    state={backbone_key(k):v for k,v in model.backbone.bottom_up.state_dict().items()}
    missing,unexpected=ref.load_state_dict(state,strict=False)
    assert set(missing)=={'fc.weight','fc.bias'} and not unexpected
    body=IntermediateLayerGetter(ref,{'layer1':'res2','layer2':'res3','layer3':'res4','layer4':'res5'})
    fpn=FeaturePyramidNetwork([256,512,1024,2048],256).eval()
    fpn_state={}
    for j,level in enumerate(range(2,6)):
        for suffix in ['weight','bias']:
            fpn_state[f'inner_blocks.{j}.0.{suffix}']=getattr(model.backbone,'fpn_lateral'+str(level)).state_dict()[suffix]
            fpn_state[f'layer_blocks.{j}.0.{suffix}']=getattr(model.backbone,'fpn_output'+str(level)).state_dict()[suffix]
    fpn.load_state_dict(fpn_state,strict=True)
    with torch.no_grad():
        image=torch.randn(1,3,64,64)
        own=model.backbone(image)
        standard=fpn(body(image))
        backbone_errors={a:float((own[a]-standard[b]).abs().max()) for a,b in zip(own,standard)}
        for a,b in zip(own,standard):assert torch.allclose(own[a],standard[b],atol=1e-4,rtol=1e-5),(a,backbone_errors[a])
        boxes=torch.tensor([[[0.,0.,64.,64.],[0.,0.,16.,16.],[8.,8.,24.,24.],[24.,16.,56.,48.]]])
        features=[own[k] for k in ['p2','p3','p4','p5']]
        a=model.head(features,boxes,torch.tensor([999.]),None)
        b=model.head(features,boxes,torch.full((1,4),999.),None)
        head_errors=dict(logits=float((a[0]-b[0]).abs().max()),boxes=float((a[1]-b[1]).abs().max()))
        for x,y in zip(a,b):assert torch.allclose(x,y,atol=1e-4,rtol=1e-5),head_errors
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    initial=out/'common_init_A60_v1.pth'
    torch.save(dict(model=model.state_dict(),seed=20261102,
        source_checkpoint_sha256=sha(warm_path),classes=10,algorithm='Single-step endpoint CFM; source architecture with per-query time support'),initial)
    result=dict(status='PASS_CPU_CONSTRUCTOR_WARM_MAPPING_AND_STANDARD_COMPONENT_CHECKS',
        model_parameters=sum(p.numel() for p in model.parameters()),state_tensors=len(target),
        warm_loaded_state_tensors=len(target)-12,reset_classification_keys=sorted(class_keys),
        unused_diffusion_sampler_buffers=sorted(sampling),backbone_reference_max_errors=backbone_errors,
        per_query_vs_broadcast_time_max_errors=head_errors,common_init_sha256=sha(initial),
        common_init_bytes=initial.stat().st_size,torch_version=torch.__version__,
        CPU_detector_constructors=1,CPU_reference_backbone_constructors=1,backbone_module_forwards=2,
        head_module_forwards=2,full_detector_forwards=0,optimizer_updates=0,backwards=0,
        monotonic_seconds=time.monotonic()-start,
        limits='Standard component and state checks, not author FlowDet numerical reproduction or GPU capacity/learning validation.')
    with (out/'result.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
