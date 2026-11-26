"""CPU-only model construction and exact shared initialization; no forward."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'host_compat_v1'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert P==Path('/root/uav_a57_20261102_v1')
    assert not (P/'common_initialization.pth').exists()
    assert not (P/'native_initialization_result.json').exists()
    manifest=json.loads((P/'native_constructor_inputs_FROZEN_v1.json').read_text())
    assert all(sha(P/n)==h for n,h in manifest['files_sha256'].items())
    weight=P/'assets/vim_t_midclstok_76p1acc.pth'
    assert sha(weight)=='fedb1e60267f3bc7b2a8443f2ddead8d39e019d10247a87252162b3789801fd5'
    import torch
    from model import MambaSOD
    torch.manual_seed(20261101)
    model=MambaSOD(img_size=640,patch_size=16,vim_depth=24,vim_embed_dim=192,d_state=16,
                   out_channels=256,bifpn_repeats=3,num_queries=400,num_decoder_layers=6,
                   d_ffn=1024,num_classes=10,dropout=0.0,drop_path_rate=0.1,
                   num_groups=6,use_ca=False,use_sa=False,use_fpn=False)
    # Only data-only globals needed by the public training checkpoint are allowed.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    source=checkpoint['model']
    assert all(isinstance(t,torch.Tensor) for t in source.values())
    target=model.encoder.backbone.state_dict()
    pretrain_keys={n for n in target if n.startswith(('patch_embed.','layers.','norm_f.')) or n in ['pos_embed','cls_token']}
    assert pretrain_keys<=set(source),sorted(pretrain_keys-set(source))
    assert set(source)-pretrain_keys=={'head.weight','head.bias'},sorted(set(source)-pretrain_keys)
    resized=None
    selected={}
    for n in sorted(pretrain_keys):
        value=source[n]
        if n=='pos_embed':
            assert value.shape==(1,197,192) and target[n].shape==(1,1601,192)
            cls=value[:,98:99]
            patches=torch.cat([value[:,:98],value[:,99:]],dim=1)
            patches=patches.transpose(1,2).reshape(1,192,14,14)
            patches=torch.nn.functional.interpolate(patches,size=(40,40),mode='bicubic',align_corners=False)
            patches=patches.flatten(2).transpose(1,2)
            value=torch.cat([patches[:,:800],cls,patches[:,800:]],1)
            resized={'source_grid':[14,14],'target_grid':[40,40],'source_middle_CLS_index':98,
                     'target_middle_CLS_index':800,'mode':'bicubic, align_corners=False'}
        assert value.shape==target[n].shape and value.dtype==target[n].dtype,(n,value.shape,target[n].shape)
        assert torch.isfinite(value).all(),n
        selected[n]=value
    missing,unexpected=model.encoder.backbone.load_state_dict(selected,strict=False)
    assert not unexpected and not (set(missing)&pretrain_keys)
    assert all(torch.equal(model.encoder.backbone.state_dict()[n],t) for n,t in selected.items())
    states=model.state_dict()
    digest=hashlib.sha256()
    for n,t in states.items():
        digest.update(n.encode());digest.update(t.contiguous().numpy().tobytes())
    before=digest.hexdigest()
    from native_variants import install
    for arm in ['M','Q']:
        install(model,arm)
        after=hashlib.sha256()
        for n,t in model.state_dict().items():
            after.update(n.encode());after.update(t.contiguous().numpy().tobytes())
        assert after.hexdigest()==before
    torch.save({'model':states,'seed':20261101,'parameter_tensor_digest':before,
                'scope':'Shared source-derived MambaSOD-code initialization; no detector forward or update'},P/'common_initialization.pth')
    out={'status':'CONSTRUCTED_INITIALIZED_NO_FORWARD','parameters':sum(t.numel() for t in model.parameters()),
         'state_tensors':len(states),'loaded_pretrained_tensors':len(selected),
         'new_backbone_state_names':sorted(missing),'position_embedding_conversion':resized,
         'parameter_tensor_digest':before,'shared_state_keys_and_values_B_M_Q':True,
         'common_initialization_sha256':sha(P/'common_initialization.pth'),
         'whole_detector_forwards':0,'detector_optimizer_updates':0,'GPU_component_forwards':0,
         'frozen_manifest_sha256':sha(P/'native_constructor_inputs_FROZEN_v1.json'),
         'limitations':['No full-paper reproduction, capacity, output equivalence, or AP claim.','Variant installation changes forward logic only; it adds no parameters.']}
    with (P/'native_initialization_result.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
