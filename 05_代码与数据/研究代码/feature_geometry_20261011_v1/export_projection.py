"""Read old candidate EMA weights on CPU; no detector construction or forward."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main(a):
    torch.set_num_threads(2)
    root=Path(a.prefix)/'DEIM/experiments/scale_protection12_20261010_v1'
    checkpoint=root/f'seed{a.seed}_target_protect/run/last.pth'
    expected={1:'e195f5a3d126bee9e0f7a373f5eb5fdebb78ada9fc891780e251be9dafc2ae94',2:'8113eb35c4ca0519e9757a101327cb95e709ebd3609559f55c240e1a3a472269'}
    assert sha(checkpoint)==expected[a.seed]
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    identity=payload['scale_protection12_identity']
    assert payload['last_epoch']==11 and identity['seed']==a.seed and identity['arm']=='target_protect'
    state=payload['ema']['module'];assert len(state)==794
    names=['conv.weight','norm.weight','norm.bias','norm.running_mean','norm.running_var']
    arrays={n.replace('.','_'):state['encoder.input_proj.0.'+n].detach().cpu().numpy().copy() for n in names}
    assert arrays['conv_weight'].shape==(256,256,1,1)
    assert all(np.isfinite(x).all() for x in arrays.values())
    assert (arrays['norm_running_var']>=0).all() and not torch.cuda.is_initialized()
    eps=torch.nn.BatchNorm2d(256).eps
    dest=Path(a.destination);dest.mkdir(parents=True,exist_ok=False)
    np.savez(dest/f'projection_seed{a.seed}_v1.npz',**arrays,BN_eps=np.array(eps,dtype=np.float64))
    p=dest/f'projection_seed{a.seed}_v1.npz'
    report=dict(status='EXPORTED_OLD_TARGET_EMA_P3_AFFINE_WEIGHTS_CPU_ONLY',seed=a.seed,
        checkpoint_sha256=expected[a.seed],last_epoch=11,identity=identity,
        array_sha256=sha(p),array_bytes=p.stat().st_size,keys=sorted(arrays),BN_eps=eps,
        export_source_sha256=sha(__file__),CUDA_initialized=False,detector_constructions=0,
        detector_forwards=0,optimizer_updates=0,epochs_added=0,
        completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (dest/f'projection_seed{a.seed}_v1_metadata.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--destination',required=True)
    main(p.parse_args())
