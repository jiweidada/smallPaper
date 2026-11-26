"""Two frozen native-size FBCNN capacity/repeat forwards, no detector or training."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, time
import numpy as np, torch
from PIL import Image
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'restoration_probe_v1'
sys.path.insert(0,str(ROOT/'code'))
from study import json_write
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    with (OUT/'capacity_one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    assert json.loads((ROOT/'simple_codec_gate_v1/completed.json').read_text())['status']=='COMPLETED'
    sources=json.loads((ROOT/'fbcnn_source_v1/source_manifest.json').read_text(encoding='utf-8-sig'))
    for f in sources['files']:assert sha(ROOT/'fbcnn_source_v1'/f['name'])==f['sha256']
    weights=ROOT/'assets/fbcnn_color.pth';assert sha(weights)=='8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8'
    spec=importlib.util.spec_from_file_location('pinned_fbcnn',ROOT/'fbcnn_source_v1/network_fbcnn.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=module.FBCNN(in_nc=3,out_nc=3,nc=[64,128,256,512],nb=4,act_mode='R')
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().float().eval().requires_grad_(False)
    obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images']
    m=max(obs,key=lambda x:(x['width']*x['height'],x['file_name']))
    assert sha(m['q20'])==m['jpeg_sha256']
    with Image.open(m['q20']) as im:a=np.asarray(im.convert('RGB'))
    inp=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).unsqueeze(0).cuda().float()/255
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.monotonic()
    with torch.inference_mode():y,qf=model(inp)
    torch.cuda.synchronize();elapsed=time.monotonic()-tic
    assert tuple(y.shape)==tuple(inp.shape) and torch.isfinite(y).all() and torch.isfinite(qf).all()
    restored=y[0].clamp(0,1).cpu().permute(1,2,0).numpy()
    quant=np.rint(restored*255).astype(np.uint8)
    Image.fromarray(quant).save(OUT/'capacity_restored.png',format='PNG')
    first=restored.copy();del y
    with torch.inference_mode():y2,qf2=model(inp)
    torch.cuda.synchronize()
    second=y2[0].clamp(0,1).cpu().permute(1,2,0).numpy()
    repeat_max=float(np.max(np.abs(first-second)))
    assert repeat_max<=1e-6
    result=dict(status='PASS',image=m,file_name=m['file_name'],native_wh=[m['width'],m['height']],
                output_png_sha256=sha(OUT/'capacity_restored.png'),source_commit=sources['commit'],
                weight_sha256=sha(weights),precision='FP32; no AMP/TF32; native resolution, no tiling',
                parameter_count=sum(p.numel() for p in model.parameters()),predicted_qf_raw=qf.cpu().tolist(),
                first_forward_seconds=elapsed,repeat_max_abs_float=repeat_max,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                restorer_forwards=2,detector_forwards=0,training_updates=0,elapsed_seconds=time.monotonic()-start)
    json_write(OUT/'capacity_pass.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
