"""Bounded train-output probe: frozen raw checkpoint, no optimizer or model gradients."""
import argparse,datetime,hashlib,json,pathlib,random,subprocess,time,traceback
import numpy as np
import torch
from PIL import Image
from probe_runtime import ROOT,verify_package,sha,build_model,cpu_tree
from evaluate_rules import analyze

def state_sha(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main(seed):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    package=verify_package();plan=json.loads((ROOT/'sample_plan.json').read_text());recipe=json.loads((ROOT/'baseline_recipes.json').read_text())[str(seed)]
    cp=ROOT/'assets'/f'baseline24_seed{seed}.pth';assert sha(cp)==plan['checkpoints'][str(seed)]['sha256']
    resultdir=ROOT/f'results_seed{seed}';resultdir.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text());assert manifest['formal_inputs']==64 and manifest['plan_sha256']==sha(ROOT/'sample_plan.json')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not busy.strip(),'GPU still has a compute process; old backup/release must finish first'
    raw=torch.load(cp,map_location='cpu',weights_only=False);assert raw['last_epoch']==23
    model=build_model(recipe);model.load_state_dict(raw['model'],strict=True);del raw
    model.requires_grad_(False);model.train();before=state_sha(model)
    model.cuda();saved_buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    def restore():
        with torch.no_grad():
            for name,b in model.named_buffers():b.copy_(saved_buffers[name])
    total_started=time.monotonic();events=[]
    def run_batch(images,targets,tag,dn_seed,gradient_check=False):
        restore();py_rng=random.getstate();np_rng=np.random.get_state();cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
        random.seed(dn_seed);np.random.seed(dn_seed);torch.manual_seed(dn_seed);torch.cuda.manual_seed_all(dn_seed)
        gt=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        with torch.no_grad():output=model(images.cuda(),gt)
        torch.cuda.synchronize();seconds=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated()
        output=cpu_tree(output);restore();random.setstate(py_rng);np.random.set_state(np_rng);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        frozen={'outputs':output,'targets':cpu_tree(targets),'tag':tag,'forward_DN_seed':dn_seed}
        torch.save(frozen,resultdir/(tag+'_predictions.pt'))
        analysis=analyze(output,cpu_tree(targets),recipe,check_grad=gradient_check)
        analysis.update(tag=tag,forward_seconds=seconds,peak_allocated_bytes=peak,optimizer_steps=0,model_parameter_gradients=0)
        (resultdir/(tag+'_analysis.json')).write_text(json.dumps(analysis,indent=2,allow_nan=False))
        event={'tag':tag,'forward_seconds':seconds,'peak_allocated_MiB':peak/2**20,'GT':[len(t['labels']) for t in targets],'missing':[j['GT']-j['native_covered'] for j in analysis['inputs']]}
        events.append(event);(resultdir/'progress.json').write_text(json.dumps({'seed':seed,'finished_batches':len(events),'last':event,'elapsed_seconds':time.monotonic()-total_started}))
        print(json.dumps(event),flush=True)
    for bi in range(2):
        images=[];targets=[]
        for k in range(4*bi,4*bi+4):
            p=ROOT/'smoke'/f'{k:03d}.png';z=np.load(ROOT/'smoke'/f'{k:03d}_targets.npz')
            images.append(torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1).float()/255)
            targets.append({'boxes':torch.from_numpy(z['boxes'].copy()).float(),'labels':torch.from_numpy(z['labels'].copy()).long()})
        run_batch(torch.stack(images),targets,f'smoke{bi:02d}',39261019+bi,gradient_check=(bi==0))
    for item in manifest['batches']:
        path=ROOT/'inputs'/item['file'];assert sha(path)==item['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert hashlib.sha256(payload['images'].numpy().tobytes()).hexdigest()==item['float_input_sha256']
        run_batch(payload['images'],payload['targets'],f"batch{item['batch']:02d}",payload['spec']['forward_DN_seed'],gradient_check=(item['batch']==0))
    restore();model.cpu();assert state_sha(model)==before;assert sha(cp)==plan['checkpoints'][str(seed)]['sha256'];verify_package()
    result={'state':'complete','seed':seed,'formal_inputs':64,'smoke_inputs':8,'formal_batches':16,'optimizer_steps':0,'model_gradients':0,'BN_and_state_restored':True,'checkpoint_sha256':sha(cp),'plan_sha256':sha(ROOT/'sample_plan.json'),'elapsed_seconds':time.monotonic()-total_started,'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'events':events}
    (resultdir/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True);a=p.parse_args()
    try:main(a.seed)
    except BaseException as e:
        d=ROOT/f'results_seed{a.seed}';d.mkdir(exist_ok=True)
        (d/'failure.json').write_text(json.dumps({'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'no_automatic_retry':True}));raise
