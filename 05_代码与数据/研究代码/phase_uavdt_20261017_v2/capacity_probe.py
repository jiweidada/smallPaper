"""Disposable real-batch 1/2/3-process AMP capacity and aggregate throughput measurement."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

def worker(root,arm,case):
    import numpy as np
    import torch
    from uavdt_protocol import install_native_adapter
    install_native_adapter()
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    plan=json.loads((root/'plan.json').read_text());dist_utils.setup_seed(plan['seed']);torch.set_num_threads(4)
    cfg=YAMLConfig(plan['configs'][arm]);model=cfg.model
    state=torch.load(plan['initialization'],map_location='cpu',weights_only=False)['model']
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected and all('p2_skip' in k for k in missing)
    model=model.cuda().train();criterion=cfg.criterion.cuda();criterion.set_model(model);criterion.set_epoch(4)
    optimizer=cfg.optimizer;scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    ema=cfg.ema
    if ema is not None:ema=ema.to('cuda')
    loader=cfg.train_dataloader;loader.set_epoch(4);dataset=loader.dataset
    # Memory stress uses real high-density GT and forced enabled training augmentations.
    dataset._transforms.mosaic_prob=1.;loader.collate_fn.mixup_prob=1.
    density=sorted(range(len(dataset)),key=lambda i:len(dataset.coco.imgToAnns[dataset.ids[i]]),reverse=True)[:4]
    samples,targets=loader.collate_fn([dataset[i] for i in density])
    dense_counts=[len(t['labels']) for t in targets]
    def update(samples,targets,step,epoch):
        samples=samples.cuda(non_blocking=True)
        targets=[{k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):outputs=model(samples,targets=targets)
        with torch.autocast('cuda',enabled=False):losses=criterion(outputs,targets,epoch=epoch,step=step,global_step=step,epoch_step=len(loader))
        loss=sum(losses.values());assert torch.isfinite(loss).all(),'Nonfinite disposable AMP loss'
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if ema is not None:ema.update(model)
        return float(loss.detach()),int(scaler.get_scale()<before)
    torch.cuda.reset_peak_memory_stats();skips=0
    for step in range(3):_,s=update(samples,targets,step,4);skips+=s
    dense_peak=torch.cuda.max_memory_allocated();del samples,targets;torch.cuda.empty_cache()
    dataset._transforms.mosaic_prob=.5;loader.collate_fn.mixup_prob=.5;loader.set_epoch(0);criterion.set_epoch(0)
    iterator=iter(loader)
    for step in range(8):samples,targets=next(iterator);_,s=update(samples,targets,step,0);skips+=s
    dump(case/(arm+'_ready.json'),{'pid':os.getpid(),'dense_GT_per_image':dense_counts,'dense_peak_allocated':dense_peak})
    deadline=time.monotonic()+300
    while not (case/'start.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Disposable capacity start barrier timed out')
        time.sleep(.05)
    torch.cuda.synchronize();started=time.monotonic();seconds=[]
    for step in range(40):
        one=time.monotonic();samples,targets=next(iterator);loss,s=update(samples,targets,step+8,0);skips+=s
        torch.cuda.synchronize();seconds.append(time.monotonic()-one)
    elapsed=time.monotonic()-started
    assert skips<51,'All disposable AMP updates skipped; capacity result is invalid'
    record={'arm':arm,'pid':os.getpid(),'passed':True,'disposable_updates':51,'timed_steps':40,'batch_size':4,
            'timed_seconds':elapsed,'seconds_per_step_mean':float(np.mean(seconds)),
            'seconds_per_step_p95':float(np.percentile(seconds,95)),'images_per_second':160/elapsed,
            'dense_GT_per_image':dense_counts,'dense_peak_allocated_bytes':dense_peak,
            'max_allocated_bytes':torch.cuda.max_memory_allocated(),'amp_skipped_steps':skips,
            'formal_training_weights_written':False,'final_disposable_loss':loss}
    dump(case/(arm+'_result.json'),record)

def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    assert p.returncode==0
    a=p.stdout.strip().splitlines()[0].split(',');return {'used_MiB':int(a[0]),'free_MiB':int(a[1]),'utilization_percent':int(a[2])}

def parent(root):
    target=root/'capacity_v2';target.mkdir(exist_ok=False)
    initial=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not any(s.strip().isdigit() for s in initial.splitlines()),'GPU must be idle before disposable capacity testing'
    initial_free=gpu()['free_MiB'];cases=[]
    for count,arms in [(1,['c']),(2,['c','c_mean']),(3,['c','c_mean','baseline'])]:
        case=target/str(count);case.mkdir();children={};handles={};peak=0;minimum_free=10**9;start=time.monotonic();released=False
        for arm in arms:
            log=(case/(arm+'.log')).open('xb');handles[arm]=log
            children[arm]=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),str(root),'--worker',arm,'--case',str(case)],
                cwd=root/'code',env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        problem=None
        while any(p.poll() is None for p in children.values()):
            snapshot=gpu();peak=max(peak,snapshot['used_MiB']);minimum_free=min(minimum_free,snapshot['free_MiB'])
            failed=[a for a,p in children.items() if p.poll() is not None and p.returncode!=0]
            if failed:problem='Disposable probe failed: '+','.join(failed);break
            if not released and all((case/(a+'_ready.json')).exists() for a in arms):
                dump(case/'start.json',{'started_monotonic':time.monotonic()});released=True
            if time.monotonic()-start>900:problem='Disposable capacity case exceeded 900 seconds';break
            time.sleep(.4)
        if problem:
            for p in children.values():
                if p.poll() is None:p.terminate()
        for p in children.values():p.wait(timeout=30)
        for h in handles.values():h.close()
        results={a:json.loads((case/(a+'_result.json')).read_text()) for a in arms if (case/(a+'_result.json')).exists()}
        passed=len(results)==count and all(p.returncode==0 for p in children.values())
        throughput=sum(r['timed_steps']*r['batch_size'] for r in results.values())/max(r['timed_seconds'] for r in results.values()) if passed else 0.
        record={'concurrency':count,'passed':passed,'peak_GPU_used_MiB':peak,'minimum_GPU_free_MiB':minimum_free,
                'aggregate_images_per_second':throughput,'workers':results,'failure':problem,
                'conservative_sum_peak_plus_context_MiB':sum(r['max_allocated_bytes'] for r in results.values())/1024**2+768*count if passed else None,
                'process_returncodes':{a:p.returncode for a,p in children.items()}}
        dump(case/'case_result.json',record);cases.append(record);print(json.dumps({k:v for k,v in record.items() if k!='workers'}),flush=True)
        if not passed:break
    safe=[c for c in cases if c['passed'] and c['minimum_GPU_free_MiB']>=2048
          and c['conservative_sum_peak_plus_context_MiB']+2048<=initial_free]
    assert safe,'No safe real-batch AMP concurrency fits with 2 GiB headroom; no formal training may start'
    best=max(c['aggregate_images_per_second'] for c in safe)
    selected=min(c['concurrency'] for c in safe if c['aggregate_images_per_second']>=best*.95)
    record={'state':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'settings':{'batch':4,'input':[640,640],'AMP':True,'threads':4,'dense_Mosaic_Mixup_stress':True,'headroom_required_MiB':2048,
                        'selection':'Prefer the lower concurrency when aggregate throughput differs by at most 5%.'},
            'cases':cases,'selected_concurrency':selected,'maximum_tested_safe_concurrency':max(c['concurrency'] for c in safe),
            'formal_training_started':False}
    dump(root/'capacity_receipt.json',record);print(json.dumps({'selected_concurrency':selected,'capacity_receipt':str(root/'capacity_receipt.json')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--worker');parser.add_argument('--case',type=Path);a=parser.parse_args()
    if a.worker:worker(a.root,a.worker,a.case)
    else:parent(a.root)
