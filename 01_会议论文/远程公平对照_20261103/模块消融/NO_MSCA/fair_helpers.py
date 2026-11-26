"""One authorized BASE/FULL pair; frozen neural/loss/data/evaluator source.

This isolated driver adds explicit initialization/data controls and evidence
recording. It preserves the original FP32 update order, EMA implementation,
post-update flatcosine LR and validation-dependent late stage-one restores.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, copy, gzip, hashlib, importlib, json, math, random, re, sys, time, traceback
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
METRICS = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']

def now(): return datetime.now(timezone.utc).isoformat()
def readj(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()
def savej(p,v):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n'); os.replace(tmp,p)
def appendj(p,v):
    with Path(p).open('a') as f: f.write(json.dumps(v,allow_nan=False)+'\n')
def savept(p,v):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); torch.save(v,tmp); os.replace(tmp,p)
def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous()
        h.update(json.dumps([k,str(a.dtype),list(a.shape)]).encode()); h.update(a.numpy().tobytes())
    return h.hexdigest()
def batch_hash(samples,targets):
    state={'samples':samples}
    for i,t in enumerate(targets):
        for k,v in t.items():
            if torch.is_tensor(v): state[f'target.{i}.{k}']=v
            else: raise TypeError(f'Unrecorded target field {k}')
    return tensor_hash(state)
def seeds(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def worker_init(_):
    s=torch.initial_seed()%2**32; random.seed(s); np.random.seed(s)

def controls():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    seeds()

def build(arm):
    import source_loader
    source_loader.SNAPSHOT=ROOT/'source_snapshot'
    cfgdata=readj(ROOT/'execution_configs'/f'{arm}.yml')
    workspace,utils,evaluator_module=source_loader.register_selected_source(cfgdata['DEIM']['backbone'])
    for name in ['engine.data.transforms.mosaic','engine.optim','engine.deim.matcher','engine.deim.deim_criterion']:
        importlib.import_module(name)
    for split in ['train','val']:
        for op in cfgdata[f'{split}_dataloader']['dataset']['transforms']['ops']:
            assert '_pymodule' in workspace.GLOBAL_CONFIG.get(op['type'],{}), f"Unregistered transform: {op['type']}"
    mod=importlib.import_module('engine.core.yaml_config')
    cfg=mod.YAMLConfig(str(ROOT/'execution_configs'/f'{arm}.yml'))
    model=cfg.model
    from ablation_model import apply
    assert cfgdata['ablation']['arm'] == arm
    model.ablation_initialization = apply(model, arm, ROOT/'reference')
    criterion=cfg.criterion; post=cfg.postprocessor
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)=={'NO_MPS':7535617,'NO_MSCA':3699662}[arm]
    train=cfg.train_dataloader; val=cfg.val_dataloader
    assert len(train)==1617 and len(train.dataset)==6471 and len(val.dataset)==548
    assert train.batch_size==val.batch_size==4 and train.drop_last and not val.drop_last
    assert train.num_workers==val.num_workers==4
    train.generator=torch.Generator(); train.sampler.generator=torch.Generator()
    train.worker_init_fn=worker_init
    val.generator=torch.Generator().manual_seed(1000000007); val.worker_init_fn=worker_init
    evaluator=evaluator_module.CocoEvaluator(val.dataset.coco,['bbox'])
    # Execute only the unchanged mathematical function, omitting plot imports.
    lrfile=ROOT/'source_snapshot/engine/optim/lr_scheduler.py'
    tree=ast.parse(lrfile.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='flat_cosine_schedule')
    scope={'math':math}; exec(compile(ast.Module(body=[fn],type_ignores=[]),str(lrfile),'exec'),scope)
    return cfg,model,criterion,post,train,val,evaluator,scope['flat_cosine_schedule']

def prepare_epoch(loader,epoch):
    loader.set_epoch(epoch)
    loader.generator.manual_seed(1000003*epoch)
    loader.sampler.generator.manual_seed(1000003*epoch)

@torch.no_grad()
def evaluate(model,post,val,evaluator,out,epoch,epoch_hook):
    model.eval(); evaluator.cleanup(); predictions=[]; seen=[]
    for samples,targets in val:
        samples=samples.cuda(non_blocking=True)
        outputs=model(samples)
        sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
        results=post(outputs,sizes)
        res={int(t['image_id']):{k:v.detach().cpu() for k,v in r.items()} for t,r in zip(targets,results)}
        evaluator.update(res); seen.extend(res)
        for image_id,r in res.items():
            boxes=r['boxes'].clone(); boxes[:,2:]-=boxes[:,:2]
            predictions.extend({'image_id':image_id,'category_id':int(label),'bbox':box,'score':float(score)}
                               for box,score,label in zip(boxes.tolist(),r['scores'].tolist(),r['labels'].tolist()))
    assert sorted(seen)==list(range(548)) and len(predictions)==164400
    evaluator.synchronize_between_processes(); evaluator.accumulate(); evaluator.summarize()
    coco=evaluator.coco_eval['bbox']; assert coco.params.maxDets==[1,10,100]
    values=coco.stats.tolist(); assert len(values)==12 and all(math.isfinite(x) for x in values)
    predpath=out/'latest.predictions.json'
    savej(predpath,predictions)
    arrays=out/'latest.evaluation_arrays.npz'
    np.savez_compressed(arrays,precision=coco.eval['precision'],recall=coco.eval['recall'],scores=coco.eval['scores'])
    # Evidence is captured before any validation-dependent restoration.
    return dict(zip(METRICS,values)),sha(predpath),sha(arrays),tensor_hash(model.state_dict())
