"""One bounded capacity run: three complete forwards, at most two disposable updates."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib, hashlib, json, os, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import a20_host as host
import torch
from PIL import Image

CALLS=0
UPDATES=0
RECORDS=[]
LAST_SHAPES=[]
PHASE='before_initialization'

def utc():return datetime.now(timezone.utc).isoformat()
def event(x):
    x={'utc':utc(),**x}
    with (ROOT/'capacity_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(x,ensure_ascii=False),flush=True)

def enter(module,args):
    global CALLS
    assert CALLS<3,'Frozen forward budget exhausted'
    CALLS+=1
    event({'event':'COMPLETE_FORWARD_ENTER','stage':PHASE,'call_index':CALLS,'input_shape':list(args[0].shape)})

def shape_hook(module,args,result):
    global LAST_SHAPES
    LAST_SHAPES=[list(x.shape) for x in result]
    B=args[0].shape[0]
    assert LAST_SHAPES==[[B,96,200,200],[B,192,100,100],[B,384,50,50],[B,768,25,25]],LAST_SHAPES

def finite_tree(x,path='output'):
    if isinstance(x,torch.Tensor) and (x.is_floating_point() or x.is_complex()):
        assert bool(torch.isfinite(x).all()),'Nonfinite '+path
    elif isinstance(x,dict):
        for k,v in x.items():finite_tree(v,path+'.'+str(k))
    elif isinstance(x,(list,tuple)):
        for i,v in enumerate(x):finite_tree(v,path+'.'+str(i))

def memory():
    return {'allocated_peak_MiB':torch.cuda.max_memory_allocated()/1048576,
            'reserved_peak_MiB':torch.cuda.max_memory_reserved()/1048576,
            'allocated_end_MiB':torch.cuda.memory_allocated()/1048576,
            'reserved_end_MiB':torch.cuda.memory_reserved()/1048576}

def main():
    global PHASE,UPDATES
    protocol=json.loads((ROOT/'capacity_protocol_FROZEN.json').read_text(encoding='utf-8'))
    assert protocol['status']=='FROZEN_BEFORE_ANY_NEW_FORWARD'
    for row in protocol['files']:
        assert host.sha(ROOT/row['path'])==row['sha256'],row['path']
    assert not (ROOT/'capacity_one_shot.lock').exists(),'Existing execution lock: inspect, never rerun'
    lock=os.open(str(ROOT/'capacity_one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(lock,json.dumps({'utc':utc(),'pid':os.getpid(),'protocol_sha256':host.sha(ROOT/'capacity_protocol_FROZEN.json')}).encode());os.close(lock)
    event({'event':'CAPACITY_PROCESS_START','pid':os.getpid(),'protocol_sha256':host.sha(ROOT/'capacity_protocol_FROZEN.json')})
    torch.set_num_threads(4)
    model,criterion,transform,load=host.build_host()
    assert load['host_state_digest']==protocol['expected_host_state_digest']
    model.register_forward_pre_hook(enter)
    model.backbone.register_forward_hook(shape_hook)
    data=json.loads((ROOT/'capacity_inputs/input_manifest.json').read_text(encoding='utf-8'))
    tensors=[];targets=[]
    for i,c in enumerate(data['cases']):
        p=ROOT/'capacity_inputs'/c['image_file']
        assert host.sha(p)==c['image_sha256']
        assert host.sha(ROOT/'capacity_inputs'/c['annotation_file'])==c['annotation_sha256']
        with Image.open(p) as im:tensors.append(transform(im.convert('RGB')).squeeze(0))
        targets.append({'labels':torch.tensor(c['labels'],dtype=torch.int64),
            'boxes':torch.tensor(c['boxes_cxcywh_normalized'],dtype=torch.float32),
            'image_id':torch.tensor([i],dtype=torch.int64),
            'area':torch.tensor(c['resized_box_area'],dtype=torch.float32),
            'iscrowd':torch.zeros(c['valid_gt_count'],dtype=torch.int64),
            'orig_size':torch.tensor(c['orig_size_hw'],dtype=torch.int64),
            'size':torch.tensor([800,800],dtype=torch.int64)})
    gpu_snapshot=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    event({'event':'GPU_AVAILABILITY_BEFORE_ALLOCATION','snapshot':gpu_snapshot.strip()})
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    assert free>=20*1024**3,('Less than frozen20GiB available',free)
    model=model.cuda();criterion=criterion.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(0.9,0.999),weight_decay=0.000065,foreach=False)
    original_step=optimizer.step
    def counted_step(*args,**kwargs):
        global UPDATES
        assert UPDATES<2,'Frozen optimizer budget exhausted'
        result=original_step(*args,**kwargs);UPDATES+=1
        event({'event':'DISPOSABLE_OPTIMIZER_UPDATE','update_index':UPDATES,'stage':PHASE})
        return result
    optimizer.step=counted_step
    scaler=torch.amp.GradScaler('cuda',init_scale=128.0,growth_interval=2000)
    torch.manual_seed(protocol['runtime_seed']);torch.cuda.manual_seed_all(protocol['runtime_seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for stage,batch,training in [('eval_fp32_b1',1,False),('train_amp_b1',1,True),('train_amp_b2',2,True)]:
        PHASE=stage
        model.train(training);criterion.train(training)
        optimizer.zero_grad(set_to_none=True)
        images=torch.stack(tensors[:batch]).cuda()
        gt=[{k:v.cuda() for k,v in t.items()} for t in targets[:batch]]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        event({'event':'STAGE_START','stage':stage,'batch':batch,'training':training,'target_counts':[len(t['labels']) for t in gt]})
        extra={}
        if not training:
            with torch.inference_mode():output=model(images)
            finite_tree(output)
        else:
            before=UPDATES
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=True):output=model(images,targets=gt)
            finite_tree(output)
            original_nan_to_num=torch.nan_to_num
            guarded_loss_tensors=[]
            def observe_before_nan_sanitize(value,*args,**kwargs):
                assert bool(torch.isfinite(value).all()),'Nonfinite native raw loss before nan_to_num'
                guarded_loss_tensors.append(float(value.detach().sum()))
                return original_nan_to_num(value,*args,**kwargs)
            torch.nan_to_num=observe_before_nan_sanitize
            try:
                with torch.autocast('cuda',enabled=False):losses=criterion(output,gt,epoch=0)
            finally:torch.nan_to_num=original_nan_to_num
            finite_tree(losses,'losses')
            loss=sum(losses.values());assert bool(torch.isfinite(loss))
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            gradients={k:{'finite':0,'nonzero':0,'absent':0} for k in ['backbone','encoder','decoder']}
            absent=[]
            for name,param in model.named_parameters():
                group=name.split('.')[0]
                if param.grad is None:
                    gradients[group]['absent']+=1;absent.append(name);continue
                assert bool(torch.isfinite(param.grad).all()),'Nonfinite gradient '+name
                gradients[group]['finite']+=1
                gradients[group]['nonzero']+=int(bool(torch.any(param.grad!=0)))
            assert all(v['nonzero']>0 for v in gradients.values()),gradients
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),0.1,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();assert UPDATES==before+1,'AMP optimizer step skipped'
            extra={'loss_total':float(loss.detach()),'loss_terms':{k:float(v.detach()) for k,v in losses.items()},
                'raw_loss_sanitizer_guard_count':len(guarded_loss_tensors),'gradient_groups':gradients,
                'parameters_without_gradient':absent,'grad_norm_before_clipping':float(norm),
                'scaler_scale_after':scaler.get_scale(),'optimizer_state_tensors':sum(isinstance(v,torch.Tensor) for st in optimizer.state.values() for v in st.values()),
                'optimizer_state_numel':sum(v.numel() for st in optimizer.state.values() for v in st.values() if isinstance(v,torch.Tensor))}
        torch.cuda.synchronize()
        row={'stage':stage,'batch':batch,'elapsed_seconds':time.perf_counter()-started,
             'backbone_feature_shapes':LAST_SHAPES,'pred_logits_shape':list(output['pred_logits'].shape),
             'pred_boxes_shape':list(output['pred_boxes'].shape),'complete_forward_calls_so_far':CALLS,
             'disposable_optimizer_updates_so_far':UPDATES,**memory(),**extra}
        RECORDS.append(row);event({'event':'STAGE_COMPLETE',**row})
        del output,images,gt
        if training:del loss,losses
    assert CALLS==3 and UPDATES==2
    result={'utc':utc(),'status':'BOUNDED_CAPACITY_PASS_3_FORWARDS_2_DISPOSABLE_UPDATES',
            'protocol_sha256':host.sha(ROOT/'capacity_protocol_FROZEN.json'),'complete_model_forwards':CALLS,
            'capacity_only_optimizer_updates':UPDATES,'formal_training_epochs':0,
            'host_state_digest_before_GPU':load['host_state_digest'],'host_parameters':load['host_parameters'],
            'gpu_name':torch.cuda.get_device_name(0),'gpu_total_MiB':total/1048576,'gpu_snapshot_before_allocation':gpu_snapshot,
            'stages':RECORDS,'no_accuracy_or_failure_mechanism_claim':True,
            'capacity_weights_saved':False,'no_automatic_training_or_followup_job':True}
    host.write(ROOT/'capacity_result.json',result);event({'event':'CAPACITY_TERMINAL','status':result['status'],'forwards':CALLS,'updates':UPDATES})

if __name__=='__main__':
    try:main()
    except Exception:
        result={'utc':utc(),'status':'CAPACITY_STOPPED_WITH_FAILURE_EVIDENCE','stage':PHASE,
                'complete_model_forwards':CALLS,'capacity_only_optimizer_updates':UPDATES,
                'stages_completed':RECORDS,'traceback':traceback.format_exc(),
                'no_automatic_retry_or_configuration_change':True}
        if torch.cuda.is_initialized():result['memory']=memory()
        host.write(ROOT/'capacity_result.json',result);event({'event':'CAPACITY_TERMINAL_FAILURE',**result})
        raise
