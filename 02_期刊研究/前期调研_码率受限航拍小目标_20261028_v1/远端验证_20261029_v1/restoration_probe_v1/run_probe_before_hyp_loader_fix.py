"""Finite 2x2 regional restoration diagnostic. No optimizer or deployable policy."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse, copy, hashlib, importlib.util, json, multiprocessing as mp, os, sys, time, traceback
import cv2, numpy as np, torch
from PIL import Image
ROOT=Path('/root/c10_region_restore_20261029_v1');OUT=ROOT/'restoration_probe_v1'
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'simple_codec_gate_v1'))
from study import json_write, state_hash
from evaluate_fixed import prepare_pixels,detection_records,ground_truth_input
from run_gate import metrics,matches
from ultralytics import YOLO
from ultralytics.utils import nms
from ultralytics.utils.loss import v8DetectionLoss
START=time.monotonic()
NAMES=['subset_%02d'%i for i in range(16)]+['alpha_025','alpha_050','alpha_075']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(stage,**kwargs):
    r=dict(stage=stage,elapsed_seconds=time.monotonic()-START,pid=os.getpid(),**kwargs)
    json_write(OUT/'live.json',r);print(json.dumps(r),flush=True)
def weights(h,w,fraction=.05):
    x=(np.arange(w,dtype=np.float32)+.5)/w;y=(np.arange(h,dtype=np.float32)+.5)/h
    wx=np.clip((.5+fraction/2-x)/fraction,0,1);wy=np.clip((.5+fraction/2-y)/fraction,0,1)
    z=np.stack([wy[:,None]*wx[None,:],wy[:,None]*(1-wx[None,:]),(1-wy[:,None])*wx[None,:],(1-wy[:,None])*(1-wx[None,:])])
    assert z.min()>=0 and np.max(np.abs(z.sum(0)-1))<1e-6
    return z
def blend(a,b,ws,index):
    if index==0:return a
    if index==15:return b
    scale=ws[[i for i in range(4) if index&(1<<i)]].sum(0)[...,None]
    return np.rint(np.clip(a.astype(np.float32)+scale*(b.astype(np.float32)-a),0,255)).astype(np.uint8)
def actions(a,b):
    ws=weights(*a.shape[:2]);out=[blend(a,b,ws,i) for i in range(16)]
    for alpha in [.25,.5,.75]:out.append(np.rint(a.astype(np.float32)*(1-alpha)+b.astype(np.float32)*alpha).astype(np.uint8))
    assert np.array_equal(out[0],a) and np.array_equal(out[15],b)
    return out
def dataset():
    gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text())
    obs=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images']
    return gt,obs,ground_truth_input(gt,obs)
def batch_for(gt,image_id):
    valid=[a for a in gt['annotations'] if a['image_id']==image_id and not a['iscrowd']]
    xywh=[]
    for a in valid:
        x,y,w,h=a['bbox'];xywh.append([(x+w/2)/640,(y+h/2)/640,w/640,h/640])
    return dict(batch_idx=torch.zeros(len(valid),device='cuda'),cls=torch.tensor([a['category_id'] for a in valid],device='cuda').float().view(-1,1),
                bboxes=torch.tensor(xywh,device='cuda').float().reshape(-1,4))
def detector(arm):
    c=json.loads((ROOT/'results'/arm/'completed.json').read_text());assert sha(c['primary_checkpoint'])==c['primary_checkpoint_sha256']
    model=YOLO(c['primary_checkpoint']).model.cuda().float().eval().requires_grad_(False)
    if isinstance(model.args,dict):model.args=SimpleNamespace(**model.args)
    model.model[-1].end2end=False;model.model[-1].export=False
    # Reproduce the previously frozen FP32 evaluator's backend settings.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    return model,v8DetectionLoss(model,tal_topk=10)
class FixedAssignment(torch.nn.Module):
    def __init__(self,assigned):super().__init__();self.assigned=assigned
    def forward(self,*args,**kwargs):return self.assigned
def raw_loss(criterion,raw,batch):
    loss,parts=criterion.loss(raw['one2many'] if 'one2many' in raw else raw,batch)
    assert torch.isfinite(loss).all()
    return float(loss.sum()),loss.detach().cpu().tolist()

def contract():
    assert json.loads((OUT/'capacity_pass.json').read_text())['status']=='PASS'
    a=np.arange(35*51*3,dtype=np.uint16).reshape(35,51,3)%256;a=a.astype(np.uint8);b=255-a
    ac=actions(a,b);assert len(ac)==19 and np.array_equal(ac[0],a) and np.array_equal(ac[15],b)
    gt,obs,inp_gt=dataset();meta=obs[0];batch=batch_for(inp_gt,meta['image_id']);rows={}
    for arm in ['D0','DA']:
        model,criterion=detector(arm);capture={}
        hook=criterion.assigner.register_forward_hook(lambda module,args,out:capture.update(assigned=tuple(x.detach().clone() for x in out)))
        with Image.open(meta['q20']) as im:pixels=np.asarray(im.convert('RGB'))
        tensor=prepare_pixels(pixels,meta['geometry'])
        with torch.inference_mode():
            decoded,raw=model(tensor);loss,parts=raw_loss(criterion,raw,batch)
            hook.remove();fixed=v8DetectionLoss(model,tal_topk=10);fixed.assigner=FixedAssignment(capture['assigned'])
            fixed_loss,_=raw_loss(fixed,raw,batch)
            assert abs(loss-fixed_loss)<1e-6
            det=nms.non_max_suppression(decoded,conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=500,nc=10,end2end=False)[0]
        pred,_=detection_records(det,meta)
        old=[p for p in json.loads((ROOT/'evaluation'/arm/'q20_predictions_input.json').read_text()) if p['image_id']==meta['image_id']]
        assert len(pred)==len(old)
        current=np.asarray([p['bbox']+[p['score'],p['category_id']] for p in pred]);previous=np.asarray([p['bbox']+[p['score'],p['category_id']] for p in old])
        delta=float(np.max(np.abs(current-previous))) if len(pred) else 0
        assert delta<1e-4,(arm,delta)
        rows[arm]=dict(loss=loss,loss_parts=parts,fixed_loss=fixed_loss,baseline_prediction_max_abs_delta=delta)
        del model,criterion,fixed,raw,decoded,det,tensor;torch.cuda.empty_cache()
    json_write(OUT/'contract_pass.json',dict(status='PASS',protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),
        region_math_and_exact_endpoints=True,detector_forwards=2,training_updates=0,models=rows))
    event('CONTRACT_PASS',models=rows)

def restore(obs):
    capacity=json.loads((OUT/'capacity_pass.json').read_text());assert capacity['status']=='PASS'
    manifest=json.loads((ROOT/'fbcnn_source_v1/source_manifest.json').read_text(encoding='utf-8-sig'))
    for f in manifest['files']:assert sha(ROOT/'fbcnn_source_v1'/f['name'])==f['sha256']
    spec=importlib.util.spec_from_file_location('pinned_fbcnn',ROOT/'fbcnn_source_v1/network_fbcnn.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.FBCNN(in_nc=3,out_nc=3,nc=[64,128,256,512],nb=4,act_mode='R')
    path=ROOT/'assets/fbcnn_color.pth';assert sha(path)==capacity['weight_sha256']
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model=model.cuda().float().eval().requires_grad_(False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    rows=[];calls=0
    for i,m in enumerate(obs):
        assert sha(m['q20'])==m['jpeg_sha256'];dest=OUT/'restored'/(Path(m['file_name']).stem+'.png');dest.parent.mkdir(exist_ok=True)
        if m['image_id']==capacity['image']['image_id']:
            import shutil
            assert sha(OUT/'capacity_restored.png')==capacity['output_png_sha256'];shutil.copyfile(OUT/'capacity_restored.png',dest)
            qf=capacity['predicted_qf_raw'];elapsed=capacity['first_forward_seconds'];cached=True
        else:
            with Image.open(m['q20']) as im:a=np.asarray(im.convert('RGB'))
            tensor=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).unsqueeze(0).cuda().float()/255
            torch.cuda.synchronize();tic=time.monotonic()
            with torch.inference_mode():y,predqf=model(tensor)
            torch.cuda.synchronize();elapsed=time.monotonic()-tic;calls+=1
            assert torch.isfinite(y).all() and tuple(y.shape)==tuple(tensor.shape)
            img=np.rint(y[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            Image.fromarray(img).save(dest,format='PNG');qf=predqf.cpu().tolist();cached=False
            del y,tensor
        rows.append(dict(image_id=m['image_id'],file=str(dest),sha256=sha(dest),predicted_qf_raw=qf,forward_seconds=elapsed,reused_capacity_image=cached))
        if (i+1)%10==0:event('RESTORATION',images=i+1,new_restorer_forwards=calls)
    json_write(OUT/'restoration_cache.json',dict(images=rows,unique_images=len(rows),new_restorer_forwards=calls,prior_capacity_forwards=2,training_updates=0))
    del model;torch.cuda.empty_cache();return {r['image_id']:r for r in rows}

def selections(losses):
    l=np.asarray(losses,dtype=float);utility=l[0]-l[:16];singles=[utility[1<<i] for i in range(4)]
    add=np.asarray([sum(singles[i] for i in range(4) if mask&(1<<i)) for mask in range(16)])
    pairs={(i,j):utility[(1<<i)|(1<<j)]-singles[i]-singles[j] for i in range(4) for j in range(i+1,4)}
    quad=np.asarray([add[mask]+sum(v for (i,j),v in pairs.items() if mask&(1<<i) and mask&(1<<j)) for mask in range(16)])
    alpha_indices=[0,16,17,18,15]
    choices=dict(independent=int(np.argmax(add)),quadratic=int(np.argmax(quad)),finite_loss_best=int(np.argmin(l[:16])),
                 per_image_alpha_loss_best=alpha_indices[int(np.argmin(l[alpha_indices]))])
    hold=[7,11,13,14,15]
    return dict(choices=choices,utility=utility.tolist(),singles=singles,pair_interactions={str(i)+'_'+str(j):v for (i,j),v in pairs.items()},
                additive_prediction=add.tolist(),quadratic_prediction=quad.tolist(),
                holdout_additive_MAE=float(np.mean(np.abs(utility[hold]-add[hold]))),
                holdout_quadratic_MAE=float(np.mean(np.abs(utility[hold]-quad[hold]))))

def inference(obs,inp_gt,restored):
    forwards=0
    for arm in ['D0','DA']:
        model,criterion=detector(arm);initial=state_hash(model);allpred=defaultdict(list);image_rows=[]
        for i,m in enumerate(obs):
            assert sha(restored[m['image_id']]['file'])==restored[m['image_id']]['sha256']
            with Image.open(m['q20']) as im:a=np.asarray(im.convert('RGB'))
            with Image.open(restored[m['image_id']]['file']) as im:b=np.asarray(im.convert('RGB'))
            variants=actions(a,b);batch=batch_for(inp_gt,m['image_id']);capture={};fixed=None
            hook=criterion.assigner.register_forward_hook(lambda module,args,out:capture.update(assigned=tuple(t.detach().clone() for t in out)))
            losses=[];fixed_losses=[];parts=[];preds=[]
            for k,pixels in enumerate(variants):
                tensor=prepare_pixels(pixels,m['geometry'])
                with torch.inference_mode():
                    decoded,raw=model(tensor);forwards+=1
                    loss,p=raw_loss(criterion,raw,batch)
                    if k==0:
                        hook.remove();fixed=v8DetectionLoss(model,tal_topk=10);fixed.assigner=FixedAssignment(capture['assigned'])
                    fl,_=raw_loss(fixed,raw,batch)
                    if k==0:assert abs(fl-loss)<1e-6
                    det=nms.non_max_suppression(decoded,conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=500,nc=10,end2end=False)[0]
                pp,_=detection_records(det,m);preds.append(pp);allpred[NAMES[k]].extend(pp)
                losses.append(loss);fixed_losses.append(fl);parts.append(p)
            dynamic=selections(losses);frozen=selections(fixed_losses)
            for name,k in dynamic['choices'].items():allpred[name].extend(preds[k])
            row=dict(image_id=m['image_id'],file_name=m['file_name'],losses=losses,loss_parts=parts,fixed_assignment_losses=fixed_losses,
                     dynamic=dynamic,fixed_assignment=frozen)
            image_rows.append(row)
            with (OUT/(arm+'_image_diagnostics.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
            del tensor,decoded,raw,det,fixed,variants
            if (i+1)%10==0:event('REGIONAL_DETECTOR',arm=arm,images=i+1,detector_forwards=forwards)
        assert state_hash(model)==initial
        for name,pred in allpred.items():json_write(OUT/'predictions'/arm/(name+'.json'),pred)
        json_write(OUT/(arm+'_loss_diagnostics.json'),image_rows)
        del model,criterion;torch.cuda.empty_cache()
    return forwards

def evaluate_one(job):
    arm,name=job
    gt,obs,inp_gt=dataset();pred=json.loads((OUT/'predictions'/arm/(name+'.json')).read_text())
    ev,mi=metrics(inp_gt,pred,500)
    originals=[];mapping={m['image_id']:m for m in obs}
    for p in pred:
        m=mapping[p['image_id']];r=m['geometry']['ratio'];l,t,_,_=m['geometry']['pad_ltrb'];x,y,w,h=p['bbox']
        originals.append(dict(image_id=p['image_id'],category_id=p['category_id'],score=p['score'],bbox=[(x-l)/r,(y-t)/r,w/r,h/r]))
    _,mo=metrics(gt,originals,500)
    assert abs(mi['AP']-mo['AP'])<1e-10
    scores=dict(AP=mo['AP'],AP50=mo['AP50'],AP_small_original=mo['AP_small'],AP_small_input=mi['AP_small'],
                matches=[matches(ev,inp_gt,c,k) for c in [.05,.1,.25,.5] for k in [.5,.75]])
    json_write(OUT/'metrics'/arm/(name+'.json'),scores);return arm,name,scores

def main():
    args=argparse.ArgumentParser();args.add_argument('--contract',action='store_true');a=args.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if a.contract:return contract()
    c=json.loads((OUT/'contract_pass.json').read_text());assert c['status']=='PASS'
    assert c['protocol_sha256']==sha(OUT/'protocol.json') and c['script_sha256']==sha(__file__)
    with (OUT/'one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    json_write(OUT/'frozen.json',dict(protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),pid=os.getpid(),training_updates=0))
    gt,obs,inp_gt=dataset();restored=restore(obs);fw=inference(obs,inp_gt,restored);assert fw==3800
    names=NAMES+['independent','quadratic','finite_loss_best','per_image_alpha_loss_best']
    jobs=[(arm,name) for arm in ['D0','DA'] for name in names];results=defaultdict(dict)
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn')) as pool:
        for count,(arm,name,scores) in enumerate(pool.map(evaluate_one,jobs),1):
            results[arm][name]=scores;event('SCORING',completed=count,total=len(jobs),arm=arm,condition=name)
    previous=json.loads((ROOT/'evaluation/initial_diagnostic_result.json').read_text())
    for arm in ['D0','DA']:assert abs(results[arm]['subset_00']['AP_small_input']-previous[arm]['q20']['AP_small_input_500'])<1e-6
    interpretation={}
    for arm in ['D0','DA']:
        baseline=results[arm]['subset_00']['AP_small_input'];q=results[arm]['quadratic']['AP_small_input'];ind=results[arm]['independent']['AP_small_input']
        global_names=['subset_00','alpha_025','alpha_050','alpha_075','subset_15'];best=max(global_names,key=lambda n:results[arm][n]['AP_small_input']);glob=results[arm][best]['AP_small_input']
        interpretation[arm]=dict(joint_gain_over_JPEG_pp=100*(q-baseline),joint_gain_over_independent_pp=100*(q-ind),
            strongest_fixed_global_alpha_diagnostic=best,joint_gain_over_strongest_fixed_global_pp=100*(q-glob),
            preliminary_potential_screen=100*(q-ind)>=.5 and 100*(q-glob)>=.5)
    json_write(OUT/'completed.json',dict(status='COMPLETED',metrics=results,interpretation=interpretation,
              elapsed_seconds=time.monotonic()-START,detector_forwards=fw,unique_restoration_images=100,new_restorer_forwards=99,
              training_updates=0,all_GT_selections_diagnostic_only=True,P3_validated=False,new_policy_training_authorized_by_this_result=False))
    event('COMPLETED',interpretation=interpretation)
if __name__=='__main__':
    try:main()
    except Exception as e:
        json_write(OUT/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
