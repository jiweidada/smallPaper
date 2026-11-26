"""Bounded frozen-checkpoint dual-head diagnostic. No optimizer or training path."""
from pathlib import Path
from io import StringIO
import ast,collections,contextlib,copy,hashlib,importlib.metadata,itertools,json,math,os,subprocess,sys,time,traceback,zipfile
import cv2,numpy as np,torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.utils import ops,nms,events
# Match the historical evaluator's trainer/validator import surface without executing study.py.
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils.tal import make_anchors
events.enabled=False
R=Path(__file__).resolve().parent
START=time.monotonic();COUNTS={'whole_detector_forwards':0,'standalone_native_TAL_forward_calls':0,'extra_native_head_inference_decodes':0,'all_native_head_inference_decodes':0,'optimizer_updates':0}
CFG=json.loads((R/'native_dual_head_protocol_FROZEN.json').read_text())
ROOT=Path(CFG['remote_root']);CAP=CFG['call_caps'];STAGE='preflight';COMPLETED_IMAGES=0
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def status(extra=None):
    out={'status':'RUNNING','stage':STAGE,'pid':os.getpid(),'completed_images':COMPLETED_IMAGES,'elapsed_seconds':time.monotonic()-START,'counts':dict(COUNTS)}
    if extra:out.update(extra)
    write(R/'status.json',out);return out
def guard():
    assert time.monotonic()-START<=CAP['wall_seconds'],'wall cap exceeded'
def bump(name):
    guard()
    if name in CAP:assert COUNTS[name]<CAP[name],name+' cap reached'
    COUNTS[name]+=1
    if name=='whole_detector_forwards':status()
def state_digest(model):
    h=hashlib.sha256()
    for name,x in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load_helpers():
    path=ROOT/'code/evaluate_fixed.py';tree=ast.parse(path.read_text());wanted=set(CFG['helper_functions']);nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in wanted];assert {x.name for x in nodes}==wanted
    env=globals().copy();exec(compile(ast.Module(body=nodes,type_ignores=[]),'hash_verified_selected_evaluator_helpers','exec'),env)
    return {name:env[name] for name in wanted}
def fidelity(fresh,old):
    f=CFG['fidelity'];floor=f['old_source_prediction_score_floor'];a=[x for x in fresh if x['score']>=floor];b=[x for x in old if x['score']>=floor]
    assert len(a)==len(b),('saved high-score count mismatch',len(a),len(b))
    maxbox=maxscore=0.
    for cat in range(10):
        aa=[x for x in a if x['category_id']==cat];bb=[x for x in b if x['category_id']==cat];assert len(aa)==len(bb),('category count',cat)
        if not aa:continue
        ba=np.asarray([x['bbox'] for x in aa]);be=np.asarray([x['bbox'] for x in bb]);ss=np.asarray([x['score'] for x in aa]);se=np.asarray([x['score'] for x in bb])
        box=np.max(abs(ba[:,None]-be[None]),axis=2);sc=abs(ss[:,None]-se[None]);cost=box/f['box_abs_tolerance_input_pixels']+sc/f['score_abs_tolerance'];ii,jj=linear_sum_assignment(cost)
        eb=float(box[ii,jj].max());es=float(sc[ii,jj].max());assert eb<=f['box_abs_tolerance_input_pixels'] and es<=f['score_abs_tolerance'],('cached fidelity',cat,eb,es)
        maxbox=max(maxbox,eb);maxscore=max(maxscore,es)
    return {'detections_ge_floor':len(a),'max_box_abs_input_px':maxbox,'max_score_abs':maxscore,'pass':True}
def same_fp(ev,gt,budget):
    row=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.5))[0]);small={x['id'] for x in gt['annotations'] if not x['iscrowd'] and x['area']<1024};valid={x['id'] for x in gt['annotations'] if not x['iscrowd']};events=[]
    for e in ev.evalImgs:
        if e is None or e['aRng']!=ev.params.areaRng[0] or e['maxDet']!=500:continue
        assert np.asarray(e['dtIgnore']).dtype==np.bool_
        for j,score in enumerate(e['dtScores']):
            if e['dtIgnore'][row,j]:continue
            match=int(e['dtMatches'][row,j]);events.append((float(score),int(e['image_id']),int(e['category_id']),int(e['dtIds'][j]),match))
    events.sort(key=lambda x:(-x[0],x[1],x[2],x[3]));fp=0;tp=set();threshold=None;included=0
    for score,items in itertools.groupby(events,key=lambda x:x[0]):
        block=list(items);newfp=sum(x[-1]==0 for x in block)
        if fp+newfp>budget:break
        fp+=newfp;tp.update(x[-1] for x in block if x[-1]);included+=len(block);threshold=score
    return {'FP_budget':budget,'actual_FP':fp,'global_score_threshold':threshold,'included_nonignored':included,'TP':len(tp),'small_TP':len(tp&small),'small_FN':len(small-tp),'small_recall':len(tp&small)/len(small),'all_valid_GT':len(valid)}
def main():
    global STAGE,COMPLETED_IMAGES
    assert Path(CFG['remote_output'])==R
    assert not (R/'complete.json').exists() and not (R/'failure.json').exists()
    for name,value in CFG['versions'].items():assert importlib.metadata.version(name)==value,name
    for name,h in CFG['remote_asset_sha256'].items():assert sha(ROOT/name)==h,name
    package=Path(CFG['host_package'])
    for row in CFG['source_files']:assert sha(package/row['path'].split('ultralytics/',1)[1])==row['sha256'],row['path']
    for case in CFG['cases']:assert sha(case['path'])==case['sha256'],case['file_name']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip();procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    assert len(gpu.splitlines())==1 and int(gpu.split(',')[-1])<=CFG['host_requirements']['GPU_max_used_MiB_before_start'];assert not procs,procs
    write(R/'start.json',{'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'protocol_sha256':sha(R/'native_dual_head_protocol_FROZEN.json'),'runner_sha256':sha(__file__),'GPU_before':gpu,'processes_before':procs,'source_and100_image_hashes_verified':True,'torchvision_imported_before_model':'torchvision' in sys.modules})
    helpers=load_helpers();metadata=read(ROOT/'receipts/diagnostic_observations.json')['images'];gt=read(ROOT/'dataset/diagnostic_gt_orig.json');gtinput=helpers['ground_truth_input'](gt,metadata)
    oldpred=read(ROOT/'evaluation/D0/source_predictions_input.json');oldby=collections.defaultdict(list)
    for x in oldpred:oldby[x['image_id']].append(x)
    gtby=collections.defaultdict(list)
    for x in gtinput['annotations']:
        if not x['iscrowd']:gtby[x['image_id']].append(x)
    torch.set_num_threads(4)
    model=YOLO(CFG['checkpoint']['path']).model.cuda().float().eval().requires_grad_(False);head=model.model[-1]
    assert head.nc==10 and head.reg_max==1 and head.cv2 is not None and head.one2one_cv2 is not None
    assert head.stride.tolist()==[8.,16.,32.]
    original_end2end=head.end2end;head.end2end=False;head.export=False;head.max_det=500;head.agnostic_nms=False
    before=state_digest(model);criterion=model.init_criterion();assert criterion.__class__.__name__=='E2ELoss'
    assigners={'M':criterion.one2many.assigner,'O':criterion.one2one.assigner}
    assert (assigners['M'].topk,assigners['M'].topk2)==(10,10) and (assigners['O'].topk,assigners['O'].topk2)==(7,1)
    write(R/'runtime_model_info.json',{'model_class':model.__class__.__name__,'head_class':head.__class__.__name__,'nc':head.nc,'reg_max':head.reg_max,'strides':head.stride.tolist(),'criterion_class':criterion.__class__.__name__,'assigners':{k:{'class':v.__class__.__name__,'topk':v.topk,'topk2':v.topk2,'alpha':v.alpha,'beta':v.beta,'stride_val':v.stride_val} for k,v in assigners.items()},'state_dict_digest_before':before,'parameters':sum(x.numel() for x in model.parameters()),'GPU':torch.cuda.get_device_name(0),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'torchvision_imported':'torchvision' in sys.modules})
    for a in assigners.values():assert a.alpha==.5 and a.beta==6.;a.register_forward_pre_hook(lambda m,args:bump('standalone_native_TAL_forward_calls'))
    model.register_forward_pre_hook(lambda m,args:bump('whole_detector_forwards'))
    original_decode=head._inference
    def counted_decode(raw):bump('all_native_head_inference_decodes');return original_decode(raw)
    head._inference=counted_decode
    outputs={a['id']:[] for a in CFG['arms']};native_original={'M':[],'O':[]};fidelity_rows=[];capturemeta=[];first=None
    (R/'captures').mkdir(exist_ok=False);STAGE='collect'
    torch.cuda.reset_peak_memory_stats()
    for index,meta in enumerate(metadata):
        guard();head.end2end=False
        with Image.open(CFG['cases'][index]['path']) as image:pixels=np.asarray(image.convert('RGB'))
        tensor=helpers['prepare_pixels'](pixels,meta['geometry'])
        with torch.inference_mode():
            decoded,raw=model(tensor)
            assert set(raw)=={'one2many','one2one'}
            parts={'M':raw['one2many'],'O':raw['one2one']}
            assert [[x.shape[-2],x.shape[-1]] for x in parts['M']['feats']]==CFG['host_requirements']['grid_shapes']
            assert all(torch.equal(a,b) for a,b in zip(parts['M']['feats'],parts['O']['feats']))
            for part in parts.values():
                assert part['boxes'].shape==(1,4,8400) and part['scores'].shape==(1,10,8400)
                assert torch.isfinite(part['boxes']).all() and torch.isfinite(part['scores']).all()
            bump('extra_native_head_inference_decodes');again=head._inference(parts['M']);assert torch.equal(decoded,again)
            box={'M':ops.xywh2xyxy(decoded[:,:4,:].transpose(1,2)).contiguous()};prob={'M':decoded[:,4:,:].transpose(1,2).contiguous()}
            head.end2end=True;bump('extra_native_head_inference_decodes');other=head._inference(parts['O']);box['O']=other[:,:4,:].transpose(1,2).contiguous();prob['O']=other[:,4:,:].transpose(1,2).contiguous();head.end2end=False
            if index==0:first={'tensor':tensor.clone(),'raw':{k:{n:v[n].clone() for n in ['boxes','scores']} for k,v in parts.items()}}
            capture={};lineage={}
            anchors,strides=make_anchors(parts['M']['feats'],head.stride,.5);annotations=gtby[meta['image_id']]
            classes=torch.tensor([x['category_id'] for x in annotations],dtype=torch.float32,device='cuda').view(1,-1,1)
            xyxy=torch.tensor([[x['bbox'][0],x['bbox'][1],x['bbox'][0]+x['bbox'][2],x['bbox'][1]+x['bbox'][3]] for x in annotations],dtype=torch.float32,device='cuda').view(1,-1,4)
            mask=torch.ones((1,len(annotations),1),device='cuda',dtype=torch.bool)
            for k in ['M','O']:
                tar=assigners[k](prob[k],box[k],anchors*strides,classes,xyxy,mask)
                for n in ['boxes','scores']:capture[k+'_'+n]=parts[k][n].squeeze(0).cpu().numpy()
                capture[k+'_decoded_xyxy']=box[k].squeeze(0).cpu().numpy();capture[k+'_target_scores']=tar[2].squeeze(0).cpu().numpy();capture[k+'_fg']=tar[3].squeeze(0).cpu().numpy();capture[k+'_target_gt_idx']=tar[4].squeeze(0).cpu().numpy()
            for arm in CFG['arms']:
                b,s,p=arm['boxes'],arm['scores'],arm['postprocess'];name=arm['id']
                if p=='NMS':
                    xywh=ops.xyxy2xywh(box[b]);z=torch.cat([xywh.transpose(1,2),prob[s].transpose(1,2)],dim=1)
                    ds,indices=nms.non_max_suppression(z,conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=500,nc=10,end2end=False,return_idxs=True);det=ds[0];ids=indices[0]
                else:
                    scores,labels,ids=head.get_topk_index(prob[s],500);det=torch.cat([head._gather(box[b],ids),scores,labels],dim=-1)[0]
                    if name=='Obox_Oscore_TOPK' and index==0:first['one2one_topk_before_floor']=det.unsqueeze(0).clone()
                    ids=ids[0];keep=det[:,4]>=.001;det=det[keep];ids=ids[keep]
                inp,orig=helpers['detection_records'](det,meta);outputs[name].extend(inp);lineage[name]=ids.cpu().tolist()
                if name=='Mbox_Mscore_NMS':
                    fidelity_rows.append({'index':index,'image_id':meta['image_id'],**fidelity(inp,oldby[meta['image_id']])});native_original['M'].extend(orig)
                if name=='Obox_Oscore_TOPK':
                    native_original['O'].extend(orig)
            capture['anchors_px']=(anchors*strides).cpu().numpy();capture['strides']=strides.cpu().numpy();capture['GT_xyxy']=xyxy.squeeze(0).cpu().numpy();capture['GT_classes']=classes.view(-1).cpu().numpy()
            if index in CFG['fidelity_gate_cases']:
                for level,x in enumerate(parts['M']['feats']):capture['shared_feature_'+str(level)]=x.cpu().numpy()
            np.savez_compressed(R/'captures'/('case_%03d.npz'%index),**capture)
            capturemeta.append({'index':index,'image_id':meta['image_id'],'file_name':meta['file_name'],'GT_uids':[x['stable_uid'] for x in annotations],'GT_ids':[x['id'] for x in annotations],'lineage':lineage})
        COMPLETED_IMAGES=index+1;status({'first4_fidelity_gate_passed':COMPLETED_IMAGES>=4})
        if COMPLETED_IMAGES in [1,4,20,40,60,80,100]:print(json.dumps({'images':COMPLETED_IMAGES,'counts':COUNTS,'elapsed':time.monotonic()-START}),flush=True)
    STAGE='native_repeat';head.end2end=True
    with torch.inference_mode():
        repeat,raw=model(first['tensor'])
        assert torch.equal(repeat,first['one2one_topk_before_floor']),'native one2one reconstruction mismatch'
        for k,name in [('M','one2many'),('O','one2one')]:
            for n in ['boxes','scores']:assert torch.equal(raw[name][n],first['raw'][k][n]),('raw repeat',k,n)
    head.end2end=original_end2end;assert state_digest(model)==before
    write(R/'capture_metadata.json',capturemeta);write(R/'cached_prediction_fidelity.json',fidelity_rows)
    for name,records in outputs.items():write(R/(name+'_predictions_input.json'),records)
    STAGE='metrics';status();evaluators={};metrics={}
    for name,records in outputs.items():
        guard();ev,point=helpers['coco_eval'](gtinput,copy.deepcopy(records));evaluators[name]=ev
        metrics[name]={'input_area_AP':point,'same_FP':[same_fp(ev,gtinput,b) for b in CFG['metrics']['same_FP_budgets']], 'thresholds':[helpers['match_summary'](ev,gtinput,c,.5) for c in CFG['metrics']['score_thresholds']]}
    old=read(ROOT/'evaluation/D0/source_metrics.json');fresh=metrics['Mbox_Mscore_NMS']['input_area_AP'];tol=CFG['fidelity']['fresh_one2many_vs_cached_AP_abs_tolerance']
    assert abs(fresh['AP_small_500']-old['AP_small_input_500'])<=tol
    assert abs(fresh['AP500']-old['AP500_input_coordinate_crosscheck'])<=tol
    for k,name in [('M','Mbox_Mscore_NMS'),('O','Obox_Oscore_TOPK')]:
        _,original=helpers['coco_eval'](gt,copy.deepcopy(native_original[k]));metrics[name]['original_area_AP']=original;write(R/(name+'_predictions_original.json'),native_original[k])
    nativeM=evaluators['Mbox_Mscore_NMS'];nativeO=evaluators['Obox_Oscore_TOPK'];identity=np.arange(len(metadata));bm=helpers['bootstrap_ap'](nativeM,identity);bo=helpers['bootstrap_ap'](nativeO,identity)
    assert abs(bm-metrics['Mbox_Mscore_NMS']['input_area_AP']['AP_small_500'])<1e-12 and abs(bo-metrics['Obox_Oscore_TOPK']['input_area_AP']['AP_small_500'])<1e-12
    STAGE='bootstrap';status();bc=CFG['metrics']['bootstrap'];rng=np.random.default_rng(bc['seed']);draws=rng.integers(0,len(metadata),size=(bc['draws'],len(metadata)));sample=[]
    for i,draw in enumerate(draws):
        guard();sample.append(helpers['bootstrap_ap'](nativeO,draw)-helpers['bootstrap_ap'](nativeM,draw))
        if (i+1)%200==0:status({'bootstrap_completed':i+1});print(json.dumps({'bootstrap':i+1,'elapsed':time.monotonic()-START}),flush=True)
    assert np.isfinite(sample).all();np.savez_compressed(R/'paired_native_bootstrap.npz',draws=draws,delta=np.asarray(sample))
    for name,h in CFG['remote_asset_sha256'].items():assert sha(ROOT/name)==h,name
    for row in CFG['source_files']:assert sha(package/row['path'].split('ultralytics/',1)[1])==row['sha256']
    assert COUNTS['whole_detector_forwards']==101 and COUNTS['standalone_native_TAL_forward_calls']==200 and COUNTS['extra_native_head_inference_decodes']==200 and COUNTS['all_native_head_inference_decodes']==301
    result={'phase':'A30','status':'DUAL_HEAD_FACTORIAL_DIAGNOSTIC_COMPLETED','counts':COUNTS,'images':100,'metrics':metrics,'native_O_minus_M_input_small_AP':bo-bm,'native_O_minus_M_paired_image_bootstrap95':np.quantile(sample,[.025,.975]).tolist(),'bootstrap_scope':bc,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'state_dict_before_after_digest':before,'weights_source_images_unchanged':True,'fidelity_all100_passed':True,'native_end2end_repeat_bitwise_equal':True,'elapsed_seconds':time.monotonic()-START,'new_algorithm_or_mainline_validated':False,'limitations':['Consumed development images,single frozen checkpoint,internalCOCO ignore rules.','Factorial swaps are diagnostic counterfactuals,not trained methods.','TAL computed on final checkpoint is not historical training supervision.']}
    write(R/'result.json',result);STAGE='archive';status()
    members=[p for p in R.rglob('*') if p.is_file() and p.name not in ['stdout.log','launch.json','one_shot.lock','status.json'] and p.suffix not in ['.zip']]
    manifest={str(p.relative_to(R)):sha(p) for p in sorted(members)};write(R/'evidence_manifest.json',manifest)
    archive=R/'evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in members:z.write(p,arcname=str(p.relative_to(R)))
        z.write(R/'evidence_manifest.json',arcname='evidence_manifest.json')
    complete={'status':'COMPLETED','exit_code':0,'counts':COUNTS,'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'manifest_files':len(manifest),'elapsed_seconds':time.monotonic()-START,'automatic_followups':False}
    write(R/'complete.json',complete);status({'status':'COMPLETED'});print(json.dumps(complete),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        failure={'status':'FAILED_STOPPED_NO_RETRY','exit_code':1,'stage':STAGE,'counts':COUNTS,'completed_images':COMPLETED_IMAGES,'error':repr(error),'traceback':traceback.format_exc(),'automatic_retry':False};write(R/'failure.json',failure);status(failure);traceback.print_exc();sys.exit(1)
