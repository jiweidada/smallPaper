"""355b frozen mean/zero Haar interventions. CPU only; all writes under OUT."""
import os,sys
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='SEQUENTIAL')
import json,time,hashlib,importlib.util,contextlib,io,gc,traceback,platform
from pathlib import Path
OUT=Path(__file__).resolve().parent
EN=OUT.parent.parent
OLD=EN/'code/c_axis_precheck_20261025_v1'
SRC=EN/'code/haar_detail_review_20261025_v1'
TRAIN=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train')
spec=importlib.util.spec_from_file_location('legacy352',OLD/'352_precheck.py')
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
np,torch=L.np,L.torch
COUNTS={'model_loads':0,'full_model_batch_calls':0,'full_model_image_forwards':0,'calibration_backbone_batch_calls':0,'calibration_backbone_image_forwards':0,'COCO_evaluations':0,'optimizer_steps':0,'training_started':False}
H=torch.tensor([[1,1,1,1],[-1,1,-1,1],[-1,-1,1,1],[1,-1,-1,1]],dtype=torch.float64)/2
ARMS=['F','D0_mean','Dx_mean','Dy_mean','Dd_mean','OFF_mean','D0_zero','Dx_zero','Dy_zero','Dd_zero','OFF_zero']
PRIMARY=ARMS[:6]
MAP={'D0':0,'Dx':1,'Dy':2,'Dd':3}
START=time.monotonic()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return L.sha(p)
def put(name,obj):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def txt(name,s):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:f.write(s)
def event(kind,**kw):
    row={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'event':kind,**kw}
    with (OUT/'355b_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
def timecheck():
    if time.monotonic()-START>5400:raise TimeoutError('90 minute frozen hard timeout')
L.OUT=OUT
L.put=lambda name,obj:put(name.replace('352_','355b_',1),obj)
L.textfile=lambda name,s:txt(name.replace('352_','355b_',1),s)
L.event=event

def prepare():
    assert not (OUT/'355b_input_manifest.json').exists()
    source_receipt=read(SRC/'355_delivery_SHA256.json')
    for r in source_receipt['files']:assert sha(SRC/r['relative'])==r['sha256'],r['relative']
    oldfiles=read(OLD/'352_input_manifest.json')['files']
    for r in oldfiles:assert sha(r['path'])==r['sha256'],('changed352input',r['path'])
    priorfiles=read(OLD/'352_delivery_SHA256.json')['files']
    selected=['352_c_inference_receipt.json','352_c_predictions.json','352_c_metrics.json','352_cohorts.json','352_image_manifest.json']
    files={Path(r['path']) for r in oldfiles}
    for rel in selected:
        p=OLD/rel
        r=next(x for x in priorfiles if x['path']==rel)
        assert sha(p)==r['sha256'],rel
        files.add(p)
    for p in sorted((OLD/'352_c_batch_cache').glob('*.pt')):
        rel=p.relative_to(OLD).as_posix()
        r=next(x for x in priorfiles if x['path'].replace('\\','/')==rel)
        assert sha(p)==r['sha256'],rel
        files.add(p)
    assert len(list((OLD/'352_c_batch_cache').glob('*.pt')))==137
    data=read(TRAIN/'annotations/train.json');valims=read(OLD/'352_image_manifest.json')
    ordered=sorted(data['images'],key=lambda x:int(x['id']))
    assert len(ordered)==6471 and len(valims)==548
    rng=np.random.default_rng(20261025)
    selected=sorted([ordered[int(i)] for i in rng.choice(len(ordered),256,replace=False)],key=lambda x:int(x['id']))
    valnames={Path(r['path']).name for r in valims};valhash={r['sha256'] for r in valims}
    samples=[]
    for r in selected:
        p=TRAIN/'images'/r['file_name'];digest=sha(p)
        assert r['file_name'] not in valnames and digest not in valhash,('train_val_overlap',r)
        samples.append({'split':'train','image_id':int(r['id']),'file_name':r['file_name'],'path':str(p),'width':r['width'],'height':r['height'],'sha256':digest})
    put('355b_calibration_images_frozen.json',samples)
    files.update([Path(__file__),OUT/'355b_protocol_frozen.json',OUT/'355b_执行计划_冻结.txt',OUT/'355b_calibration_images_frozen.json',SRC/'355b_主控对修订预检草案的补充.txt',SRC/'355_修订预检草案.txt',TRAIN/'annotations/train.json'])
    records=[]
    for p in sorted(files,key=str):records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    # Both selected train images and all validation images bound before any forward.
    for r in valims:assert sha(r['path'])==r['sha256'],r['image_id']
    put('355b_input_manifest.json',{'frozen_before_new_forward':True,'files':records,'calibration_images':samples,'validation_images':valims,'detector_forwards_before_freeze':0,'source352_inputs_verified':True,'source355_inputs_verified':True,'train_val_overlap_by_name_and_SHA':False})
    event('prepared',calibration_images=256,val_images=548,arms=ARMS,manifest_sha256=sha(OUT/'355b_input_manifest.json'),forward_calls=0)

def verify():
    m=read(OUT/'355b_input_manifest.json');n=0
    for r in m['files']+m['calibration_images']+m['validation_images']:
        assert sha(r['path'])==r['sha256'],('changed_input',r['path']);n+=1
    return n

def algebra(W):
    W=W.detach().double().reshape(256,16,4)
    Q=torch.einsum('ocp,kp->ock',W,H)
    gen=torch.Generator().manual_seed(20261025)
    x=torch.randn(2,16,4,3,5,dtype=torch.float64,generator=gen)
    mu=torch.randn(16,4,dtype=torch.float64,generator=gen)
    q=torch.einsum('bcphw,kp->bckhw',x,H)
    result={'orthogonality_max_abs':float((H@H.T-torch.eye(4,dtype=torch.float64)).abs().max()),'Haar_roundtrip_max_abs':float((torch.einsum('bckhw,kp->bcphw',q,H)-x).abs().max()),'arms':{}}
    for arm in ARMS[1:]:
        base,mode=arm.split('_');removed=torch.zeros(4,dtype=torch.float64)
        if base=='OFF':removed[:]=1
        else:removed[MAP[base]]=1
        mask=1-removed
        wp=torch.einsum('ock,k,kp->ocp',Q,mask,H)
        bias=torch.einsum('ock,ck,k->o',Q,mu,removed) if mode=='mean' else torch.zeros(256,dtype=torch.float64)
        reference=torch.einsum('ock,bckhw->bohw',Q,q*mask[None,None,:,None,None]+(mu[None,:,:,None,None]*removed[None,None,:,None,None] if mode=='mean' else 0))
        efficient=torch.einsum('ocp,bcphw->bohw',wp,x)+bias[None,:,None,None]
        err=float((reference-efficient).abs().max())
        idem=torch.einsum('ocp,kp->ock',wp,H)*mask[None,None,:]
        idem=torch.einsum('ock,kp->ocp',idem,H)
        result['arms'][arm]={'direct_vs_projection_max_abs':err,'projection_idempotence_max_abs':float((idem-wp).abs().max()),'DC_zero_sum_max_abs':float(wp.sum(-1).abs().max()) if base=='D0' else None}
        assert err<1e-12 and float((idem-wp).abs().max())<1e-12
        if base=='D0':assert float(wp.sum(-1).abs().max())<1e-12
    assert result['orthogonality_max_abs']==0 and result['Haar_roundtrip_max_abs']<1e-12
    put('355b_algebra_validation.json',result)
    return Q

def make_batch(records):
    from PIL import Image
    from torchvision.transforms import functional as TF
    tensors=[]
    for r in records:
        with Image.open(r['path']) as im:
            assert im.size==(r['width'],r['height'])
            tensors.append(TF.to_tensor(TF.resize(im.convert('RGB'),[640,640])))
    return torch.stack(tensors),torch.tensor([[r['width'],r['height']] for r in records])

class Moments:
    def __init__(self,Q,full_cov=False):
        self.Q=Q;self.n=0;self.sum=torch.zeros(64,dtype=torch.float64);self.sq=torch.zeros(64,dtype=torch.float64)
        self.cov=torch.zeros(64,64,dtype=torch.float64) if full_cov else None
        self.forward_calls=0;self.max_reconstruction_error_sampled_sites=0.
    def hook(self,mod,args,output):
        x=args[0].detach().double();b,c,h,w=x.shape
        assert c==64 and (h,w)==(80,80)
        q=torch.einsum('bcphw,kp->bckhw',x.reshape(b,16,4,h,w),H)
        a=q.permute(1,2,0,3,4).reshape(64,-1)
        self.n+=a.shape[1];self.sum+=a.sum(1);self.sq+=(a*a).sum(1)
        if self.cov is not None:self.cov+=a@a.T
        # Small fixed site sample checks exact decomposition against the real restore output.
        ref=torch.einsum('ock,bckn->bon',self.Q,q.flatten(-2)[...,::101])
        observed=output.detach().double().flatten(-2)[...,::101]
        self.max_reconstruction_error_sampled_sites=max(self.max_reconstruction_error_sampled_sites,float((ref-observed).abs().max()))
        self.forward_calls+=1
    def summary(self):
        mean=(self.sum/self.n).reshape(16,4);sq=(self.sq/self.n).reshape(16,4)
        result={'spatial_sites':self.n,'hook_calls':self.forward_calls,'mu_channel_by_band':mean.tolist(),'std_channel_by_band':torch.clamp(sq-mean*mean,min=0).sqrt().tolist(),'band_mean_of_channel_means':mean.mean(0).tolist(),'band_mean_abs_channel_mean':mean.abs().mean(0).tolist(),'restore_output_mean':torch.einsum('ock,ck->o',self.Q,mean).tolist(),'restore_reconstruction_max_abs_sampled_sites':self.max_reconstruction_error_sampled_sites}
        if self.cov is not None:
            c=self.cov.reshape(16,4,16,4)/self.n;cross=torch.zeros(4,4,dtype=torch.float64)
            for k in range(4):
                for l in range(4):cross[k,l]=torch.einsum('oc,od,cd->',self.Q[:,:,k],self.Q[:,:,l],c[:,k,:,l])
            reconstructed=float(cross.sum());orig=torch.einsum('ock,ckdl,odl->',self.Q,c,self.Q)
            result.update(output_band_uncentered_cross_moments=cross.tolist(),diagonal_energy_sum=float(cross.diag().sum()),six_cross_terms_sum_times2=float(cross.sum()-cross.diag().sum()),residual_output_squared_norm_expectation=reconstructed,quadratic_reconstruction_abs_error=abs(reconstructed-float(orig)))
        return result,mean

def calibrate(model,Q):
    records=read(OUT/'355b_calibration_images_frozen.json');moment=Moments(Q)
    handle=model.backbone.p2_skip.restore.register_forward_hook(moment.hook)
    initial=L.state_sha(model);seconds=0.
    try:
        for start in range(0,len(records),4):
            timecheck();batch=records[start:start+4];x,_=make_batch(batch)
            with torch.inference_mode():
                tick=time.perf_counter();outputs=model.backbone(x);elapsed=time.perf_counter()-tick
            seconds+=elapsed;COUNTS['calibration_backbone_batch_calls']+=1;COUNTS['calibration_backbone_image_forwards']+=len(batch)
            event('calibration_progress',images=start+len(batch),total=256,seconds=elapsed) if start%64==0 or start+len(batch)==256 else None
            del x,outputs
    finally:handle.remove()
    assert L.state_sha(model)==initial
    info,mu=moment.summary();info.update(source='train256 only; eval mode; per-image and spatial averaging',detector_updates=0,backbone_only=True,seconds=seconds,calibration_images_manifest_sha256=sha(OUT/'355b_calibration_images_frozen.json'))
    with (OUT/'355b_training_means.npz').open('xb') as f:np.savez(f,mu=mu.numpy(),H=H.numpy(),band_names=np.array(['DC','Hx','Hy','HH']))
    info['training_means_sha256']=sha(OUT/'355b_training_means.npz');put('355b_calibration_statistics.json',info)
    event('calibration_complete',images=256,seconds=seconds,band_mean=info['band_mean_of_channel_means'],mu_sha256=info['training_means_sha256'])
    return mu

def arm_parameters(arm,W,Q,mu):
    base,mode=arm.split('_');removed=torch.zeros(4,dtype=torch.float64)
    if base=='OFF':removed[:]=1
    else:removed[MAP[base]]=1
    wp=torch.einsum('ock,k,kp->ocp',Q,1-removed,H).reshape_as(W).float()
    bias=torch.einsum('ock,ck,k->o',Q,mu,removed).float() if mode=='mean' else torch.zeros(256)
    return wp,bias

def infer(model,post,arm,Q,W,mu):
    records=read(OLD/'352_image_manifest.json');cache=OUT/f'355b_{arm}_batch_cache';cache.mkdir()
    module=model.backbone.p2_skip.restore;handle=None;moment=None
    if arm=='F':
        assert torch.equal(module.weight,W)
        moment=Moments(Q,full_cov=True);handle=module.register_forward_hook(moment.hook)
    else:
        wp,bias=arm_parameters(arm,W,Q,mu)
        with torch.no_grad():module.weight.copy_(wp)
        if arm.endswith('_mean'):
            handle=module.register_forward_hook(lambda mod,args,out:out+bias[None,:,None,None])
        put(f'355b_{arm}_intervention.json',{'mean_replacement':arm.endswith('_mean'),'restore_bias':bias.tolist(),'restore_bias_L2':float(bias.norm()),'restore_weight_tensor_sha256':hashlib.sha256(module.weight.detach().numpy().tobytes()).hexdigest(),'training_mu_sha256':sha(OUT/'355b_training_means.npz')})
    expected=L.state_sha(model);rows=[];seconds=0.;maxdiff={'pred_logits':0.,'pred_boxes':0.}
    try:
        for start in range(0,len(records),4):
            timecheck();batch=records[start:start+4];x,sizes=make_batch(batch)
            with torch.inference_mode():
                tick=time.perf_counter();outputs=model(x);elapsed=time.perf_counter()-tick;seconds+=elapsed
                COUNTS['full_model_batch_calls']+=1;COUNTS['full_model_image_forwards']+=len(batch)
                record={'image_ids':[r['image_id'] for r in batch],'pred_logits':outputs['pred_logits'].cpu(),'pred_boxes':outputs['pred_boxes'].cpu(),'forward_seconds':elapsed}
                with (cache/f'batch_{start//4:03d}.pt').open('xb') as f:torch.save(record,f)
                assert all(torch.isfinite(record[k]).all() for k in maxdiff),('nonfinite_output',arm,start)
                if arm=='F':
                    reference=torch.load(OLD/f'352_c_batch_cache/batch_{start//4:03d}.pt',map_location='cpu',weights_only=False)
                    assert reference['image_ids']==record['image_ids']
                    for key in maxdiff:maxdiff[key]=max(maxdiff[key],float((record[key]-reference[key]).abs().max()))
                    assert max(maxdiff.values())<=1e-6,('F raw reproduction failed',start,maxdiff)
                results=post(outputs,sizes,for_eval=True,resize_pad=None)
                for rec,r in zip(batch,results):
                    boxes=r['boxes'].clone();boxes[:,2:]-=boxes[:,:2]
                    for box,score,category in zip(boxes.tolist(),r['scores'].tolist(),r['labels'].tolist()):
                        rows.append({'image_id':rec['image_id'],'category_id':int(category),'bbox':box,'score':score})
            event('arm_progress',arm=arm,images=start+len(batch),total=548,forward_seconds=seconds) if start%80==0 or start+len(batch)==548 else None
            del outputs,x
    finally:
        if handle is not None:handle.remove()
        assert L.state_sha(model)==expected,('state mutated during eval',arm)
        with torch.no_grad():module.weight.copy_(W)
    put(f'355b_{arm}_predictions.json',rows)
    validation_error=None
    try:L.validate_predictions(rows,[r['image_id'] for r in records])
    except AssertionError as exc:validation_error=repr(exc)
    info={'arm':arm,'valid_predictions':validation_error is None,'validation_error':validation_error,'images':548,'full_model_batch_calls':137,'forward_seconds':seconds,'gradient_all_none':all(p.grad is None for p in model.parameters()),'no_model_state_mutation_except_deliberate_projection':True,'original_weights_restored':torch.equal(module.weight,W),'optimizer_steps':0}
    if arm=='F':
        info['raw_max_abs_diff_vs352']=maxdiff
        ms,_=moment.summary();put('355b_F_activation_statistics.json',ms)
        oldrows=read(OLD/'352_c_predictions.json')
        assert len(oldrows)==len(rows)
        info['exported_bbox_max_abs_diff_vs352']=max(max(abs(a-b) for a,b in zip(r['bbox'],s['bbox'])) for r,s in zip(rows,oldrows))
        info['exported_score_max_abs_diff_vs352']=max(abs(r['score']-s['score']) for r,s in zip(rows,oldrows))
        assert all((r['image_id'],r['category_id'])==(s['image_id'],s['category_id']) for r,s in zip(rows,oldrows))
    put(f'355b_{arm}_inference_receipt.json',info)
    event('arm_inference_complete',arm=arm,valid=info['valid_predictions'],seconds=seconds)
    return rows,info

def compare_tp(F,M,ids):
    n=len(ids);f=sum(F[i]['matched'] for i in ids);m=sum(M[i]['matched'] for i in ids)
    rescued=[i for i in ids if M[i]['matched'] and not F[i]['matched']];lost=[i for i in ids if F[i]['matched'] and not M[i]['matched']]
    assert m-f==len(rescued)-len(lost)
    return {'GT':n,'F_TP':f,'arm_TP':m,'net_TP':m-f,'rescued':len(rescued),'lost':len(lost),'rescued_gt_ids':rescued,'lost_gt_ids':lost,'net_recall_delta_pp':100*(m-f)/n,'net_TP_drop_fraction':(f-m)/f if f else None,'rescue_fraction_of_F_misses':len(rescued)/(n-f) if n>f else None,'gross_lost_fraction_of_F_TP':len(lost)/f if f else None}

def decide(stats,cohort_results,valid):
    if not all(valid.get(a,False) for a in PRIMARY):return {'decision':'invalid','all_primary_arms_valid':False}
    losses={a:{k:100*(stats['F'][k]-stats[a][k]) for k in L.NAMES} for a in ARMS if a in stats}
    a_components={'AP_loss_le_0_1pp':losses['D0_mean']['AP']<=.1,'APs_loss_le_0_1pp':losses['D0_mean']['APs']<=.1,'S_net_TP_drop_le_1pct':cohort_results['D0_mean']['0.5']['S']['net_TP_drop_fraction']<=.01}
    b_components={a:losses['D0_mean']['APs']<losses[a]['APs'] for a in ['Dx_mean','Dy_mean','Dd_mean']}
    gates={'a_prunable':all(a_components.values()),'b_direction':all(b_components.values()),'c_branch_active':losses['OFF_mean']['APs']>.3}
    decision='insufficient' if not gates['c_branch_active'] else ('pass' if all(gates.values()) else 'mixed')
    return {'decision':decision,'all_primary_arms_valid':True,'gates':gates,'a_components':a_components,'b_components':b_components,'losses_pp':losses,'S_TP_net_drop_fraction_D0':cohort_results['D0_mean']['0.5']['S']['net_TP_drop_fraction'],'OFF_mean_APs_loss_pp':losses['OFF_mean']['APs']}

def seal(result):
    changed=[];verified=0
    try:verified=verify()
    except Exception as exc:changed.append(repr(exc))
    if changed:result.update(decision='input_changed_hard_stop',input_verify_errors=changed)
    result.update(actual_execution_counts=COUNTS,elapsed_seconds=time.monotonic()-START,detector_training_started=False,remote_or_GPU_access=False,final_input_hash_verification={'verified_records':verified,'passed':not changed})
    put('355b_result.json',result);put('355b_execution_counts.json',COUNTS)
    lines=['355b｜均值替换预检实际结果｜客户端日期2026-10-25','Material Passport: run; experiment_id=355b_mean_replacement_precheck_v1; local raw data; source=355+355b; detector optimizer steps=0.','结论：'+result['decision'],'']
    if 'stats' in result:
        lines+=['各臂指标原始小数（AP/APs单位为0—1；loss_pp=100*(F-arm)）：','arm\tAP\tAPs\tAP_loss_pp\tAPs_loss_pp\tS_TP\tS净下降%']
        for a in ARMS:
            if a in result['stats']:
                s=result['stats'][a];loss=result.get('losses_pp',{}).get(a,{})
                ss=result.get('cohort_results',{}).get(a,{}).get('0.5',{}).get('S',{})
                row=[a,format(s['AP'],'.12f'),format(s['APs'],'.12f'),format(loss.get('AP',0.),'+.9f'),format(loss.get('APs',0.),'+.9f'),str(ss.get('arm_TP','N/A')),format(100*ss.get('net_TP_drop_fraction',0.),'+.9f')]
                lines.append('\t'.join(row))
            else:lines.append(a+'\tN/A（未执行或无效；见逐臂回执）')
        lines+=['','全部12项指标：']
        for a,s in result['stats'].items():lines.append(a+': '+json.dumps(s,ensure_ascii=False))
    for name in ['reproduction','gates','a_components','b_components','S_TP_net_drop_fraction_D0','OFF_mean_APs_loss_pp','error']:
        if name in result:lines+=['',name+': '+json.dumps(result[name],ensure_ascii=False)]
    lines+=['','实际计算次数：'+json.dumps(COUNTS,ensure_ascii=False),'旧输入终检：'+json.dumps(result['final_input_hash_verification'],ensure_ascii=False),
            '均值来自256张固定train图，未用val估计或重估；零置换仅描述。','说明：均值替换保留估计的通道无条件均值，仍改变方差、相关性和空间结构；不证明独有信息、重新训练效能或创新性。含均值补偿常量与无偏置detail-only模型不是完全同一个函数。','VisDrone未pass就不跑UAVDT。即使全部预检通过，也不启动重训。']
    txt('355b_结果与判读.txt','\n'.join(lines)+'\n')
    event('sealed_result',decision=result['decision'],counts=COUNTS)
    files=[{'relative':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='355b_delivery_SHA256.json']
    put('355b_delivery_SHA256.json',{'files':files,'manifest_excludes_itself':True,'source_and_weight_files_unchanged':not changed})

def run():
    assert not (OUT/'355b_result.json').exists()
    verify();event('started',python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device='cpu')
    model,post,load_receipt=L.load_model('c');COUNTS['model_loads']+=1
    put('355b_model_load_receipt.json',load_receipt)
    assert load_receipt['state_sha256_before']==read(OLD/'352_c_inference_receipt.json')['state_sha256_before']
    W=model.backbone.p2_skip.restore.weight.detach().clone();Q=algebra(W)
    data=read(L.VAL/'annotations/val.json');stats={};matches={};valid={};cohort_results={};reproduction={}
    cohorts=read(OLD/'352_cohorts.json');mu=None
    for arm in ARMS:
        rows,receipt=infer(model,post,arm,Q,W,mu);valid[arm]=receipt['valid_predictions']
        if not valid[arm]:
            event('invalid_arm',arm=arm,error=receipt['validation_error'])
            if arm=='F':
                seal({'decision':'F_invalid_hard_stop','error':receipt['validation_error'],'stats':stats,'valid':valid});return
            continue
        s,m=L.evaluate(arm,rows,data);COUNTS['COCO_evaluations']+=1
        stats[arm]=s;matches[arm]=m;del rows;gc.collect()
        if arm=='F':
            ref=read(OLD/'352_c_metrics.json')['metrics'];details={k:{'actual':s[k],'reference':ref[k],'difference':s[k]-ref[k]} for k in L.NAMES}
            reproduction={'raw_max_abs':receipt['raw_max_abs_diff_vs352'],'metrics':details,'passed':max(abs(r['difference']) for r in details.values())<=1e-6}
            put('355b_F_reproduction.json',reproduction)
            if not reproduction['passed']:
                seal({'decision':'reproduction_failed','stats':stats,'reproduction':reproduction});return
            event('F_reproduction_passed',raw_max_abs=reproduction['raw_max_abs'],metric_max_abs=max(abs(r['difference']) for r in details.values()))
            mu=calibrate(model,Q)
        summary={}
        for t in [.5,.75]:summary[str(t)]={g:compare_tp(matches['F'][t],m[t],cohorts[g+'_ids']) for g in ['T','S']}
        cohort_results[arm]=summary;put(f'355b_{arm}_cohort_comparison.json',summary)
        event('arm_complete',arm=arm,AP=s['AP'],APs=s['APs'],S_TP=summary['0.5']['S']['arm_TP'])
    assert L.state_sha(model)==load_receipt['state_sha256_before']
    assert all(p.grad is None for p in model.parameters()) and not any(m.training for m in model.modules())
    decision=decide(stats,cohort_results,valid)
    decision.update(stats=stats,cohort_results=cohort_results,valid=valid,reproduction=reproduction,post_run_original_model_state_unchanged=True,UAVDT_status='not_run: VisDrone screen did not pass' if decision['decision']!='pass' else 'VisDrone passed; freeze UAVDT endpoint definitions before six-arm two-seed continuation',training_status='not_started_under_any_screen_result')
    seal(decision)

if __name__=='__main__':
    try:
        if sys.argv[1:] == ['--prepare']:prepare()
        elif sys.argv[1:]==['--run']:run()
        else:raise SystemExit('use --prepare then --run')
    except Exception as exc:
        traceback.print_exc()
        if (OUT/'355b_input_manifest.json').exists() and not (OUT/'355b_result.json').exists():seal({'decision':'hard_check_failed','error':repr(exc),'traceback':traceback.format_exc()})
        raise
