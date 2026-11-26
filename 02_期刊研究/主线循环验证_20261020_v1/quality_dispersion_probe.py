"""Same-forward cache-only exploratory quality association. No detector fitting."""
import datetime, hashlib, json, os, sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'; os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK/'英文小论文/code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import numpy as np
import torch, psutil

def softmax(x,axis=-1):
    a=np.exp(x-x.max(axis=axis,keepdims=True));return a/a.sum(axis=axis,keepdims=True)
def rankdata(x):
    order=np.argsort(x,kind='stable');a=x[order];start=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];end=np.r_[start[1:],len(x)]
    r=np.empty(len(x));r[order]=np.repeat((start+end+1)/2,end-start);return r

OUT=ROOT/'quality_dispersion_probe_v2'
DATA=WORK/'英文小论文/data/08_匹配聚合覆盖探针_20261019_v1/isolated_v3'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    with Path(p).open('x',encoding='utf8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def box(x):return np.concatenate([x[...,:2]-x[...,2:]/2,x[...,:2]+x[...,2:]/2],-1)
def ious(a,b):
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
    union=(a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def corr(a,b):
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else None
def association(features, y, mask):
    f=features[mask];v=y[mask]
    if len(v)<40:return dict(n=len(v),status='NA: fewer than40queries')
    # In-sample nuisance projection is descriptive partial-rank association,
    # not a trained score head, cross-validation, causal effect, or AP estimate.
    ranked=np.apply_along_axis(rankdata,0,f[:,:-1]);classes=f[:,-1].astype(int)
    X=np.column_stack([np.ones(len(f)),ranked[:,1:],np.eye(10)[classes][:,1:]])
    z=ranked[:,0];r=rankdata(v)
    zr=z-X@np.linalg.lstsq(X,z,rcond=1e-10)[0]
    rr=r-X@np.linalg.lstsq(X,r,rcond=1e-10)[0]
    return dict(n=len(v),raw_spearman_U_vs_IoU=corr(z,r),
        partial_rank_U_vs_IoU=corr(zr,rr),nuisance_columns=X.shape[1],
        residual_U_std=float(zr.std()), residual_IoU_std=float(rr.std()))
def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);p.cpu_affinity(p.cpu_affinity()[-1::-2][:2])
    assert torch.version.cuda is None
    # Synthetic scale cancellation and distribution-location ambiguity.
    support=np.array([-4.,-1.,0.,1.,4.]);p1=np.array([.5,0,0,0,.5]);p2=np.array([0,.5,0,.5,0])
    assert np.array_equal(np.sort(p1),np.sort(p2)) and np.dot(p1,support)==np.dot(p2,support)==0
    assert np.dot(p1,support**2)>np.dot(p2,support**2)
    assert abs((2.**2/4.**2)/(3.**2)-(20.**2/4.**2)/(30.**2))<1e-12
    OUT.mkdir(exist_ok=False)
    protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cohort='BASE24 online saved training-mode development00-07, seeds1/2; all300ordinaryqueries; same32 augmentedimages and6629GTappearances per seed; no heldout',
        U='sum over4edges Var(nonuniform bin offset)*ref_dimension^2/(reg_scale^2*final_dimension^2)',
        algebra='Absolute common scale cancels: U is a relative moment statistic, not absolute pixel uncertainty. Identical top-k probabilities with different bin locations need not have identical U, but this is not evidence of detection benefit.',
        target='max IoU to same predicted-class GT in the same image; unmatched class gets0; this is not one-to-one matching or AP',
        controls=['production max class logit','log finalwidth','log finalheight','four edge entropies','20 native LQE inputs: top4probabilities and mean peredge','equal-index-coordinate analogue of U','log refwidth/finalwidth','log refheight/finalheight','predicted class one-hot'],
        analysis='Raw Spearman and descriptive partial-rank association via in-sample linear nuisance projection. No score head is trained, no significance test or AP inference. Fullallqueries,score>=.1,score>=.3 and fixedinputsize<32 subset reported; every cohort, both seeds, no choosing the best.',
        decision='Screen only: if U shows no consistent inverse residual association across the two seeds, do not train on it. Consistent association merely retains a hypothesis; it cannot establish novelty or benefit. No additional numeric gate is inferred after outcomes.',
        sample_dependence='Queries and repeatedaugmentedimageappearances are dependent; no query-level confidence interval or seed-independence claim.',
        source='Each prediction fingerprint checked against co-saved original GPU analysis, no data09/data10 index splicing',
        synthetic_tests=True,new_model_forwards=0,training_steps=0,remote_operations=False,script_sha256=sha(__file__))
    dump(OUT/'01_protocol.json',protocol)
    results=[]; source=[]
    for side,seed in [('A',1),('B',2)]:
        ff=[];yy=[];ss=[];sizes=[]; batches=[]; max_box_error=0.;max_cancel_error=0.
        for b in range(8):
            pp=DATA/side/f'frozen_predictions/development/batch{b:02d}_predictions.pt'
            ap=DATA/side/f'development/batch{b:02d}_analysis.json'
            a=json.loads(ap.read_text(encoding='utf8'));digest=sha(pp);assert digest==a['prediction_sha256']
            source.append(dict(seed=seed,batch=b,prediction_sha256=digest,analysis_sha256=sha(ap)))
            v=torch.load(pp,map_location='cpu',weights_only=False);o=v['outputs']
            logits=o['pred_corners'].numpy().astype(float).reshape(4,300,4,33)
            prob=softmax(logits,axis=-1);reg=abs(float(o['reg_scale']));up=abs(float(o['up']))
            # Match native weighting_function's FP32 arithmetic before copying.
            up_t=o['up'];scale_t=o['reg_scale'];step=(abs(up_t[0])*abs(scale_t)+1)**(2/(32-2))
            w=torch.cat([-abs(up_t[0])*abs(scale_t)*2,*[-step**i+1 for i in range(15,0,-1)],torch.zeros_like(up_t[0][None]),*[step**i-1 for i in range(1,16)],abs(up_t[0])*abs(scale_t)*2],0).numpy().astype(float)
            mean=prob@w;variance=(prob*(w-mean[...,None])**2).sum(-1)
            idx=np.arange(33,dtype=float);im=prob@idx;iv=(prob*(idx-im[...,None])**2).sum(-1)
            boxes=o['pred_boxes'].numpy().astype(float);refs=o['ref_points'].numpy().astype(float)
            assert (boxes[...,2:]>0).all() and (refs[...,2:]>0).all()
            dimensions=boxes[...,[2,3,2,3]];refdim=refs[...,[2,3,2,3]]
            U=(variance*refdim**2/(reg**2*dimensions**2)).sum(-1)
            Ui=(iv*refdim**2/(reg**2*dimensions**2)).sum(-1)
            relative=1+(mean[...,:2]+mean[...,2:])/reg
            max_cancel_error=max(max_cancel_error,float(np.abs(U-(variance/np.tile((reg*relative)**2,(1,1,2))).sum(-1)).max()))
            reconstructed=np.concatenate([refs[...,:2]+(mean[...,2:]-mean[...,:2])*refs[...,2:]/(2*reg),refs[...,2:]*relative],-1)
            max_box_error=max(max_box_error,float(np.abs(reconstructed-boxes).max()))
            assert max_box_error<1e-5
            entropy=-(prob*np.log(np.maximum(prob,1e-300))).sum(-1)
            top=np.sort(prob,axis=-1)[...,-4:][...,::-1];native_stat=np.concatenate([top,top.mean(-1,keepdims=True)],-1).reshape(4,300,20)
            cl=o['pred_logits'].numpy().astype(float); label=cl.argmax(-1);score=cl.max(-1)
            current=[]
            for image in range(4):
                target=v['targets'][image];gt=target['boxes'].numpy().astype(float);gtlabel=target['labels'].numpy()
                matrix=ious(box(boxes[image]),box(gt))
                matrix=np.where(label[image,:,None]==gtlabel[None,:],matrix,0)
                quality=matrix.max(-1) if len(gt) else np.zeros(300)
                feat=np.column_stack([np.log(np.maximum(U[image],1e-12)),score[image],np.log(boxes[image,:,2:]),entropy[image],native_stat[image],np.log(np.maximum(Ui[image],1e-12)),np.log(refs[image,:,2:]/boxes[image,:,2:]),label[image]])
                ff.append(feat);yy.append(quality);ss.append(score[image]);sizes.append(np.sqrt(boxes[image,:,2:].prod(-1))*640)
                current.append(dict(image=image,queries=300,mean_max_sameclass_IoU=float(quality.mean())))
            batches.append(dict(batch=b,images=current))
        f=np.concatenate(ff);y=np.concatenate(yy);score=np.concatenate(ss);size=np.concatenate(sizes)
        groups={'all':np.ones(len(y),dtype=bool),'score_ge_.1':score>=np.log(.1/.9),'score_ge_.3':score>=np.log(.3/.7),'score_ge_.1_size_lt32':(score>=np.log(.1/.9))&(size<32)}
        row=dict(seed=seed,queries=len(y),box_reconstruction_max_abs_error=max_box_error,absolute_scale_cancellation_max_abs_error=max_cancel_error,groups={k:association(f,y,m) for k,m in groups.items()},batches=batches)
        # Also preserve all8 batch descriptive associations, without selecting.
        row['batch_allquery_partial_correlations']=[association(f[b*1200:(b+1)*1200],y[b*1200:(b+1)*1200],np.ones(1200,dtype=bool)) for b in range(8)]
        dump(OUT/f'seed{seed}.json',row);results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['batches','batch_allquery_partial_correlations']}),flush=True)
    dump(OUT/'02_results.json',dict(protocol=protocol,seeds=results,sources=source,novelty_or_detection_gain_established=False))
if __name__=='__main__':
    try: main()
    except BaseException as e:
        import traceback
        if OUT.exists():dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),blind_retry=False))
        raise
