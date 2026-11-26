"""Training-annotation-only, held-out-instance scale-coherence audit. No detector."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,time
import numpy as np
P=Path(__file__).resolve().parent;ENG=P.parents[1]
MAN=P.parent/'A43_ECDet三组蒸馏最小验证_20261031_v1/train_data_manifest.json'
RAW=ENG/'相位机制论文_20261018/修订_20261018_v2/data/annotations/train/raw'
LAM=.05;MIN_N=8
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def load_rows():
    m=json.loads(MAN.read_text(encoding='utf-8-sig'));rows=[]
    for r in m['rows']:
        f=RAW/(Path(r['file']).stem+'.txt');assert sha(f)==r['annotation_sha256'],f
        objects=[]
        for ri,line in enumerate(f.read_text().splitlines()):
            x,y,w,h,score,c,trunc,occ=map(float,line.split(',')[:8]);ww,hh=r['width'],r['height']
            x0,x1=np.clip([x,x+w],0,ww);y0,y1=np.clip([y,y+h],0,hh)
            if score<=0 or not 1<=int(c)<=10 or x1<=x0 or y1<=y0:continue
            wi,hi=(x1-x0)*640/ww,(y1-y0)*640/hh
            objects.append([int(c)-1,(x0+x1)/ww-1,(y0+y1)/hh-1,math.log(math.sqrt(wi*hi)),min(wi,hi),(x1-x0)*(y1-y0),ri])
        assert len(objects)==r['valid_gt']
        rows.append({'file':r['file'],'calibration':stable(r['file'])%2==0,'a':np.asarray(objects,dtype=np.float64).reshape(-1,7)})
    assert len(rows)==6471 and sum(len(r['a']) for r in rows)==343204
    return rows
def summarize(rows,mask):
    a=[x for x in rows if mask(x)]
    if not a:return {'targets':0}
    im={}
    for x in a:
        t=im.setdefault(x['file'],[[],[],[]])
        for k,name in enumerate(['global','spatial','shuffled']):t[k].append(x['error_'+name])
    md=np.asarray([[np.mean(v) for v in q] for q in im.values()])
    gm=float(np.mean([x['error_global'] for x in a]));sp=float(np.mean([x['error_spatial'] for x in a]));sh=float(np.mean([x['error_shuffled'] for x in a]))
    return {'targets':len(a),'images':len(im),'mean_absolute_log_size_error':{'global':gm,'spatial':sp,'shuffled':sh},
        'spatial_minus_global':sp-gm,'spatial_relative_error_reduction':1-sp/gm,
        'spatial_minus_shuffled':sp-sh,'image_mean_spatial_minus_global':float(np.mean(md[:,1]-md[:,0])),
        'fraction_images_spatial_better_than_global':float(np.mean(md[:,1]<md[:,0]))}
def main():
    out=P/'02_场景尺度统计核查_v1.json';assert not out.exists(),'Do not silently rerun or overwrite results'
    start=time.monotonic();rows=load_rows();cal=[r for r in rows if r['calibration']];audit=[r for r in rows if not r['calibration']]
    # Eliminate per-image intercepts to estimate cross-class size offsets without assuming a known GSD.
    gram=np.zeros((10,10));rhs=np.zeros(10)
    for r in cal:
        a=r['a'];n=len(a)
        if not n:continue
        X=np.eye(10)[a[:,0].astype(int)];X-=X.mean(0);z=a[:,3]-a[:,3].mean()
        gram+=X.T@X/n;rhs+=X.T@z/n
    mu=np.linalg.solve(gram+np.eye(10)*1e-6,rhs);mu-=mu.mean()
    scored=[];skips=Counter()
    for r in audit:
        a=r['a'];n=len(a)
        if n<MIN_N:skips['fewer_than_8_objects']+=1;continue
        rng=np.random.default_rng(stable(r['file']+':A54-folds'))
        order=rng.permutation(n);fold=np.zeros(n,dtype=int);fold[order[n//2:]]=1
        C=a[:,0].astype(int);residual=a[:,3]-mu[C]
        X=np.column_stack([np.ones(n),a[:,1:3]])
        for k in [0,1]:
            support=np.flatnonzero(fold!=k);targets=np.flatnonzero(fold==k)
            A=X[support];z=residual[support];q=len(support)
            beta=np.linalg.solve(A.T@A/q+np.diag([0,LAM,LAM]),A.T@z/q)
            # A fixed permutation breaks location/size correspondence, preserving the support's size/class set.
            S=A[rng.permutation(q)];bs=np.linalg.solve(S.T@S/q+np.diag([0,LAM,LAM]),S.T@z/q)
            for j in targets:
                pred={'global':float(z.mean()+mu[C[j]]),'spatial':float(X[j]@beta+mu[C[j]]),'shuffled':float(X[j]@bs+mu[C[j]])}
                x={'file':r['file'],'annotation_row':int(a[j,6])+1,'class0':int(C[j]),'fold':k,'support_objects':q,
                    'short_side_640':float(a[j,4]),'original_area':float(a[j,5]),'GT_log_geomean_640':float(a[j,3])}
                for name,p in pred.items():x['pred_'+name]=p;x['error_'+name]=abs(p-a[j,3])
                scored.append(x)
    result={'status':'COMPLETE_ANNOTATION_ONLY_DESCRIPTIVE_AUDIT','purpose':'Test whether a spatial scene-size field is statistically plausible, not whether ECDet lacks it or an algorithm improves AP.',
        'protocol_fixed_before_execution':str(P/'audit_scope_FROZEN_v1.json'),'train_manifest_sha256':sha(MAN),'script_sha256':sha(Path(__file__)),
        'train_annotation_SHA_matches':6471,'valid_training_GT':343204,'validation_data_used':False,
        'calibration_images':len(cal),'audit_images':len(audit),'audit_image_skips':dict(skips),'canonical_class_log_offsets':mu.tolist(),
        'fixed_ridge_slope_penalty':LAM,'minimum_objects_per_audit_image':MIN_N,'folds':'stable image-hash RNG, two disjoint object halves; each target excluded from support',
        'all':summarize(scored,lambda x:True),'original_small':summarize(scored,lambda x:x['original_area']<=1024),
        'original_small_shortside_4to8':summarize(scored,lambda x:x['original_area']<=1024 and 4<=x['short_side_640']<8),
        'per_class':{str(c):summarize(scored,lambda x,c=c:x['class0']==c) for c in range(10)},
        'scope_limits':['Support uses GT box sizes and classes: privileged statistical evidence, not deployable predictions.',
            'Objects are held out, not scenes or a formal paper test set. All data are from train6471.',
            'Class offsets are statistical size effects, not calibrated physical object sizes or measured GSD.',
            'A linear field is a plausibility probe; it is not the selected detection module or a causal diagnosis.',
            'Only native stretch640 and horizontal flip are relevant to the current pilot; no Mosaic/extra augmentation is assumed.'],
        'new_detector_forwards':0,'new_detector_optimizer_updates':0,'new_COCO_evaluate_calls':0,'seconds':time.monotonic()-start}
    with (P/'02_逐目标统计记录_v1.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scored[0]));writer.writeheader();writer.writerows(scored)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','calibration_images','audit_images','all','original_small_shortside_4to8','seconds']},ensure_ascii=False))
if __name__=='__main__':main()
