"""Frozen finite raw-annotation audit. No model, training, fitting or inference."""
import json, hashlib, collections, csv, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
E=ROOT/'data/00_实验计划与证据索引'
BASE=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
VAL=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def overlap(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(-1);aa=a[:,2:].prod(-1);bb=b[:,2:].prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-300),inter/np.maximum(aa[:,None],1e-300)
def hashed(seed,row):return hashlib.sha256(f'missingGT_20261027_v1|{seed}|{row}'.encode()).hexdigest()

started=time.perf_counter();protocol=HERE/'02_漏标审计_执行前协议.json'
proto_sha=sha(protocol);(HERE/'02_漏标审计_执行前协议.sha256').write_text(proto_sha,encoding='utf-8')
ds=read(VAL/'annotations/val.json');ims={im['id']:im for im in ds['images']};by=collections.defaultdict(list)
for a in ds['annotations']:by[a['image_id']].append(a)
sources={str(protocol):proto_sha,str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(VAL/'annotations/val.json'):sha(VAL/'annotations/val.json')}
assert sources[str(VAL/'annotations/val.json')]=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
raw={};rawcounts=collections.Counter();rawmatch={}
for iid,im in ims.items():
    p=VAL/'annotations'/Path(im['file_name']).with_suffix('.txt');sources[str(p)]=sha(p)
    rr=np.array([[float(v) for v in ln.rstrip(',').split(',')] for ln in p.read_text(encoding='utf-8-sig').splitlines() if ln.strip()],float).reshape(-1,8)
    raw[iid]=rr
    rawcounts.update({str(int(k)):int((rr[:,5]==k).sum()) for k in set(rr[:,5])})
    valid=rr[(rr[:,4]==1)&(rr[:,5]>=1)&(rr[:,5]<=10)&(rr[:,2]>0)&(rr[:,3]>0)]
    assert len(valid)==len(by[iid]),(iid,len(valid),len(by[iid]))
    # Verify integer raw-box/class mapping is bijective at the known conversion rounding tolerance.
    used=set();delta=0.
    for a in by[iid]:
        choices=np.flatnonzero((valid[:,5].astype(int)==a['category_id']+1)&(np.max(np.abs(valid[:,:4]-a['bbox']),axis=1)<=.02))
        choices=[int(k) for k in choices if int(k) not in used]
        assert choices,(iid,a['id'])
        k=min(choices,key=lambda j:float(np.max(np.abs(valid[j,:4]-a['bbox']))));used.add(k)
        delta=max(delta,float(np.max(np.abs(valid[k,:4]-a['bbox']))))
    assert len(used)==len(valid);rawmatch[iid]={'n':len(valid),'max_delta_px':delta}

pool=[];controls=[];summaries={};all_candidate={}
for seed in [1,2]:
    p=BASE/f'seed{seed}/predictions.json';sources[str(p)]=sha(p);pred=read(p)
    pby=collections.defaultdict(list)
    for row,d in enumerate(pred):pby[d['image_id']].append(row)
    summary=collections.Counter();strata=collections.Counter();controlby=collections.defaultdict(list)
    for iid,im in ims.items():
        inds=pby[iid];inds=sorted(inds,key=lambda k:-pred[k]['score'])
        nc=collections.Counter();keep=[]
        for k in inds:
            c=pred[k]['category_id']
            if nc[c]<100:keep.append(k)
            nc[c]+=1
        summary['exported_rows']+=len(inds);summary['cap100_retained']+=len(keep)
        kk=[k for k in keep if pred[k]['score']>=.25]
        summary['cap_score_ge025']+=len(kk)
        if not kk:continue
        boxes=np.array([pred[k]['bbox'] for k in kk],float)
        I,_=overlap(boxes,[a['bbox'] for a in by[iid]])
        mx=I.max(1,initial=0.)
        r=raw[iid];ig=[]
        for cat in [0,11]:
            _,A=overlap(boxes,r[r[:,5]==cat,:4]);ig.append(A.max(1,initial=0.))
        for j,k in enumerate(kk):
            d=pred[k];c=d['category_id'];score=d['score']
            if seed==1 and score>=.5:
                same=np.array([a['category_id']==c for a in by[iid]],bool)
                sm=float(I[j,same].max(initial=0.))
                if sm>=.5 and iid not in [51,263,428,512]:
                    controlby[c].append({'seed':seed,'prediction_row':k,'image_id':iid,'category_id':c,'score':score,'bbox':d['bbox'],'GT_sameclass_max_IoU':sm,'selection_hash':hashed(seed,k),'review_role':'annotated_control'})
            if mx[j]>=.1:continue
            summary['all_GT_IoU_lt01']+=1
            if ig[0][j]>=.5 or ig[1][j]>=.5:
                summary['low_overlap_in_ignore_or_others']+=1;continue
            summary['eligible_all548']+=1
            band='025_050' if score<.5 else '050_100'
            v={'seed':seed,'prediction_row':k,'image_id':iid,'category_id':c,'score':score,'bbox':d['bbox'],'all_GT_max_IoU':float(mx[j]),'raw0_detection_IoA':float(ig[0][j]),'raw11_detection_IoA':float(ig[1][j]),'score_band':band,'selection_hash':hashed(seed,k),'review_role':'candidate'}
            all_candidate[(seed,k)]=v
            if iid in [51,263,428,512]:
                summary['excluded_anchor_image_occurrences']+=1;continue
            pool.append(v);summary['eligible_available544']+=1;strata[f'{c}|{band}']+=1
    if seed==1:
        for c in range(10):
            vals=sorted(controlby[c],key=lambda x:x['selection_hash'])
            if vals:controls.append(vals[0])
    summaries[str(seed)]={'counts':dict(summary),'strata':dict(strata)}
selected=[]
for seed in [1,2]:
    for c in range(10):
        for band in ['025_050','050_100']:
            vals=sorted((v for v in pool if v['seed']==seed and v['category_id']==c and v['score_band']==band),key=lambda x:x['selection_hash'])
            selected.extend(vals[:2])
allsel=sorted(selected+controls,key=lambda x:hashlib.sha256(f"display|{x['seed']}|{x['prediction_row']}".encode()).hexdigest())
for i,v in enumerate(allsel,1):
    v['case_id']=f'R{i:03d}'
    im=ims[v['image_id']];v['file_name']=im['file_name'];v['image_wh']=[im['width'],im['height']]
    path=VAL/'images'/im['file_name'];sources[str(path)]=sha(path)
save('03_漏标审计_样本与总体.json',{'protocol_sha256':proto_sha,'summaries':summaries,'raw_category_counts':dict(rawcounts),'raw_GT_mapping_all548':{'images':len(rawmatch),'GT':sum(v['n'] for v in rawmatch.values()),'max_abs_px':max(v['max_delta_px'] for v in rawmatch.values())},'candidate_sample_by_seed':dict(collections.Counter(v['seed'] for v in selected)),'candidate_images_by_seed':{str(s):len({v['image_id'] for v in selected if v['seed']==s}) for s in [1,2]},'controls':len(controls),'cases':allsel,'sources':sources,'elapsed_seconds':time.perf_counter()-started})
save('03_漏标审计_完整合格池.json',{'available_pool':pool,'old_anchor_pool':[v for v in all_candidate.values() if v['image_id'] in [51,263,428,512]]})
print(json.dumps({'summaries':summaries,'selected_candidates':len(selected),'controls':len(controls),'images':len({v['image_id'] for v in allsel}),'seconds':time.perf_counter()-started},ensure_ascii=False,indent=2))
