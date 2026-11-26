"""Independent scalar matching, backup coverage and global-prefix audit."""
from pathlib import Path
import ast,json,math,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;E=D/'downloaded_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
cfg=read(D/'native_prefix_protocol_FROZEN.json');out=read(D/'query_retention_analysis_result.json')
tree=ast.parse((D/'pinned_o27_metrics.py').read_text(encoding='utf-8'))
names=['matlab_round','ignore_integral','region_keep','prepare_cases']
ns={'np':np};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'same_source_aligned_ignore_policy','exec'),ns)
cases=ns['prepare_cases'](read(D/'raw_annotations_verified.json')['cases'])
raw=read(E/'raw_native_predictions_without_region_filter.json')
def iou(a,b):
    iw=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
    ih=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    inter=iw*ih;union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.
records=[];events=[];pairs=0
for image,(case,p) in enumerate(zip(cases,raw)):
    keep=ns['region_keep'](p['boxes'],case['integral'])
    dets=[(box,int(label),float(score),int(qid)) for k,(box,label,score,qid) in enumerate(zip(p['boxes'],p['labels'],p['scores'],p['query_ids'])) if keep[k]]
    available=set(range(len(case['gt_xywh'])));matched={}
    for detindex in sorted(range(len(dets)),key=lambda j:-dets[j][2]):
        box,label,score,qid=dets[detindex];best=-1;best_iou=-1.
        for g in sorted(available):
            if int(case['gt_labels'][g])!=label:continue
            value=iou(box,case['gt_xywh'][g]);pairs+=1
            if value>best_iou:best_iou=value;best=g
        tp=best>=0 and best_iou>=.5
        if tp:available.remove(best);matched[best]=score
        events.append((score,int(not tp),image))
    with np.load(E/'captures'/('case%02d.npz'%image),allow_pickle=False) as z:
        retained={depth:set(map(int,z[f'depth{depth}_rank_valid'][:250])) for depth in [1,2]}
    for g,(gt,label,rawrow) in enumerate(zip(case['gt_xywh'],case['gt_labels'],case['gt_row_indices'])):
        best={1:-math.inf,2:-math.inf}
        for box,dl,score,qid in dets:
            if dl!=label:continue
            value=iou(box,gt);pairs+=1
            if value>=.5:
                for depth in [1,2]:
                    if qid in retained[depth]:best[depth]=max(best[depth],score)
        tp_score=matched.get(g,-math.inf)
        records.append(dict(image=image,label=int(label),raw_row=int(rawrow),small=bool(gt[2]*gt[3]<=1024),
                            tp=tp_score,bin=sum(tp_score>=x for x in [.1,.25,.5,.75]),best=best))
rows=np.load(D/'native_support_rows.npz',allow_pickle=False)
assert len(records)==out['GT_rows']
for j,r in enumerate(records):
    for k,field in [('image','image'),('label','label'),('raw_row','raw_row'),('small','small'),('tp','native_TP_score'),('bin','final_score_bin')]:
        assert r[k]==rows[field][j],(j,k)
    for depth in [1,2]:assert r['best'][depth]==rows[f'best_retained_score_depth{depth}'][j]
weights=np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)
rng=np.random.default_rng(cfg['metrics']['bootstrap_seed']);draws=rng.integers(0,12,size=(1000,12))
assert np.array_equal(draws,weights['draws'])
assert np.array_equal(weights['weights'][:,1:],np.column_stack([np.bincount(d,minlength=12) for d in draws]))
boot=np.load(D/'paired_support_bootstrap.npz',allow_pickle=False)
groups=[]
for score,fp,image in sorted(events,key=lambda x:-x[0]):
    if not groups or groups[-1][0]!=score:groups.append([score,[]])
    groups[-1][1].append((fp,image))
def cutoff(w):
    fp=0;t=math.inf
    for score,items in groups:
        cost=sum(flag*int(w[i]) for flag,i in items)
        if fp+cost>cfg['metrics']['global_FP_budget']:break
        fp+=cost;t=score
    return t,fp
def stat(w,t,depth,adjusted=False):
    total=[0,0];lost=[0,0];strata={}
    for r in records:
        count=int(w[r['image']])
        if not count or r['tp']<t:continue
        kind=0 if r['small'] else 1;bad=r['best'][depth]<t
        total[kind]+=count;lost[kind]+=count*bad
        if adjusted:
            key=(r['image'],r['label'],r['bin']);v=strata.setdefault(key,[0,0,0,0]);v[kind]+=count;v[2+kind]+=count*bad
    rates=[lost[i]/total[i] if total[i] else None for i in [0,1]]
    gap=rates[0]-rates[1] if all(x is not None for x in rates) else None
    numerator=denominator=0.
    for ns,nl,ls,ll in strata.values():
        if ns and nl:
            common=min(ns,nl);numerator+=common*(ls/ns-ll/nl);denominator+=common
    return rates,gap,(numerator/denominator if denominator else None)
maximum=0.;gaps=[]
for j in range(1001):
    w=weights['weights'][:,j];t,fp=cutoff(w)
    assert t==boot['thresholds'][j] and fp==boot['actual_FP'][j]
    rates,gap,adjusted=stat(w,t,1,adjusted=j in [0,1,500,1000])
    expected=boot['depth1'][j]
    if gap is None:assert math.isnan(expected)
    else:maximum=max(maximum,abs(gap-float(expected)));assert abs(gap-float(expected))<=1e-14
    if j:gaps.append(gap)
    if j==0:
        p=out['point']['1'];assert rates==[p['small']['rate'],p['larger']['rate']]
        if adjusted is None:assert p['adjusted_small_minus_larger'] is None
        else:
            error=abs(adjusted-p['adjusted_small_minus_larger']);maximum=max(maximum,error);assert error<=1e-14
    if j in [0,1,500,1000]:
        _,g2,_=stat(w,t,2);expected2=boot['depth2'][j]
        if g2 is None:assert math.isnan(expected2)
        else:assert abs(g2-expected2)<=1e-14
finite=sorted(x for x in gaps if x is not None)
def quantile(p):
    x=(len(finite)-1)*p;i=math.floor(x);f=x-i
    return finite[i]*(1-f)+finite[min(i+1,len(finite)-1)]*f
if finite:
    ci=[quantile(.025),quantile(.975)]
    assert max(abs(a-b) for a,b in zip(ci,out['paired_image_intervals']['1']['small_minus_larger_95']))<=1e-14
receipt=dict(status='PASS_SCALAR_ALL_GT_SUPPORTS_ALL1000_PRIMARY_BOOTSTRAPS',
             GT_rows_checked=len(records),scalar_IoU_pairs=pairs,primary_columns_checked=1001,
             secondary_columns_checked=[0,1,500,1000],max_statistic_error=maximum,
             new_model_calls=0,scope='Independent numerical matching,backup coverage and prefix statistics; same source-aligned ignore policy,not independent causal or novelty review.')
with (D/'independent_retention_verification.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt))
