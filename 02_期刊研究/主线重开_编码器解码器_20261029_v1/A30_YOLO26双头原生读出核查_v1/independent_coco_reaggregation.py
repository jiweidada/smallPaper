"""Independent NumPy COCO-style matching for the diagnostic's internal crowd policy.

No Ultralytics,Torch,pycocotools or SciPy import. Not official VisDrone evaluation.
"""
from pathlib import Path
import collections,copy,json,math,sys,time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;D=R/'downloaded_evidence'
THRESHOLDS=np.linspace(.5,.95,10);RECALL=np.linspace(0,1,101)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def matrix_iou(ds,gs,crowd):
    if not len(ds) or not len(gs):return np.zeros((len(ds),len(gs)))
    a=np.asarray([x['bbox'] for x in ds],dtype=np.float64);b=np.asarray([x['bbox'] for x in gs],dtype=np.float64)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:]);wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1]
    da=a[:,2]*a[:,3];ga=b[:,2]*b[:,3];union=da[:,None]+ga[None,:]-inter;union[:,crowd]=np.broadcast_to(da[:,None],union.shape)[:,crowd]
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def match_image(ds,gs,area):
    ds=sorted(ds,key=lambda x:-x['score'])[:500]
    gs=sorted(gs,key=lambda x:bool(x.get('iscrowd',0)) or x['area']<area[0] or x['area']>area[1])
    gi=np.asarray([bool(x.get('iscrowd',0)) or x['area']<area[0] or x['area']>area[1] for x in gs],dtype=bool);crowd=np.asarray([bool(x.get('iscrowd',0)) for x in gs],dtype=bool)
    overlaps=matrix_iou(ds,gs,crowd);dm=np.zeros((10,len(ds)),dtype=np.int64);di=np.zeros((10,len(ds)),dtype=bool);gm=np.zeros((10,len(gs)),dtype=bool)
    for ti,threshold in enumerate(THRESHOLDS):
        for j,d in enumerate(ds):
            best=-1;level=min(threshold,1-1e-10)
            for k,g in enumerate(gs):
                if gm[ti,k] and not crowd[k]:continue
                if best>=0 and not gi[best] and gi[k]:break
                if overlaps[j,k]<level:continue
                level=overlaps[j,k];best=k
            if best>=0:
                dm[ti,j]=gs[best]['id'];di[ti,j]=gi[best];gm[ti,best]=True
    outside=np.asarray([x['bbox'][2]*x['bbox'][3]<area[0] or x['bbox'][2]*x['bbox'][3]>area[1] for x in ds],dtype=bool)
    di|=(dm==0)&outside[None,:]
    return {'scores':np.asarray([x['score'] for x in ds]),'ids':np.asarray([x['_id'] for x in ds]),'matches':dm,'ignore':di,'gt_ignore':gi}
def category_precision(records):
    positives=sum(int((~x['gt_ignore']).sum()) for x in records)
    if not positives:return None
    scores=np.concatenate([x['scores'] for x in records]);order=np.argsort(-scores,kind='stable');matches=np.concatenate([x['matches'] for x in records],axis=1)[:,order];ignore=np.concatenate([x['ignore'] for x in records],axis=1)[:,order]
    tp=np.cumsum((matches>0)&~ignore,axis=1);fp=np.cumsum((matches==0)&~ignore,axis=1);out=np.zeros((10,101))
    for ti in range(10):
        if not tp.shape[1]:continue
        rc=tp[ti]/positives;pr=tp[ti]/(tp[ti]+fp[ti]+np.spacing(1));pr=np.maximum.accumulate(pr[::-1])[::-1];idx=np.searchsorted(rc,RECALL,side='left');good=idx<len(pr);out[ti,good]=pr[idx[good]]
    return out
def ap(records,images,cats,draw=None):
    chosen=images if draw is None else [images[int(i)] for i in draw];p=[category_precision([records[(c,im)] for im in chosen]) for c in cats];p=[x for x in p if x is not None]
    return (float(np.mean(p)),float(np.mean([x[0] for x in p]))) if p else (None,None)
def operating_points(records,images,cats,small_ids,budgets):
    entries=[]
    for c in cats:
        for im in images:
            r=records[(c,im)]
            for j,s in enumerate(r['scores']):
                if not r['ignore'][0,j]:entries.append((float(s),im,c,int(r['ids'][j]),int(r['matches'][0,j])))
    entries.sort(key=lambda x:(-x[0],x[1],x[2],x[3]));groups=[]
    import itertools
    for score,block in itertools.groupby(entries,key=lambda x:x[0]):groups.append((score,list(block)))
    result=[]
    for budget in budgets:
        fp=0;tp=set();threshold=None
        for score,items in groups:
            more=sum(x[4]==0 for x in items)
            if fp+more>budget:break
            fp+=more;tp.update(x[4] for x in items if x[4]);threshold=score
        result.append({'FP_budget':budget,'actual_FP':fp,'TP':len(tp),'small_TP':len(tp&small_ids),'small_FN':len(small_ids-tp),'global_score_threshold':threshold})
    return result
def main():
    start=time.monotonic();cfg=read(R/'native_dual_head_protocol_FROZEN.json');raw=Path(cfg['local_raw_root']);gt=read(raw/'dataset/diagnostic_gt_orig.json');meta=read(raw/'receipts/diagnostic_observations.json')['images'];mapping={x['image_id']:x for x in meta}
    gt=copy.deepcopy(gt)
    for g in gt['annotations']:
        z=mapping[g['image_id']]['geometry'];s=z['ratio'];l,t=z['pad_ltrb'][:2];x,y,w,h=g['bbox'];g['bbox']=[x*s+l,y*s+t,w*s,h*s];g['area']=w*h*s*s
    images=sorted(x['id'] for x in gt['images']);cats=sorted(x['id'] for x in gt['categories']);assert cats==list(range(10))
    grouped_gt=collections.defaultdict(list)
    for x in gt['annotations']:grouped_gt[(x['category_id'],x['image_id'])].append(x)
    expected=read(D/'result.json');small_ids={x['id'] for x in gt['annotations'] if not x['iscrowd'] and x['area']<1024};all_results={};native_small={};maxerror=0.
    for arm in cfg['arms']:
        name=arm['id'];pred=read(D/(name+'_predictions_input.json'));grouped=collections.defaultdict(list)
        for i,x in enumerate(pred):x['_id']=i+1;grouped[(x['category_id'],x['image_id'])].append(x)
        sets={}
        for label,limits in [('all',[0,1e10]),('small',[0,math.nextafter(1024.,-math.inf)])]:
            sets[label]={(c,im):match_image(grouped[(c,im)],grouped_gt[(c,im)],limits) for c in cats for im in images}
        allAP,AP50=ap(sets['all'],images,cats);smallAP,_=ap(sets['small'],images,cats);got={'AP500':allAP,'AP50_500':AP50,'AP_small_500':smallAP}
        for k,v in got.items():maxerror=max(maxerror,abs(v-expected['metrics'][name]['input_area_AP'][k]));assert abs(v-expected['metrics'][name]['input_area_AP'][k])<1e-12,(name,k,v,expected['metrics'][name]['input_area_AP'][k])
        ops=operating_points(sets['all'],images,cats,small_ids,cfg['metrics']['same_FP_budgets'])
        for x,y in zip(ops,expected['metrics'][name]['same_FP']):
            for k in ['FP_budget','actual_FP','TP','small_TP','small_FN','global_score_threshold']:assert x[k]==y[k],(name,k,x[k],y[k])
        all_results[name]={'AP':got,'same_FP':ops}
        if name in cfg['metrics']['native_endpoints']:native_small[name]=sets['small']
        print(json.dumps({'verified_arm':name,'AP':got,'elapsed':time.monotonic()-start}),flush=True)
    boot=np.load(D/'paired_native_bootstrap.npz',allow_pickle=False);draws=boot['draws'];delta=boot['delta'];wanted=np.random.default_rng(cfg['metrics']['bootstrap']['seed']).integers(0,100,size=(1000,100));assert np.array_equal(draws,wanted)
    for i in [0,499,999]:
        m,_=ap(native_small['Mbox_Mscore_NMS'],images,cats,draws[i]);o,_=ap(native_small['Obox_Oscore_TOPK'],images,cats,draws[i]);assert abs(o-m-delta[i])<1e-12,(i,o-m,delta[i])
    assert np.allclose(np.quantile(delta,[.025,.975]),expected['native_O_minus_M_paired_image_bootstrap95'],rtol=0,atol=1e-15)
    result={'status':'PASS_ALL8_INTERNAL_COCO_AP_AND_FIXED_FP_WITH3_BOOTSTRAP_DRAWS','AP_max_abs_error':maxerror,'arms_verified':8,'same_FP_rows_verified':24,'bootstrap_draws_seed_checked':1000,'bootstrap_AP_draws_independently_recomputed':[0,499,999],'new_model_calls':0,'scope':'Independent internalCOCOcrowd reaggregation;not officialVisDrone,novelty or causality validation.','results':all_results,'elapsed_seconds':time.monotonic()-start}
    with (R/'independent_metric_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False))
if __name__=='__main__':main()
