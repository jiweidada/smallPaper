"""Fixed-pool image-common score-location diagnostic; no model imports/fitting."""
import sys, json, hashlib, time, copy, contextlib, io, collections, csv
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
E=ROOT/'data/00_实验计划与证据索引'
ER=E/'单一候选评价依赖补齐与构建核验_20261008_v1/isolated_runtime'
sys.path.insert(0,str(ER))
from pycocoeval.coco import COCO
from pycocoeval.cocoeval import COCOeval
BASE=E/'正式132轮基线整图双问题诊断_20261008_v1/native_transform_v2'
ANN=Path('C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val/annotations/val.json')
NAMES=['AP','AP50','AP75','APs','APm','APl']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (HERE/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def overlap(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(-1)
    return inter/np.maximum(a[:,2:].prod(-1)[:,None]+b[:,2:].prod(-1)[None,:]-inter,1e-300)

def accumulate(data,scores,mask=None,legacy_id0=False):
    aps=np.full((4,10,10),np.nan)
    for (area,cat),x in data.items():
        ids=x['row'];keep=np.ones(len(ids),bool) if mask is None else mask[ids]
        ids=ids[keep];matches=x['matches'][:,keep]
        ignored=x['ignored'][:,keep]
        if legacy_id0:matches=matches.copy();matches[matches==1]=0
        npig=x['npig']
        if not npig:continue
        order=np.argsort(-scores[ids],kind='mergesort')
        tps=(matches[:,order]>0)&~ignored[:,order]
        fps=(matches[:,order]==0)&~ignored[:,order]
        tp=np.cumsum(tps,axis=1,dtype=float);fp=np.cumsum(fps,axis=1,dtype=float)
        rc=tp/npig;pr=tp/(tp+fp+np.spacing(1))
        pr=np.maximum.accumulate(pr[:,::-1],axis=1)[:,::-1]
        for t in range(10):
            inds=np.searchsorted(rc[t],np.linspace(0,1,101),side='left')
            q=np.zeros(101);ok=inds<len(ids)
            q[ok]=pr[t,inds[ok]]
            aps[area,cat,t]=q.mean()
    result={'AP':float(np.nanmean(aps[0])),'AP50':float(np.nanmean(aps[0,:,0])),
            'AP75':float(np.nanmean(aps[0,:,5])),'APs':float(np.nanmean(aps[1])),
            'APm':float(np.nanmean(aps[2])),'APl':float(np.nanmean(aps[3]))}
    return result,aps

def inv_count(p,n):
    p=np.asarray(p,float);n=np.sort(np.asarray(n,float))
    l=np.searchsorted(n,p,'left');r=np.searchsorted(n,p,'right')
    ties=int((r-l).sum());strict=int((len(n)-r).sum())
    return len(p)*len(n),strict,ties

def inversions(data,scores,ii,short,allmax,samemax,othermax,classes):
    output=[]
    for cat in range(10):
        x=data[(0,cat)];ids=x['row']
        for ti in [0,5]:
            t=.5+ti*.05
            tp=(x['matches'][ti]>0)&~x['ignored'][ti]
            fp=(x['matches'][ti]==0)&~x['ignored'][ti]
            kinds={'all':np.ones(len(ids),bool),'near_annotated':allmax[ids]>=.1,
                'no_GT_01':allmax[ids]<.1,'sameclass_ge_threshold':samemax[ids]>=t,
                'wrongclass_ge_threshold':(samemax[ids]<t)&(othermax[ids]>=t),
                'localization_near_label':(samemax[ids]>=.1)&(samemax[ids]<t)&(othermax[ids]<t)}
            for floor in [0.,.25]:
                for kind,negfilter in kinds.items():
                    for sizebin in [-1,0,1,2,3,4]:
                        size=np.ones(len(ids),bool) if sizebin<0 else np.digitize(short[ids],[4,8,16,32])==sizebin
                        pp=ids[tp&(scores[ids]>=floor)&size]
                        nn=ids[fp&(scores[ids]>=floor)&size&negfilter]
                        den,strict,ties=inv_count(scores[pp],scores[nn]);wd=ws=wt=0
                        if len(pp) and len(nn):
                            for image in np.intersect1d(np.unique(ii[pp]),np.unique(ii[nn])):
                                z=inv_count(scores[pp[ii[pp]==image]],scores[nn[ii[nn]==image]])
                                wd+=z[0];ws+=z[1];wt+=z[2]
                        cd=den-wd;cs=strict-ws;ct=ties-wt
                        output.append({'class':cat,'IoU':t,'score_floor':floor,'FP_kind':kind,'pred_short_bin':sizebin,
                            'TP':len(pp),'FP':len(nn),'same_image_pairs':wd,'same_image_strict_inversions':ws,'same_image_ties':wt,
                            'same_image_inversion_rate':(ws+.5*wt)/wd if wd else None,
                            'cross_image_pairs':cd,'cross_image_strict_inversions':cs,'cross_image_ties':ct,
                            'cross_image_inversion_rate':(cs+.5*ct)/cd if cd else None})
    return output

def main():
    start=time.perf_counter();proto=HERE/'05_跨图排序_执行前协议.json'
    proto_sha=sha(proto);(HERE/'05_跨图排序_执行前协议.sha256').write_text(proto_sha,encoding='utf-8')
    sources={str(proto):proto_sha,str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(ANN):sha(ANN)}
    for p in [ER/'pycocoeval/coco.py',ER/'pycocoeval/cocoeval.py',ER/'pycocoeval/mask.py',ER/'pycocoeval/_mask.cp310-win_amd64.pyd']:sources[str(p)]=sha(p)
    ds=read(ANN);assert len(ds['images'])==548 and len(ds['annotations'])==38759
    ims={im['id']:im for im in ds['images']};by=collections.defaultdict(list)
    for a in ds['annotations']:by[a['image_id']].append(a)
    shifted=copy.deepcopy(ds)
    for a in shifted['annotations']:a['id']+=1
    out={'protocol_sha256':proto_sha,'seed_results':{},'sources':sources,'new_model_forwards':0,'training':0,'CPU_evaluator_calls':0}
    for seed in [1,2]:
        st=time.perf_counter();p=BASE/f'seed{seed}/predictions.json';sources[str(p)]=sha(p)
        ep=BASE/f'seed{seed}/evaluation.json';sources[str(ep)]=sha(ep)
        preds=read(p);s=np.array([d['score'] for d in preds]);ii=np.array([d['image_id'] for d in preds]);classes=np.array([d['category_id'] for d in preds])
        assert len(preds)==164400 and np.all((s>0)&(s<1))
        logits=np.log(s)-np.log1p(-s)
        imgids=sorted(ims);image_index={iid:j for j,iid in enumerate(imgids)};ix=np.array([image_index[i] for i in ii])
        pby={iid:np.flatnonzero(ii==iid) for iid in imgids}
        assert all(len(v)==300 for v in pby.values())
        means=np.array([logits[pby[iid]].mean() for iid in imgids])
        centered=logits-means[ix];mono=logits+1
        order_checks=0
        for iid,rows in pby.items():
            for zz in [mono,centered]:
                assert np.array_equal(np.argsort(-s[rows],kind='stable'),np.argsort(-zz[rows],kind='stable'))
                a=np.argsort(s[rows],kind='stable')
                assert np.array_equal(np.diff(s[rows][a])==0,np.diff(zz[rows][a])==0)
                order_checks+=1
        log=io.StringIO()
        print(f'seed{seed}: native CPU COCO evaluation, fixed outputs',flush=True)
        with contextlib.redirect_stdout(log):
            gt=COCO();gt.dataset=copy.deepcopy(shifted);gt.createIndex();dt=gt.loadRes(copy.deepcopy(preds))
            ev=COCOeval(gt,dt,'bbox',using_tiny_metrice=False,using_vtiny_metrice=False,max_dets=100)
            ev.params.imgIds=imgids;ev.evaluate();ev.accumulate();ev.summarize()
        (HERE/f'06_seed{seed}_positiveID_native_eval.txt').write_text(log.getvalue(),encoding='utf-8')
        out['CPU_evaluator_calls']+=1
        lists=collections.defaultdict(list)
        for e in ev.evalImgs:
            if e is None:continue
            area=ev.params.areaRng.index(e['aRng']);cat=int(e['category_id'])
            lists[(area,cat)].append(e)
        data={};arrays={}
        for key,ee in lists.items():
            ee=sorted(ee,key=lambda x:x['image_id'])
            x={'row':np.concatenate([np.array(e['dtIds'],int)-1 for e in ee]),
                'matches':np.concatenate([e['dtMatches'] for e in ee],axis=1).astype(int),
                'ignored':np.concatenate([e['dtIgnore'] for e in ee],axis=1).astype(bool),
                'npig':int(sum(np.count_nonzero(~np.array(e['gtIgnore'],bool)) for e in ee))}
            data[key]=x
            for k,v in x.items():arrays[f'a{key[0]}_c{key[1]}_{k}']=np.array(v)
        native,na=accumulate(data,s);monom,_=accumulate(data,mono);center,ca=accumulate(data,centered)
        native_error=max(abs(native[k]-ev.stats[j]) for j,k in enumerate(NAMES));mono_error=max(abs(native[k]-monom[k]) for k in NAMES)
        assert native_error<=1e-12 and mono_error<=1e-12,(native_error,mono_error)
        legacy,_=accumulate(data,s,legacy_id0=True);old=read(ep)['stats'];legacy_error=max(abs(legacy[k]-old[j]) for j,k in enumerate(NAMES))
        assert legacy_error<=1e-12,('legacy_CPU_parity',legacy_error,legacy,old[:6])
        allmax=np.zeros(len(preds));samemax=allmax.copy();othermax=allmax.copy();short=allmax.copy()
        for iid,rows in pby.items():
            boxes=np.array([preds[k]['bbox'] for k in rows]);gts=by[iid];gc=np.array([g['category_id'] for g in gts]);I=overlap(boxes,[g['bbox'] for g in gts])
            allmax[rows]=I.max(1,initial=0)
            for loc,k in enumerate(rows):
                samemax[k]=I[loc,gc==classes[k]].max(initial=0);othermax[k]=I[loc,gc!=classes[k]].max(initial=0)
            wh=np.array([ims[iid]['width'],ims[iid]['height']]);short[rows]=(boxes[:,2:]*640/wh).min(-1)
        supported=allmax>=.1
        near_base,_=accumulate(data,s,supported);near_center,_=accumulate(data,centered,supported)
        aspect=np.array([ims[iid]['width']/ims[iid]['height'] for iid in imgids])
        nconf=np.array([(s[pby[iid]]>=.5).sum() for iid in imgids])
        strata=np.digitize(aspect,[1.5,2])*4+np.digitize(nconf,[10,50,100])
        perm_results={};perm_maps={};keys=['AP','AP50','AP75','APs']
        print(f'seed{seed}: parity passed; evaluating frozen image-location permutations',flush=True)
        for arm_index,arm in enumerate(['unconditional','conditional'],1):
            rng=np.random.default_rng(20261027+seed*1000+arm_index);maps=[];res=[]
            for rep in range(199):
                order=np.arange(len(imgids))
                if arm=='unconditional':order=rng.permutation(order)
                else:
                    for k in np.unique(strata):
                        members=np.flatnonzero(strata==k);order[members]=rng.permutation(members)
                z=logits-means[order][ix]
                v,_=accumulate(data,z);res.append({k:v[k] for k in keys});maps.append(order)
            perm_results[arm]={'all':res,'q05':{k:float(np.quantile([v[k] for v in res],.05)) for k in keys},
                'q50':{k:float(np.quantile([v[k] for v in res],.5)) for k in keys},'q95':{k:float(np.quantile([v[k] for v in res],.95)) for k in keys}}
            perm_maps[arm]=np.array(maps)
        inv=inversions(data,s,ii,short,allmax,samemax,othermax,classes)
        invp=HERE/f'06_seed{seed}_inversions.csv'
        with invp.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)
        counts={str(int(50+t*5)):{'TP':int(sum(np.count_nonzero((x['matches'][t]>0)&~x['ignored'][t]) for (a,c),x in data.items() if a==0)),
            'FP':int(sum(np.count_nonzero((x['matches'][t]==0)&~x['ignored'][t]) for (a,c),x in data.items() if a==0))} for t in [0,5]}
        advances=all(center[k]>native[k]+1e-10 and center[k]>perm_results['conditional']['q95'][k]+1e-10 and near_center[k]>=near_base[k]-1e-10 for k in ['AP','APs'])
        arrays.update({'scores':s,'image_ids':ii,'classes':classes,'image_mean_logits':means,'image_order':np.array(imgids),'strata':strata,
            'allGT_max_IoU':allmax,'sameclass_max_IoU':samemax,'otherclass_max_IoU':othermax,'pred_short640':short,
            'unconditional_permutations':perm_maps['unconditional'],'conditional_permutations':perm_maps['conditional']})
        np.savez_compressed(HERE/f'06_seed{seed}_fixed_rank_arrays.npz',**arrays)
        sr={'native_positiveID':native,'native_legacyID0':legacy,'centered':center,'delta_pp':{k:100*(center[k]-native[k]) for k in NAMES},
            'annotation_supported_native':near_base,'annotation_supported_centered':near_center,
            'annotation_supported_delta_pp':{k:100*(near_center[k]-near_base[k]) for k in NAMES},
            'per_class_native_AP':np.nanmean(na[0],axis=1).tolist(),'per_class_centered_AP':np.nanmean(ca[0],axis=1).tolist(),
            'permutations':perm_results,'counts_unchanged_all_arms':counts,
            'checks':{'native_rank_AP_error':native_error,'global_monotone_AP_error':mono_error,'legacy_CPU_AP_error':legacy_error,
                'image_order_and_tie_checks':order_checks,'matching_and_cap':'fixed array identities reused after proving image order unchanged'},
            'n_no_GT_01_retained':int(sum(np.count_nonzero(~supported[x['row']]) for (a,c),x in data.items() if a==0)),
            'conditional_blocks':dict(collections.Counter(strata.tolist())),
            'advance_exact_hypothesis':bool(advances),'seconds':time.perf_counter()-st}
        out['seed_results'][str(seed)]=sr
        save(f'06_seed{seed}_rank_results.json',sr)
        print(json.dumps({'seed':seed,'delta_pp':sr['delta_pp'],'near_delta_pp':sr['annotation_supported_delta_pp'],
            'native_AP_pp':100*native['AP'],'conditional_q95_AP_pp':100*perm_results['conditional']['q95']['AP'],
            'centered_AP_pp':100*center['AP'],'advance':bool(advances),'seconds':sr['seconds']},ensure_ascii=False),flush=True)
    assert sha(proto)==proto_sha
    unchanged={p:sha(Path(p))==h for p,h in sources.items()}
    assert all(unchanged.values())
    out['input_hashes_unchanged']=unchanged
    out['both_seeds_advance']=all(v['advance_exact_hypothesis'] for v in out['seed_results'].values())
    out['seconds']=time.perf_counter()-start
    save('06_跨图排序_完整结果.json',out)
    print(json.dumps({'completed':True,'both_seeds_advance':out['both_seeds_advance'],'CPU_evaluator_calls':out['CPU_evaluator_calls'],'seconds':out['seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
