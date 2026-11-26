"""1000 paired natural-val replicas for cached baseline/synthetic predictions only.

Vectorized COCO precision envelope is checked bitwise against the bound native
accumulator and independent full dataset clones. Never averages per-image AP.
"""
import importlib,json,time,copy,gc,hashlib
C=importlib.import_module('343_context');E=importlib.import_module('343_evaluator')
B=importlib.import_module('343_bootstrap');U=importlib.import_module('343_bootstrap_units')
import numpy as np

def prepare(ev):
    p=ev._paramsEval;n=len(p.imgIds);a=len(p.areaRng);k=len(p.catIds);tables=[]
    assert list(p.imgIds)==list(range(548)) and p.areaRngLbl[:2]==['all','small']
    for category in range(k):
        row=[]
        for area in (0,1):
            records=ev.evalImgs[(category*a+area)*n:(category*a+area+1)*n]
            one=[]
            for r in records:
                if r is None:one.append(None);continue
                one.append((np.asarray(r['dtScores'][:100]),r['dtMatches'][:,:100]!=0,r['dtIgnore'][:,:100],int((r['gtIgnore']==0).sum())))
            row.append(one)
        tables.append(row)
    return tables,np.asarray(p.recThrs)

def fast_ap(prepared,indices):
    tables,recall=prepared;precision=np.full((10,101,len(tables),2),-1.,dtype=float)
    for category,areas in enumerate(tables):
        for area,records in enumerate(areas):
            selected=[records[int(i)] for i in indices if records[int(i)] is not None]
            if not selected:continue
            npig=sum(r[3] for r in selected)
            if npig==0:continue
            scores=np.concatenate([r[0] for r in selected]);order=np.argsort(-scores,kind='mergesort')
            matched=np.concatenate([r[1] for r in selected],axis=1)[:,order]
            ignored=np.concatenate([r[2] for r in selected],axis=1)[:,order]
            tp=np.cumsum(matched&~ignored,axis=1).astype(float);fp=np.cumsum(~matched&~ignored,axis=1).astype(float)
            rc=tp/npig;pr=tp/(fp+tp+np.spacing(1));q=np.zeros((10,101))
            if pr.shape[1]:
                envelope=np.maximum.accumulate(pr[:,::-1],axis=1)[:,::-1]
                for t in range(10):
                    ind=np.searchsorted(rc[t],recall,side='left');valid=ind<pr.shape[1];q[t,valid]=envelope[t,ind[valid]]
            precision[:,:,category,area]=q
    out=[]
    for area in (0,1):
        a=precision[:,:,:,area];a=a[a>-1];out.append(float(a.mean()) if len(a) else -1.)
    return out

def main():
    start=time.perf_counter();ann=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json')
    original=json.loads(ann.read_text(encoding='utf-8'));dataset=B.positive_gt_ids(original)
    ix=np.load(C.HERE/'343_bootstrap_构造1000次_共享抽样.npy');assert ix.shape==(1000,548)
    assert np.array_equal(ix,np.random.default_rng(2026102103).integers(0,548,size=(1000,548),dtype=np.int64))
    formal=json.loads((C.HERE/'343_result_阶段4正式计数.json').read_text(encoding='utf-8'))
    tables={};bcounts={};proof=[];clones=[];source_stats={};calls=0
    for seed in (1,2):
        for mode in ('baseline','low_new4','random_new4'):
            name=f'seed{seed}_{mode}';path=C.HERE/f'343_val_seed{seed}_{mode}_predictions.json'
            predictions=json.loads(path.read_text(encoding='utf-8'));ev,stats,_,_=E.evaluate(dataset,predictions);source_stats[name]=stats
            prepared=prepare(ev);tables[name]=prepared
            for trial,sample in enumerate((ix[0],ix[999],np.tile(np.array([0,5]),274))):
                native=B.ap_aps(B.sampled_evaluator(ev,sample,ap_only=True));fast=fast_ap(prepared,sample)
                assert native==fast,(name,trial,native,fast)
                proof.append({'case':name,'trial':trial,'AP_APs_native':native,'AP_APs_vectorized':fast,'bit_equal':True});calls+=1
            if mode=='baseline':
                # New image/GT IDs, fresh IoU evaluation, exact native comparison.
                ds,ps=U.independent_clone(dataset,predictions,ix[0]);fresh,_,_,_=E.evaluate(ds,ps)
                assert B.ap_aps(fresh)==fast_ap(prepared,ix[0]);clones.append({'case':name,'trial':0,'bit_equal':True});del fresh
            bcounts[name]=np.asarray([r['cases'][mode]['0.5']['B'] for r in formal['per_seed'][str(seed)]],dtype=np.int64)
            del ev,predictions;gc.collect();print(f'bootstrap prepared {name}',flush=True)
    outputs={name:[] for name in tables}
    for r,sample in enumerate(ix):
        for name,table in tables.items():outputs[name].append(fast_ap(table,sample)+[int(bcounts[name][sample].sum())])
        if (r+1)%50==0:print(f'natural paired bootstrap {r+1}/1000 elapsed={time.perf_counter()-start:.1f}s',flush=True)
    differences={}
    for seed in (1,2):
        baseline=np.asarray(outputs[f'seed{seed}_baseline'])
        for mode in ('low_new4','random_new4'):
            delta=np.asarray(outputs[f'seed{seed}_{mode}'])-baseline
            differences[f'seed{seed}_{mode}_minus_baseline']={'percentile95_AP_APs_B':np.percentile(delta,[2.5,97.5],axis=0).tolist()}
    C.write('343_bootstrap_自然val合成_逐次AP_APs_B.json',outputs)
    C.write('343_result_阶段4自然val配对bootstrap.json',{'status':'已验证（自然val冻结baseline＋合成新框工程验证）',
        'seed':2026102103,'replicates':1000,'images_per_replica':548,'shared_index_sha256':hashlib.sha256(ix.tobytes()).hexdigest(),
        'metrics_order':['AP','APs','B'],'native_accumulator_proof':proof,'fresh_full_clone_proof':clones,
        'percentile95':{name:np.percentile(rows,[2.5,97.5],axis=0).tolist() for name,rows in outputs.items()},'paired_difference_percentile95':differences,
        'source_dataset_stats_after_GT_ID_reindex':source_stats,
        'ID_handling':'每次重复图赋新image_id，GT/detection ID分离；源匹配也先把GT id从0起重编号为正值，避免COCO dtMatches中0与未匹配哨兵混淆。该重编号只用于重采样，原baseline12项统计未覆盖。',
        'method':'每个重采样数据集重新累计TP/FP、全局稳定score排序、precision包络与101点AP；复用每图IoU/匹配，不是逐图AP平均。',
        '已执行':{'source_COCO_evaluate_calls':6,'independent_full_clone_COCO_evaluate_calls':2,'native_sampled_accumulate_calls':calls,
            'vectorized_AP_APs_recomputations':6000,'B_resampled_sums':6000,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['八run训练后科学bootstrap','真实训练','GPU','阶段5'],'未核':['主控独立复核'],'wall_seconds':time.perf_counter()-start,'source_sha256':C.sources()})

if __name__=='__main__':main()
