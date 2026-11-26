"""One follow-up: exclude the target category from privileged support entirely."""
import csv,json,time
import numpy as np
from pathlib import Path
from audit_scene_scale import P,LAM,MIN_N,load_rows,stable,summarize,sha
def main():
    out=P/'03_异类支持核查_v1.json';assert not out.exists()
    initial=json.loads((P/'02_场景尺度统计核查_v1.json').read_text(encoding='utf-8'));mu=np.asarray(initial['canonical_class_log_offsets'])
    rows=load_rows();scored=[];start=time.monotonic()
    for r in rows:
        a=r['a'];n=len(a)
        if r['calibration'] or n<MIN_N:continue
        C=a[:,0].astype(int);X=np.column_stack([np.ones(n),a[:,1:3]]);zall=a[:,3]-mu[C]
        rng=np.random.default_rng(stable(r['file']+':A54-folds'));order=rng.permutation(n);fold=np.zeros(n,dtype=int);fold[order[n//2:]]=1
        for k in [0,1]:
            for c in np.unique(C[fold==k]):
                sup=np.flatnonzero((fold!=k)&(C!=c));targets=np.flatnonzero((fold==k)&(C==c));ns=len(sup)
                if ns<4:continue
                A=X[sup];z=zall[sup];beta=np.linalg.solve(A.T@A/ns+np.diag([0,LAM,LAM]),A.T@z/ns)
                srng=np.random.default_rng(stable(r['file']+f':A54-cross-{k}-{c}'));S=A[srng.permutation(ns)];bs=np.linalg.solve(S.T@S/ns+np.diag([0,LAM,LAM]),S.T@z/ns)
                for j in targets:
                    x={'file':r['file'],'annotation_row':int(a[j,6])+1,'class0':int(c),'fold':k,'support_objects':ns,'support_categories':int(len(np.unique(C[sup]))),
                        'short_side_640':float(a[j,4]),'original_area':float(a[j,5]),'GT_log_geomean_640':float(a[j,3])}
                    preds={'global':float(z.mean()+mu[c]),'spatial':float(X[j]@beta+mu[c]),'shuffled':float(X[j]@bs+mu[c])}
                    for name,p in preds.items():x['pred_'+name]=p;x['error_'+name]=abs(p-a[j,3])
                    scored.append(x)
    result={'status':'COMPLETE_PRIVILEGED_OTHER_CLASS_SUPPORT_AUDIT','initial_result_already_visible':True,
        'followup_reason':'Initial affine audit did not isolate cross-class support; this follow-up disallows the target category in support.',
        'train_annotations_reverified':6471,'validation_data_used':False,'target_category_present_in_support':False,'minimum_other_class_support':4,
        'all':summarize(scored,lambda x:True),'original_small_shortside_4to8':summarize(scored,lambda x:x['original_area']<=1024 and 4<=x['short_side_640']<8),
        'per_class':{str(c):summarize(scored,lambda x,c=c:x['class0']==c) for c in range(10)},
        'scope':'Ground-truth support is privileged. This is not a deployed detector result, not proof of cause, and not an untouched test; coverage is restricted by availability of other-class support.',
        'new_detector_forwards':0,'new_detector_optimizer_updates':0,'new_COCO_evaluate_calls':0,'script_sha256':sha(Path(__file__)),'seconds':time.monotonic()-start}
    with (P/'03_逐目标异类支持记录_v1.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(scored[0]));wr.writeheader();wr.writerows(scored)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','all','original_small_shortside_4to8']},ensure_ascii=False))
if __name__=='__main__':main()
