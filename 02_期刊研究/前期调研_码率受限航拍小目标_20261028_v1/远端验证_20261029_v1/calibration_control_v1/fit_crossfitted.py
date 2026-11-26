"""Ordinary CPU cross-fitted calibration controls. No detector, restorer or GPU use."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sys,time
import joblib,numpy as np,scipy,sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;LOCAL=ROOT.parent
PROXY=LOCAL/'local_proxy_v1/server_receipts';REG=LOCAL/'完成结果_20261029_v1/regional_raw/restoration_probe_v1'
METHODS=['regional_threshold','regional_ridge','regional_hgb','global_ridge']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def subset(values):return sum(1<<k for k,v in enumerate(values) if v>0)
def threshold_fit(score,target):
    choices=np.concatenate(([-np.inf],np.unique(score)))
    values=[float(target[score>t].sum()) for t in choices]
    best=max(range(len(choices)),key=lambda k:(values[k],choices[k]))
    return choices[best],values[best]
def main():
    tic=time.monotonic()
    with (ROOT/'fit_one_shot.lock').open('x') as f:f.write('existing local CPU calibration; no GPU')
    assert sklearn.__version__=='1.5.1' and np.__version__=='1.26.4'
    source=json.loads((PROXY/'decisions.json').read_text(encoding='utf-8'));record=json.loads((PROXY/'decisions_frozen.json').read_text())
    assert sha(PROXY/'decisions.json')==record['decisions_sha256']
    names=sorted([r['file_name'] for r in source['D0']],key=lambda n:(hashlib.sha256(('local-calibration-v1:'+n).encode()).hexdigest(),n))
    foldmap={n:i%5 for i,n in enumerate(names)};assert len(names)==len(set(names))==100
    write('fold_manifest.json',dict(rule='SHA256 local-calibration-v1 filename sort then round robin',source_folds=foldmap,source_count=100,region_count=400))
    fits=[];result={};diagnostics={};(ROOT/'models').mkdir(exist_ok=True)
    for arm in ['D0','DA']:
        rows=source[arm];assert {r['file_name'] for r in rows}==set(names)
        truth={r['image_id']:r for r in json.loads((REG/(arm+'_loss_diagnostics.json')).read_text())}
        x=np.asarray([[[r['regional_pseudograd_scores'][k],r['confidence_region_deltas'][k],np.log1p(r['pseudo_label_count']),r['pseudo_loss']] for k in range(4)] for r in rows])
        y=np.asarray([truth[r['image_id']]['dynamic']['singles'] for r in rows]);assert x.shape==(100,4,4) and y.shape==(100,4)
        xg=np.asarray([[r['global_pseudograd_score'],sum(r['confidence_region_deltas']),np.log1p(r['pseudo_label_count']),r['pseudo_loss']] for r in rows])
        yg=np.asarray([truth[r['image_id']]['losses'][0]-truth[r['image_id']]['losses'][15] for r in rows])
        folds=np.asarray([foldmap[r['file_name']] for r in rows]);oof={m:[None]*100 for m in METHODS};pred={m:[None]*100 for m in METHODS}
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
            assert len(train)==80 and len(test)==20 and not set(train)&set(test)
            train_ids=[rows[i]['image_id'] for i in train];test_ids=[rows[i]['image_id'] for i in test]
            assert not set(train_ids)&set(test_ids)
            xx=x[train].reshape(-1,4);yy=y[train].reshape(-1);xt=x[test].reshape(-1,4)
            threshold,objective=threshold_fit(xx[:,0],yy)
            values=(xt[:,0]-threshold).reshape(20,4) if np.isfinite(threshold) else np.ones((20,4))
            for j,idx in enumerate(test):oof['regional_threshold'][idx]=subset(values[j]);pred['regional_threshold'][idx]=values[j].tolist()
            fits.append(dict(arm=arm,method='regional_threshold',fold=fold,train_ids=train_ids,test_ids=test_ids,
                 threshold=float(threshold) if np.isfinite(threshold) else '-inf(all)',training_additive_utility=objective))
            models={'regional_ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)),
                    'regional_hgb':HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,max_depth=2,max_leaf_nodes=4,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=0)}
            for method,model in models.items():
                model.fit(xx,yy);values=model.predict(xt).reshape(20,4);assert np.isfinite(values).all()
                for j,idx in enumerate(test):oof[method][idx]=subset(values[j]);pred[method][idx]=values[j].tolist()
                path=ROOT/'models'/(arm+'_'+method+'_fold'+str(fold)+'.joblib');joblib.dump(model,path)
                fits.append(dict(arm=arm,method=method,fold=fold,train_ids=train_ids,test_ids=test_ids,model_file=str(path),model_sha256=sha(path)))
            model=make_pipeline(StandardScaler(),Ridge(alpha=10));model.fit(xg[train],yg[train]);values=model.predict(xg[test]);assert np.isfinite(values).all()
            for j,idx in enumerate(test):oof['global_ridge'][idx]=15 if values[j]>0 else 0;pred['global_ridge'][idx]=float(values[j])
            path=ROOT/'models'/(arm+'_global_ridge_fold'+str(fold)+'.joblib');joblib.dump(model,path)
            fits.append(dict(arm=arm,method='global_ridge',fold=fold,train_ids=train_ids,test_ids=test_ids,model_file=str(path),model_sha256=sha(path)))
        assert all(all(v is not None for v in oof[m]) for m in METHODS)
        result[arm]=[dict(image_id=r['image_id'],file_name=r['file_name'],fold=int(folds[i]),choices={m:oof[m][i] for m in METHODS},predictions={m:pred[m][i] for m in METHODS}) for i,r in enumerate(rows)]
        diagnostics[arm]={m:dict(selection_counts={str(k):oof[m].count(k) for k in range(16)},
                              mean_actual_loss_utility=float(np.mean([truth[r['image_id']]['losses'][0]-truth[r['image_id']]['losses'][oof[m][i]] for i,r in enumerate(rows)]))) for m in METHODS}
    assert len(fits)==40
    write('crossfitted_decisions.json',result);write('fit_receipts.json',fits)
    freeze=dict(status='CROSSFITTED_DECISIONS_FROZEN_BEFORE_AP_SCORING',script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),
        decision_sha256=sha(ROOT/'crossfitted_decisions.json'),fit_receipts_sha256=sha(ROOT/'fit_receipts.json'),
        input_proxy_decisions_sha256=sha(PROXY/'decisions.json'),environment=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,joblib=joblib.__version__),
        fitted_utility_policies=40,detector_training_updates=0,detector_forwards=0,restorer_forwards=0,elapsed_seconds=time.monotonic()-tic,fold_leakage_assertions_passed=True,
        interpretation='Exploratory source-grouped development cross-fitting, not independent confirmation; no AP was read by this fitter')
    write('crossfit_frozen.json',freeze);write('crossfit_loss_diagnostics.json',diagnostics)
    print(json.dumps(freeze),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
