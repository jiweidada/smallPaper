from pathlib import Path
from bisect import bisect_left,bisect_right
import hashlib,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;A=D/'analysis';E=D/'downloaded_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def auc_reference(y,s):
    y=np.asarray(y);s=np.asarray(s);negative=sorted(float(x) for x in s[y==0]);positive=s[y==1]
    if not len(negative) or not len(positive):return float('nan')
    wins=sum(bisect_left(negative,float(x))+.5*(bisect_right(negative,float(x))-bisect_left(negative,float(x))) for x in positive)
    return wins/(len(negative)*len(positive))
def percentile_reference(values,p):
    s=sorted(float(v) for v in values if np.isfinite(v));index=(len(s)-1)*p
    lo=int(np.floor(index));hi=int(np.ceil(index));w=index-lo
    return s[lo]*(1-w)+s[hi]*w
protocol=read(D/'semantic_protocol_FROZEN.json');result=read(A/'semantic_analysis_result.json')
assert result['status']=='SEMANTIC_ACCESSIBILITY_SCREEN_COMPLETE' and result['probe_fit_entries']==7
records=read(A/'pair_records.json')['rows'];combined=read(D/'combined_cases_FROZEN.json')['cases']
by_case={r['index']:[] for r in combined}
for r in records:by_case[r['case_index']].append(r)
views=['QUERY','FINE','CONTRAST','GEOMETRY']
x={s:{v:[] for v in views} for s in ['train','val']};labels={s:[] for s in ['train','val']}
classes={s:[] for s in ['train','val']};case_ids={s:[] for s in ['train','val']};groups={s:[] for s in ['train','val']};primary=[]
for row in combined:
    pairs=by_case[row['index']]
    if not pairs:continue
    path=Path(row['capture_path']) if row['capture_phase']=='A38' else E/row['capture_relative']
    c=dict(np.load(path,allow_pickle=False));s=row['split']
    ids=np.array([[r['positive_query'],r['background_query']] for r in pairs],dtype=int).reshape(-1)
    for v,key in [('QUERY','query_features'),('FINE','fine_features'),('CONTRAST','contrast_features')]:x[s][v].append(c[key][ids].astype(np.float64))
    boxes=np.clip(c['query_boxes_xyxy'].astype(np.float64),0,1);wh=np.maximum(boxes[:,2:]-boxes[:,:2],1e-12)
    geometry=np.c_[np.log2(np.maximum(wh.prod(1),1e-12)),np.log2(wh[:,0]/wh[:,1]),
        c['native_probabilities'][:,1:11].astype(np.float64).max(1),(boxes[:,:2]+boxes[:,2:])/2]
    x[s]['GEOMETRY'].append(geometry[ids])
    labels[s].extend([1,0]*len(pairs));classes[s].extend([r['label'] for r in pairs])
    case_ids[s].extend([row['index']]*(2*len(pairs)));groups[s].extend([row['prefix']]*(2*len(pairs)))
    if s=='val':primary.extend(np.repeat([not r['native_support'] for r in pairs],2).tolist())
for s in ['train','val']:
    for v in views:x[s][v]=np.concatenate(x[s][v])
    labels[s]=np.asarray(labels[s]);classes[s]=np.asarray(classes[s]);case_ids[s]=np.asarray(case_ids[s]);groups[s]=np.asarray(groups[s])
primary=np.asarray(primary,dtype=bool)
stored=dict(np.load(A/'paired_probe_scores_and_bootstrap.npz',allow_pickle=False))
params=dict(np.load(A/'fitted_probe_parameters.npz',allow_pickle=False))
np.testing.assert_array_equal(stored['labels'],labels['val']);np.testing.assert_array_equal(stored['cases'],case_ids['val'])
np.testing.assert_array_equal(stored['groups'],groups['val']);np.testing.assert_array_equal(stored['primary_mask'],primary)
max_prediction_error=0.;normal_residuals={};std_errors={}
def verify_model(prefix,train,target,val):
    global max_prediction_error
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-6)
    np.testing.assert_allclose(params[prefix+'_mean'],mean,rtol=0,atol=1e-12)
    np.testing.assert_allclose(params[prefix+'_scale'],scale,rtol=0,atol=1e-12)
    np.testing.assert_allclose(params[prefix+'_intercept'],target.mean(0),rtol=0,atol=1e-15)
    z=(train-mean)/scale;rhs=z.T@(target-target.mean(0));weight=params[prefix+'_weight']
    residual=(z.T@z+np.eye(z.shape[1]))@weight-rhs
    normal_residuals[prefix]=float(np.max(np.abs(residual)))
    assert normal_residuals[prefix]<=1e-7+1e-10*float(np.max(np.abs(rhs)))
    return ((val-mean)/scale)@weight+target.mean(0)
y=labels['train']*2-1;scores={}
for v in views:
    scores[v]=verify_model(v,x['train'][v],y,x['val'][v])
    difference=float(np.max(np.abs(scores[v]-stored['score_'+v])));max_prediction_error=max(max_prediction_error,difference)
    np.testing.assert_allclose(scores[v],stored['score_'+v],rtol=1e-12,atol=1e-11)
qmean=x['train']['QUERY'].mean(0);qscale=np.maximum(x['train']['QUERY'].std(0),1e-6)
np.testing.assert_allclose(params['rff_input_mean'],qmean,atol=1e-12,rtol=0)
np.testing.assert_allclose(params['rff_input_scale'],qscale,atol=1e-12,rtol=0)
rng=np.random.default_rng(protocol['rff_seed']);omega=rng.normal(size=(256,512))/16.;phase=rng.uniform(0,2*np.pi,size=512)
np.testing.assert_array_equal(omega,params['rff_omega']);np.testing.assert_array_equal(phase,params['rff_phase'])
ztrain=np.sqrt(2/512)*np.cos(((x['train']['QUERY']-qmean)/qscale)@omega+phase)
zval=np.sqrt(2/512)*np.cos(((x['val']['QUERY']-qmean)/qscale)@omega+phase)
scores['QUERY_RFF']=verify_model('QUERY_RFF',ztrain,y,zval)
np.testing.assert_allclose(scores['QUERY_RFF'],stored['score_QUERY_RFF'],rtol=1e-12,atol=1e-11)
max_prediction_error=max(max_prediction_error,float(np.max(np.abs(scores['QUERY_RFF']-stored['score_QUERY_RFF']))))
target=np.eye(10)[classes['train']-1];val_classes=classes['val'][primary[::2]]
np.testing.assert_array_equal(params['train_classes'],classes['train']);np.testing.assert_array_equal(params['val_primary_classes'],val_classes)
train_counts=np.bincount(classes['train'],minlength=11)[1:11];val_counts=np.bincount(val_classes,minlength=11)[1:11]
np.testing.assert_array_equal(result['class_information_check']['train_counts'],train_counts)
np.testing.assert_array_equal(result['class_information_check']['val_primary_counts'],val_counts)
supported=(train_counts>=10)&(val_counts>=5)
for v in ['QUERY','FINE']:
    val=x['val'][v][::2][primary[::2]]
    prediction=verify_model(v+'_CLASS',x['train'][v][::2],target,val).argmax(1)+1
    np.testing.assert_array_equal(prediction,params[v+'_class_prediction'])
    correct=prediction==val_classes
    ref=result['class_information_check']['metrics'][v]
    assert abs(float(correct.mean())-ref['accuracy'])<1e-15
    per=[float(correct[val_classes==c].mean()) if np.any(val_classes==c) else None for c in range(1,11)]
    np.testing.assert_allclose([per[i] for i in np.flatnonzero(supported)],
        [ref['per_class_accuracy'][i] for i in np.flatnonzero(supported)],atol=1e-15,rtol=0)
    assert abs(float(np.mean([per[i] for i in np.flatnonzero(supported)]))-ref['supported_macro_accuracy'])<1e-15
point={v:auc_reference(labels['val'][primary],s[primary]) for v,s in scores.items()}
for v,value in point.items():assert abs(value-result['primary_AUC'][v])<1e-15
order=read(D/'bootstrap_groups_FROZEN.json')['groups']
draws=np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws']
np.testing.assert_array_equal(draws,np.random.default_rng(381031).integers(0,64,size=(1000,64),dtype=np.int64))
idx={g:np.flatnonzero((groups['val']==g)&primary) for g in order}
boot={v:np.empty(1000) for v in scores}
for b,draw in enumerate(draws):
    selected=np.concatenate([idx[order[int(i)]] for i in draw])
    for v,s in scores.items():boot[v][b]=auc_reference(labels['val'][selected],s[selected])
for v,value in boot.items():np.testing.assert_allclose(value,stored['bootstrap_'+v],atol=1e-15,rtol=0,equal_nan=True)
delta=boot['FINE']-boot['QUERY'];valid=np.isfinite(delta)
ci=[percentile_reference(delta,.025),percentile_reference(delta,.975)]
np.testing.assert_allclose(ci,result['fine_minus_query_paired_image_CI95'],atol=1e-15,rtol=0)
gates=dict(matched_support=True,fine_auc_at_least_point75=point['FINE']>=.75,
    fine_minus_query_at_least_point05=point['FINE']-point['QUERY']>=.05,paired_image_CI_lower_above_zero=ci[0]>0,
    fine_minus_query_RFF_at_least_point02=point['FINE']-point['QUERY_RFF']>=.02,
    fine_minus_geometry_at_least_point05=point['FINE']-point['GEOMETRY']>=.05,at_least990_valid_bootstrap_draws=int(valid.sum())>=990)
assert gates==result['gates'] and not all(gates.values()) and result['all_gates_passed'] is False
fits=[json.loads(line) for line in (A/'probe_fit_entries.jsonl').read_text().splitlines()]
assert [r['entry'] for r in fits]==list(range(1,8))
out=dict(status='PASS_INDEPENDENT_PROBE_NORMAL_EQUATIONS_SCORES_AND_ALL1000_GROUP_BOOTSTRAPS',
    verified_fitted_models=7,verifier_new_fits=0,normal_equation_max_residuals=normal_residuals,
    max_reconstructed_prediction_error=max_prediction_error,primary_reference_AUC=point,
    fine_minus_query_reference_AUC=point['FINE']-point['QUERY'],CI95=ci,
    bootstrap_references_checked=5000,groups=64,gates=gates,new_author_model_calls=0,mainline_validated=False)
with (D/'independent_metric_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
