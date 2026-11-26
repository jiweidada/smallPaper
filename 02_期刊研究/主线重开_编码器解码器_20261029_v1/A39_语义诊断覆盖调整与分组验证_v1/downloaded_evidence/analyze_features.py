from pathlib import Path
import argparse,hashlib,json,os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from semantic_math import auc,pair_queries,pool_numpy,ridge_fit,ridge_predict,standardize_fit

sys.stdout.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(); parser.add_argument('--evidence',required=True); parser.add_argument('--output',required=True)
args=parser.parse_args(); E=Path(args.evidence); O=Path(args.output); O.mkdir(exist_ok=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def save(name,value):
    with (O/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,default=scalar)
protocol=read(P/'semantic_protocol_FROZEN.json'); collection=read(E/'collection_result.json')
assert collection['status']=='COMPLETE_113_NEW_NATIVE_ENTRIES_WITH_FULL_FEATURE_REPEAT'
assert collection['entries']==collection['completed']==113 and collection['all_first_new_case_arrays_bitwise_repeat_equal']
for name,digest in collection['files_sha256'].items():assert sha(E/name)==digest,name
first_new=read(P/'collector_inputs.json')['cases'][0]
first=dict(np.load(E/('captures/case_%03d.npz'%first_new['index']),allow_pickle=False))
control=dict(np.load(E/'first_control_memory_and_head.npz',allow_pickle=False))
max_fine=0.;max_contrast=0.
for q,box in enumerate(first['query_boxes_xyxy']):
    fine,contrast,count=pool_numpy(control['fine_memory'],box)
    np.testing.assert_allclose(first['fine_features'][q],fine,atol=5e-4,rtol=2e-5)
    np.testing.assert_allclose(first['contrast_features'][q],contrast,atol=5e-4,rtol=2e-5)
    assert int(first['ring_counts'][q])==count
    max_fine=max(max_fine,float(np.max(np.abs(first['fine_features'][q]-fine))))
    max_contrast=max(max_contrast,float(np.max(np.abs(first['contrast_features'][q]-contrast))))
head_ref=first['query_features'].astype(np.float64)@control['head_weight'].astype(np.float64).T+control['head_bias'].astype(np.float64)
np.testing.assert_allclose(head_ref,first['raw_class_logits'],atol=1e-5,rtol=1e-6)
sampling=dict(status='PASS_FIRST_CONTROL_ALL_QUERIES',queries=len(first['query_features']),
    fine_max_abs_error=max_fine,contrast_max_abs_error=max_contrast,
    raw_linear_max_abs_error=float(np.max(np.abs(head_ref-first['raw_class_logits']))),
    scope='Independent NumPy bilinear/pooling and float64 Linear reconstruction; no author model rerun.')
save('control_feature_verification.json',sampling)

views=['QUERY','FINE','CONTRAST','GEOMETRY']
data={s:{v:[] for v in views} for s in ['train','val']}
labels={s:[] for s in ['train','val']}; classes={s:[] for s in ['train','val']}
cases={s:[] for s in ['train','val']}; unsupported={s:[] for s in ['train','val']}; groups={s:[] for s in ['train','val']}
pair_records=[];case_stats=[]
inputs=read(P/'combined_cases_FROZEN.json')
for row in inputs['cases']:
    path=Path(row['capture_path']) if row['capture_phase']=='A38' else E/row['capture_relative']
    if row['capture_phase']=='A38':assert sha(path)==row['capture_sha256']
    c=dict(np.load(path,allow_pickle=False))
    assert int(c['index'])==row['capture_index'] and c['file'].item()==row['file'] and c['split'].item()==row['split']
    pairs,stats,geometry=pair_queries(c); s=row['split']; m=len(pairs)
    case_stats.append(dict(index=row['index'],split=s,file=row['file'],**stats))
    for r in pairs:pair_records.append(dict(case_index=row['index'],split=s,**r))
    if not m:continue
    ids=np.array([[r['positive_query'],r['background_query']] for r in pairs],dtype=int).reshape(-1)
    for v,key in [('QUERY','query_features'),('FINE','fine_features'),('CONTRAST','contrast_features')]:
        data[s][v].append(c[key][ids].astype(np.float64))
    data[s]['GEOMETRY'].append(geometry[ids])
    labels[s].extend([1,0]*m)
    classes[s].extend([r['label'] for r in pairs])
    cases[s].extend([row['index']]*(2*m)); groups[s].extend([row['prefix']]*(2*m))
    unsupported[s].extend(np.repeat([not r['native_support'] for r in pairs],2).tolist())
for s in ['train','val']:
    for v in views:data[s][v]=np.concatenate(data[s][v],0) if data[s][v] else np.empty((0,5 if v=='GEOMETRY' else 256))
    labels[s]=np.array(labels[s],dtype=np.int64);classes[s]=np.array(classes[s],dtype=np.int64)
    cases[s]=np.array(cases[s],dtype=np.int64);unsupported[s]=np.array(unsupported[s],dtype=bool);groups[s]=np.array(groups[s],dtype=str)
primary=unsupported['val']; train_pairs=len(labels['train'])//2;primary_pairs=int(primary.sum())//2
eligibility=dict(train_pairs=train_pairs,primary_val_pairs=primary_pairs,
    train_images_with_pairs=len(np.unique(cases['train'])),
    val_images_with_primary_pairs=len(np.unique(cases['val'][primary])),train_prefixes_with_pairs=len(np.unique(groups['train'])),val_prefixes_with_primary_pairs=len(np.unique(groups['val'][primary])))
passed=(train_pairs>=200 and primary_pairs>=100 and eligibility['train_prefixes_with_pairs']>=32
        and eligibility['val_prefixes_with_primary_pairs']>=20)
save('pair_records.json',dict(rows=pair_records,case_stats=case_stats,eligibility=eligibility,eligible=passed))
if not passed:
    save('semantic_analysis_result.json',dict(status='INSUFFICIENT_MATCHED_SUPPORT_NO_DATA_PROBES_FITTED',
        eligibility=eligibility,gates={'matched_support':False},probe_fit_entries=0,
        mainline_validated=False,new_algorithm_approved=None))
    print(json.dumps(dict(status='INSUFFICIENT_MATCHED_SUPPORT',eligibility=eligibility),default=scalar));sys.exit(0)

fit_log=(O/'probe_fit_entries.jsonl').open('x',encoding='utf-8'); fits=0
def fit(name,x,y):
    global fits
    assert fits<7
    fits+=1;fit_log.write(json.dumps(dict(entry=fits,name=name,rows=len(x),dimensions=x.shape[1]))+'\n');fit_log.flush()
    return ridge_fit(x,y,1.)
parameters={}; scores={}; models={};training_auc={}; secondary_auc={}
y_train=labels['train']*2-1
for v in views:
    models[v]=fit(v,data['train'][v],y_train)
    scores[v]=ridge_predict(models[v],data['val'][v])
    training_auc[v]=auc(labels['train'],ridge_predict(models[v],data['train'][v]))
    secondary_auc[v]=auc(labels['val'],scores[v])
    for key,value in models[v].items():parameters[v+'_'+key]=np.asarray(value)
q_train,q_mean,q_scale=standardize_fit(data['train']['QUERY'])
q_val=(data['val']['QUERY']-q_mean)/q_scale
rng=np.random.default_rng(protocol['rff_seed'])
omega=rng.normal(size=(256,512))/np.sqrt(256.)
phase=rng.uniform(0,2*np.pi,size=512)
z_train=np.sqrt(2/512)*np.cos(q_train@omega+phase)
z_val=np.sqrt(2/512)*np.cos(q_val@omega+phase)
models['QUERY_RFF']=fit('QUERY_RFF512',z_train,y_train)
scores['QUERY_RFF']=ridge_predict(models['QUERY_RFF'],z_val)
training_auc['QUERY_RFF']=auc(labels['train'],ridge_predict(models['QUERY_RFF'],z_train))
secondary_auc['QUERY_RFF']=auc(labels['val'],scores['QUERY_RFF'])
for key,value in models['QUERY_RFF'].items():parameters['QUERY_RFF_'+key]=np.asarray(value)
parameters.update(rff_input_mean=q_mean,rff_input_scale=q_scale,rff_omega=omega,rff_phase=phase)
class_metrics={}
train_classes=classes['train']; val_classes=classes['val'][primary[::2]]
train_counts=np.bincount(train_classes,minlength=11)[1:11]
val_counts=np.bincount(val_classes,minlength=11)[1:11]
supported=(train_counts>=10)&(val_counts>=5)
class_targets=np.eye(10)[train_classes-1]
for v in ['QUERY','FINE']:
    model=fit(v+'_TRUE_CLASS',data['train'][v][::2],class_targets)
    prediction=ridge_predict(model,data['val'][v][::2][primary[::2]]).argmax(1)+1
    correct=prediction==val_classes
    per_class=[float(correct[val_classes==c].mean()) if np.any(val_classes==c) else None for c in range(1,11)]
    macro=float(np.mean([per_class[i] for i in np.flatnonzero(supported)])) if supported.any() else None
    class_metrics[v]=dict(accuracy=float(correct.mean()),supported_macro_accuracy=macro,per_class_accuracy=per_class)
    parameters[v+'_class_prediction']=prediction
    for key,value in model.items():parameters[v+'_CLASS_'+key]=np.asarray(value)
assert fits==7;fit_log.close()
parameters.update(train_classes=train_classes,val_primary_classes=val_classes)
np.savez_compressed(O/'fitted_probe_parameters.npz',**parameters)
point={v:auc(labels['val'][primary],s[primary]) for v,s in scores.items()}
group_order=read(P/'bootstrap_groups_FROZEN.json')['groups']
draws=np.load(P/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws']
assert draws.shape==(1000,64) and len(group_order)==64
indices={g:np.flatnonzero((groups['val']==g)&primary) for g in group_order}
boot={v:np.empty(1000) for v in scores}
for n,draw in enumerate(draws):
    idx=np.concatenate([indices[group_order[int(i)]] for i in draw])
    for v,s in scores.items():boot[v][n]=auc(labels['val'][idx],s[idx])
delta=boot['FINE']-boot['QUERY'];valid=np.isfinite(delta)
ci=np.percentile(delta[valid],[2.5,97.5]).tolist() if valid.any() else [None,None]
gates=dict(matched_support=True,fine_auc_at_least_point75=point['FINE']>=.75,
    fine_minus_query_at_least_point05=point['FINE']-point['QUERY']>=.05,
    paired_image_CI_lower_above_zero=ci[0] is not None and ci[0]>0,
    fine_minus_query_RFF_at_least_point02=point['FINE']-point['QUERY_RFF']>=.02,
    fine_minus_geometry_at_least_point05=point['FINE']-point['GEOMETRY']>=.05,
    at_least990_valid_bootstrap_draws=int(valid.sum())>=990)
np.savez_compressed(O/'paired_probe_scores_and_bootstrap.npz',labels=labels['val'],cases=cases['val'],groups=groups['val'],primary_mask=primary,
    **{'score_'+v:s for v,s in scores.items()},**{'bootstrap_'+v:s for v,s in boot.items()})
result=dict(status='SEMANTIC_ACCESSIBILITY_SCREEN_COMPLETE',primary_scope='Native TopQ lacks correct-classIoU0.5support at score>=0.5; original-area<=1024,unoccluded/untruncated,geometry-coverable and background-matchable. Not all detector false negatives.',
    eligibility=eligibility,probe_fit_entries=fits,primary_AUC=point,training_AUC=training_auc,
    secondary_all_matched_visible_small_AUC=secondary_auc,
    fine_minus_query_AUC=point['FINE']-point['QUERY'],fine_minus_query_paired_image_CI95=ci,
    valid_bootstrap_draws=int(valid.sum()),gates=gates,all_gates_passed=bool(all(gates.values())),
    class_information_check=dict(train_counts=train_counts.tolist(),val_primary_counts=val_counts.tolist(),
        supported_classes=(np.flatnonzero(supported)+1).tolist(),metrics=class_metrics),
    no_author_model_rerun=True,detector_optimizer_updates=0,mainline_validated=False,new_algorithm_approved=None,
    bootstrap_unit='64filename-prefixproxygroupswithallframesandliteralrepeats',
    inference_limit='Accessibility under fixed linear/RFF probes only. RFF is a limited fixed nonlinear control,not all nonlinear decoders; no information-erasure,attention-causality,novelty,compact-ViT or detector-AP conclusion.')
save('semantic_analysis_result.json',result)
print(json.dumps(dict(status=result['status'],eligibility=eligibility,primary_AUC=point,
    fine_minus_query_AUC=result['fine_minus_query_AUC'],CI95=ci,gates=gates,probe_fit_entries=fits),ensure_ascii=False,default=scalar))
