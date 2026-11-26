from pathlib import Path
import collections,importlib.util,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('a36_analysis_helpers',D/'analyze_scores.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);_,h=a.helpers()
cfg=read(D/'score_protocol_FROZEN.json');result=read(D/'score_analysis_result.json');reference=read(D/'coco_reference_result.json')
assert reference['metric_engine_calls']==8 and reference['learned_component_calls']==0
gt=read(D/'filtered_GT.json');gg=collections.defaultdict(list)
for g in gt['annotations']:gg[(g['category_id'],g['image_id'])].append(g)
small={g['id'] for g in gt['annotations'] if g['area']<=1024}
drawfile=np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False);draws=drawfile['draws']
assert np.array_equal(draws,np.random.default_rng(cfg['metric']['bootstrap']['seed']).integers(0,12,size=(1000,12)))
boot=np.load(D/'paired_score_bootstrap.npz',allow_pickle=False);records={};array_checks=0;metric_error=0.
for arm in cfg['arms']:
    pred=read(D/(arm+'_filtered_predictions.json'));group=collections.defaultdict(list)
    for i,p in enumerate(pred):group[(p['category_id'],p['image_id'])].append(dict(p,_id=i+1))
    sets={area:{(cat,im):h['match_image'](group[(cat,im)],gg[(cat,im)],limits) for cat in range(1,11) for im in range(1,13)}
          for area,limits in [('all',[0,1e10]),('small',[0,1024])]}
    records[arm]=sets
    expected={(r['area_index'],r['category_id'],r['image_id']):r for r in reference['base_eval_records'][arm]}
    for area_index,area in [(0,'all'),(1,'small')]:
        for (cat,im),r in sets[area].items():
            other=expected.get((area_index,cat,im))
            if other is None:
                assert len(r['scores'])==0 and len(r['gt_ignore'])==0;continue
            for key in ['scores','matches','ignore','gt_ignore']:
                got=np.asarray(r[key]);want=np.asarray(other[key],dtype=got.dtype)
                if want.size==0:want=want.reshape(got.shape)
                assert np.array_equal(got,want),(arm,area,cat,im,key);array_checks+=1
    got=a.metrics(sets,list(range(1,13)),h)
    for key,value in got.items():
        error=abs(value-reference['references'][arm]['-1'][key]);metric_error=max(metric_error,error);assert error<1e-12
        assert value==result['point'][arm][key]
for row in [0,499,999]:
    metrics={arm:a.metrics(records[arm],draws[row]+1,h) for arm in cfg['arms']}
    for arm in cfg['arms']:
        for key,value in metrics[arm].items():
            error=abs(value-reference['references'][arm][str(row)][key]);metric_error=max(metric_error,error);assert error<1e-12
    for key in ['AP','APsmall','AP50','AP75','APsmall75']:
        assert abs(metrics['RAW_CLASS'][key]-metrics['NATIVE'][key]-boot[key][row])<1e-12
# Prefix arithmetic is independently recomputed from pycocotools' base matches.
def scalar_prefix(arm,w):
    events=[]
    for r in reference['base_eval_records'][arm]:
        if r['area_index']!=0:continue
        image=r['image_id']-1;count=int(w[image])
        for j,score in enumerate(r['scores']):
            if r['ignore'][0][j]:continue
            gid=int(r['matches'][0][j]);events.append((score,count*int(gid==0),count*int(gid>0),count*int(gid in small)))
    events.sort(key=lambda x:-x[0]);blocks=[]
    for score,f,t,s in events:
        if not blocks or blocks[-1][0]!=score:blocks.append([score,0,0,0])
        blocks[-1][1]+=f;blocks[-1][2]+=t;blocks[-1][3]+=s
    fp=tp=stp=0;cut=None
    for score,f,t,s in blocks:
        if fp+f>111:break
        fp+=f;tp+=t;stp+=s;cut=score
    return dict(actual_FP=fp,TP=tp,small_TP=stp,threshold=cut)
for column in [-1,0,499,999]:
    w=np.ones(12,dtype=np.int64) if column==-1 else np.bincount(draws[column],minlength=12)
    values={arm:scalar_prefix(arm,w) for arm in cfg['arms']}
    if column==-1:
        for arm in cfg['arms']:assert values[arm]==result['point'][arm]['fixed_FP']
    else:
        for k in ['small_TP','TP']:assert values['RAW_CLASS'][k]-values['NATIVE'][k]==boot[k][column]
for key in boot.files:assert np.allclose(np.quantile(boot[key],[.025,.975]),result['paired_image_intervals'][key],rtol=0,atol=1e-15)
out=dict(status='PASS_ALL_BASE_COCO_MATCH_ARRAYS_TWO_ARMS_AND3_BOOTSTRAP_REFERENCES',
         matched_arrays_checked=array_checks,metric_max_abs_error=metric_error,paired_bootstrap_seed_draws_checked=1000,
         bootstrap_metric_draws_independently_evaluated=[0,499,999],fixed_FP_columns_checked=[-1,0,499,999],new_model_calls=0,
         scope='Independentpycocotoolsmatching/precisionreference;notnoveltyorindependentdatareplication.')
with (D/'independent_score_metric_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out))
