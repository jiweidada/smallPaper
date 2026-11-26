"""Independent arithmetic checks using patch matrices, not the analysis helper."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent/'A16_实际特征空间变化审计_v1'
assert not (ROOT/'independent_response_verification.json').exists()
load=lambda name:json.loads((ROOT/name).read_text(encoding='utf-8'))
protocol=load('protocol.json');summary=load('analysis_summary.json')
rows=load('per_GT_response.json');pairs=load('matched_geometry_pairs.json');images=load('per_image_response.json')
byrow={(r['case'],r['row']):r for r in rows};max_error=0.;checked=0
def sample_one(data,index,dx,dy):
    xy=data['local_coords'][index].reshape(-1,2)+np.asarray([dx,dy])*np.tile(data['point_wh'],(8,1))/800.
    a=np.floor(xy).astype(int);frac=xy-a
    blocks=data['patch'][index].reshape(-1,3,3,32).astype(np.float64)
    result=np.zeros((128,32),dtype=np.float64)
    for x in [0,1]:
        for y in [0,1]:
            weight=(frac[:,0] if x else 1-frac[:,0])*(frac[:,1] if y else 1-frac[:,1])
            result+=weight[:,None]*blocks[np.arange(128),a[:,1]+y,a[:,0]+x]
    return result.reshape(8,16,32)
for cohort,count in [('old',12),('new',25)]:
    for i in range(count):
        case=cohort+('%02d'%i)
        with np.load(ROOT/(cohort+'_patches')/('case%02d.npz'%i),allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(ROOT/'response_matrices'/(case+'.npz'),allow_pickle=False) as z:original={k:z[k] for k in z.files}
        for j in range(len(data['query_indices'])):
            dx=(sample_one(data,j,.125,0)-sample_one(data,j,-.125,0))/.25
            dy=(sample_one(data,j,0,.125)-sample_one(data,j,0,-.125))/.25
            d=np.stack((dx,dy),-1);w=data['attention_weights'][j]
            stack=(np.sqrt(w)[...,None,None]*d).reshape(-1,2)
            total=(w[...,None,None]*d).sum(1).reshape(-1,2)
            levels=[]
            for h in range(8):
                for l in range(4):
                    ww=w[h,l*4:l*4+4];mass=ww.sum()
                    levels.append((ww[:,None,None]*d[h,l*4:l*4+4]).sum(0)/np.sqrt(mass) if mass else np.zeros((32,2)))
            level=np.concatenate(levels,axis=0)
            grams={'G_stack':stack.T@stack,'G_level':level.T@level,'G_pool':total.T@total}
            for name,g in grams.items():
                error=float(np.max(np.abs(g-original[name][j])));max_error=max(max_error,error)
                assert np.allclose(g,original[name][j],atol=1e-9,rtol=1e-10),(case,j,name,error)
            ss=float(np.trace(grams['G_stack']));pp=float(np.trace(grams['G_pool']))
            cancellation=max(0.,min(1.,1-pp/ss)) if ss>0 else 0.
            row=byrow[case,j];assert abs(cancellation-row['cancellation'])<1e-11
            assert bool(ss==0)==row['zero_response'];checked+=1
assert checked==len(rows)==853 and len(images)==37
reaggregated={}
for cohort,seed in [('all',protocol['bootstrap_seed']),('new',protocol['bootstrap_seed']+1)]:
    pp=[p for p in pairs if cohort=='all' or p['cohort']==cohort]
    casekeys=[i['case'] for i in images if (cohort=='all' or i['cohort']==cohort) and i['pairs']]
    means=[]
    for key in casekeys:
        ps=[p for p in pp if p['case']==key]
        values=[byrow[key,p['bad_row']]['cancellation']-byrow[key,p['good_row']]['cancellation'] for p in ps]
        means.append(sum(values)/len(values))
    a=np.asarray(means);rng=np.random.default_rng(seed)
    bootstrap=np.asarray([a[rng.integers(0,len(a),size=len(a))].mean() for _ in range(2000)])
    stats={'mean':float(a.mean()),'low':float(np.quantile(bootstrap,.025)),'high':float(np.quantile(bootstrap,.975))}
    original=summary['statistics'][cohort]
    for k,v in stats.items():assert abs(v-original['statistics']['cancellation'][k])<1e-12
    counts={'images_with_pairs':len(casekeys),'pairs':len(pp),
        'distinct_good':len({(p['case'],p['good_gt']) for p in pp}),
        'distinct_bad':len({(p['case'],p['bad_gt']) for p in pp})}
    assert counts==original['counts']
    assert all(counts[k]>=n for k,n in protocol['minimum_data'][cohort].items())
    assert not (stats['mean']>=.10 and stats['low']>0)
    reaggregated[cohort]={'counts':counts,'statistics':stats}
assert summary['decision']=='NO_SUFFICIENT_ERROR_SPECIFIC_CANCELLATION_SIGNAL_NO_MODULE_TRAINING'
assert not summary['gate_passed'] and not summary['mainline_validated']
assert load('process/a16_spatial_response_v1_exit.json')['exit_code']==0
for f in load('freeze_manifest.json')['files']:
    assert hashlib.sha256((ROOT/f['file']).read_bytes()).hexdigest()==f['sha256']
for f in load('old_patch_packing_receipt.json')['files']:
    assert hashlib.sha256((ROOT/f['patch_file']).read_bytes()).hexdigest()==f['patch_sha256']
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_INDEPENDENT_MATRIX_AND_IMAGE_BOOTSTRAP_ARITHMETIC',
    'GT_rows':checked,'Gram_matrices':3*checked,'max_absolute_Gram_difference':max_error,
    'cohorts':reaggregated,'actual_capture_exit_code':0,'main_model_updates':0,
    'new_complete_model_forwards':0,'new_suffix_forwards':0,'new_source_core_calls':0,
    'scope':'Independent NumPy patch finite-difference/Gram and aggregate arithmetic,not another model capture or independent-data replication.',
    'interpretation':'Evidence does not meet prespecified error-specific cancellation screen;does not prove absent encoder information or rule out all alternate feature operators.'}
(ROOT/'independent_response_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['status','GT_rows','Gram_matrices','max_absolute_Gram_difference','new_complete_model_forwards']}))
