"""Pure NumPy COCO-style matching and paired image-bootstrap for two frozen score arms."""
from pathlib import Path
import ast,collections,hashlib,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def functions(path,names,globals_):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(defs)==len(names)
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(path),'exec'),globals_)
    return globals_
def helpers():
    h=functions(D/'pinned_o27_metrics.py',['matlab_round','ignore_integral','region_keep','prepare_cases'],{'np':np})
    n=functions(D/'pinned_A30_independent.py',['matrix_iou','match_image','category_precision'],
        {'np':np,'THRESHOLDS':np.linspace(.5,.95,10),'RECALL':np.linspace(0,1,101)})
    return h,n
def metrics(records,chosen,n):
    means={}
    for area in ['all','small']:
        ps=[n['category_precision']([records[area][(cat,int(im))] for im in chosen]) for cat in range(1,11)]
        ps=[p for p in ps if p is not None]
        means[area]=np.mean(ps,axis=0) if ps else np.zeros((10,101))
    return dict(AP=float(means['all'].mean()),APsmall=float(means['small'].mean()),AP50=float(means['all'][0].mean()),
                AP75=float(means['all'][5].mean()),APsmall75=float(means['small'][5].mean()))
def fixed_fp(records,weights,small_ids,budget=111):
    events=[]
    for (cat,image),r in records['all'].items():
        w=int(weights[image-1])
        if not w:continue
        for j,s in enumerate(r['scores']):
            if not r['ignore'][0,j]:events.append((float(s),w*int(r['matches'][0,j]==0),w*int(r['matches'][0,j]>0),w*int(r['matches'][0,j] in small_ids)))
    events.sort(key=lambda x:-x[0]);groups=[]
    for score,fp,tp,small in events:
        if not groups or groups[-1][0]!=score:groups.append([score,0,0,0])
        groups[-1][1]+=fp;groups[-1][2]+=tp;groups[-1][3]+=small
    used=tp=small=0;threshold=None
    for score,f,t,s in groups:
        if used+f>budget:break
        used+=f;tp+=t;small+=s;threshold=score
    return dict(actual_FP=used,TP=tp,small_TP=small,threshold=threshold)
def fixture():
    _,n=helpers();gt=[dict(id=1,bbox=[0.,0.,10.,10.],area=100.,iscrowd=0)]
    ds=[dict(_id=1,bbox=[0.,0.,10.,10.],score=.9),dict(_id=2,bbox=[30.,30.,10.,10.],score=.8)]
    a=n['category_precision']([n['match_image'](ds,gt,[0,1024])]);assert np.allclose(a,np.ones((10,101)),rtol=0,atol=1e-15)
    ds[0]['score']=.8;ds[1]['score']=.9
    b=n['category_precision']([n['match_image'](ds,gt,[0,1024])]);assert np.allclose(b,np.full((10,101),.5),rtol=0,atol=1e-15)
    c=np.array([-1.,0.,2.]);q=.25;g=1.5
    assert np.array_equal(c+q,(c+g)+(q-g))
    return dict(status='PASS_ANALYTIC_RANKING_AP_AND_ADDITIVE_GAUGE_FIXTURES',new_model_calls=0)
def main():
    cfg=read(D/'score_protocol_FROZEN.json');manifest=read(D/'execution_inputs_manifest.json')
    assert hashlib.sha256((D/'analyze_scores.py').read_bytes()).hexdigest()==manifest['analysis_sha256']
    assert read(D/'archive_verification.json')['status']=='PASS_SCORE_ARCHIVE_AND_LINEAR_PROJECTION'
    h,n=helpers();cases=h['prepare_cases'](read(D/'raw_annotations_verified.json')['cases'])
    assert len(cases)==12
    gt=dict(info={},images=[dict(id=c['id'],width=c['width'],height=c['height']) for c in cases],
            categories=[dict(id=i,name=str(i)) for i in range(1,11)],annotations=[])
    for c in cases:
        for b,label in zip(c['gt_xywh'],c['gt_labels']):
            gt['annotations'].append(dict(id=len(gt['annotations'])+1,image_id=c['id'],category_id=int(label),bbox=b.tolist(),area=float(b[2]*b[3]),iscrowd=0))
    save(D/'filtered_GT.json',gt)
    grouped_gt=collections.defaultdict(list)
    for g in gt['annotations']:grouped_gt[(g['category_id'],g['image_id'])].append(g)
    small_ids={g['id'] for g in gt['annotations'] if g['area']<=1024}
    raw=read(D/'downloaded_evidence/predictions_without_region_filter.json');arm_sets={};point={}
    for arm in cfg['arms']:
        records=[]
        for c,p in zip(cases,raw[arm]):
            assert c['file']==p['file']
            keep=h['region_keep'](p['boxes'],c['integral'])
            for j,(b,label,score) in enumerate(zip(p['boxes'],p['labels'],p['scores'])):
                if keep[j]:records.append(dict(image_id=c['id'],category_id=int(label),bbox=b,score=float(score)))
        save(D/(arm+'_filtered_predictions.json'),records)
        grouped=collections.defaultdict(list)
        for i,r in enumerate(records):r=dict(r,_id=i+1);grouped[(r['category_id'],r['image_id'])].append(r)
        sets={area:{(cat,c['id']):n['match_image'](grouped[(cat,c['id'])],grouped_gt[(cat,c['id'])],limits)
               for cat in range(1,11) for c in cases} for area,limits in [('all',[0,1e10]),('small',[0,1024])]}
        arm_sets[arm]=sets;point[arm]=metrics(sets,list(range(1,13)),n)
        point[arm]['fixed_FP']=fixed_fp(sets,np.ones(12,dtype=np.int64),small_ids)
    assert point['NATIVE']['fixed_FP']['actual_FP']==111 and point['NATIVE']['fixed_FP']['small_TP']==382
    assert point['NATIVE']['fixed_FP']['TP']==621
    draws=np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['draws'];assert draws.shape==(1000,12)
    samples={k:[] for k in ['AP','APsmall','AP50','AP75','APsmall75','small_TP','TP']}
    for draw in draws:
        chosen=draw+1;weights=np.bincount(draw,minlength=12);values={}
        for arm in cfg['arms']:
            values[arm]=metrics(arm_sets[arm],chosen,n)
            values[arm].update(fixed_fp(arm_sets[arm],weights,small_ids))
        for k in samples:samples[k].append(values['RAW_CLASS'][k]-values['NATIVE'][k])
    delta={k:point['RAW_CLASS'][k]-point['NATIVE'][k] for k in ['AP','APsmall','AP50','AP75','APsmall75']}
    delta.update({k:point['RAW_CLASS']['fixed_FP'][k]-point['NATIVE']['fixed_FP'][k] for k in ['small_TP','TP']})
    intervals={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
    gates=dict(APsmall_gain_atleast_0005=delta['APsmall']>=.005,APsmall_CI_lower_positive=intervals['APsmall'][0]>0,
               AP_no_material_loss=delta['AP']>=-.002,APsmall75_no_material_loss=delta['APsmall75']>=-.002,
               same_FP_small_TP_no_loss=delta['small_TP']>=0)
    np.savez_compressed(D/'paired_score_bootstrap.npz',**{k:np.asarray(v) for k,v in samples.items()})
    out=dict(status='COMPLETED_FROZEN_TWO_SCORE_ARM_DIAGNOSIS',point=point,RAW_minus_NATIVE=delta,paired_image_intervals=intervals,
             gates=gates,quality_harm_question_promoted=all(gates.values()),new_algorithm_validated=False,limits=cfg['interpretation_limits'])
    save(D/'score_analysis_result.json',out)
    print(json.dumps({k:out[k] for k in ['status','point','RAW_minus_NATIVE','paired_image_intervals','gates','quality_harm_question_promoted']},ensure_ascii=False))
if __name__=='__main__':
    if '--fixture' in sys.argv:print(json.dumps(fixture()))
    else:main()
