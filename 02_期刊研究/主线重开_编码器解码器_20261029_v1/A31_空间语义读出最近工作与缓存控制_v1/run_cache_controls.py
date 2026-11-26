"""Frozen NumPy-only score-field controls;no Torch,detector or optimizer import."""
from pathlib import Path
import collections,copy,json,math,os,sys,time,traceback
import numpy as np
from cache_components import R,N,A30,D,read,sha,functions,region_functions,raw_original_corners,probabilities,score_flow,topk,prediction_records
sys.stdout.reconfigure(encoding='utf-8')
START=time.monotonic();STAGE='preflight';IMAGES=0;GROUPS=0;FIELDS=0
CFG=read(R/'cache_protocol_FROZEN.json');TH=np.linspace(.5,.95,10)
ENV={'np':np,'THRESHOLDS':TH,'RECALL':np.linspace(0,1,101)}
CORE=functions(A30/'independent_coco_reaggregation.py',['matrix_iou','match_image','category_precision','ap','operating_points'],ENV)
def write(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def guard():assert time.monotonic()-START<CFG['limits']['wall_seconds']
def status(extra=None):
    out={'status':'RUNNING','stage':STAGE,'pid':os.getpid(),'images':IMAGES,'evaluated_groups':GROUPS,'bilinear_level_calls':FIELDS,'elapsed_seconds':time.monotonic()-START,'new_model_calls':0,'optimizer_updates':0}
    if extra:out.update(extra)
    write(R/'status.json',out)
def matches(ds,gs,area,input_area=False):
    """Original-coordinate IoU with explicit per-image input-area override."""
    ds=sorted(ds,key=lambda x:-x['score'])[:500]
    def ga(g):return g['_area_input'] if input_area else g['area']
    def da(d):return d['bbox'][2]*d['bbox'][3]*(d['_ratio']**2 if input_area else 1.)
    gs=sorted(gs,key=lambda g:bool(g.get('iscrowd',0)) or ga(g)<area[0] or ga(g)>area[1]);gi=np.asarray([bool(g.get('iscrowd',0)) or ga(g)<area[0] or ga(g)>area[1] for g in gs],dtype=bool);crowd=np.asarray([bool(g.get('iscrowd',0)) for g in gs],dtype=bool)
    overlap=CORE['matrix_iou'](ds,gs,crowd);dm=np.zeros((10,len(ds)),dtype=np.int64);di=np.zeros((10,len(ds)),dtype=bool);gm=np.zeros((10,len(gs)),dtype=bool)
    for ti,threshold in enumerate(TH):
        for j,d in enumerate(ds):
            best=-1;level=min(threshold,1-1e-10)
            for k,g in enumerate(gs):
                if gm[ti,k] and not crowd[k]:continue
                if best>=0 and not gi[best] and gi[k]:break
                if overlap[j,k]<level:continue
                level=overlap[j,k];best=k
            if best>=0:dm[ti,j]=gs[best]['id'];di[ti,j]=gi[best];gm[ti,best]=True
    outside=np.asarray([da(d)<area[0] or da(d)>area[1] for d in ds],dtype=bool);di|=(dm==0)&outside[None,:]
    duplicate=(dm[0]==0)&~di[0]&(np.max(overlap,axis=1)>=.5 if len(gs) else np.zeros(len(ds),dtype=bool))
    return {'scores':np.asarray([d['score'] for d in ds]),'ids':np.asarray([d['_id'] for d in ds]),'matches':dm,'ignore':di,'gt_ignore':gi,'duplicate_FP':duplicate}
def metric_preflight():
    gs=[dict(id=1,bbox=[10.,10.,40.,40.],area=1600.,_area_input=400.,iscrowd=0),dict(id=2,bbox=[80.,50.,10.,10.],area=100.,_area_input=25.,iscrowd=0)]
    ds=[dict(_id=1,bbox=[10.,10.,40.,40.],score=.9,_ratio=.5),dict(_id=2,bbox=[80.,50.,10.,10.],score=.8,_ratio=.5),dict(_id=3,bbox=[80.,50.,10.,10.],score=.7,_ratio=.5),dict(_id=4,bbox=[100.,100.,50.,50.],score=.6,_ratio=.5)]
    for input_area in [False,True]:
        g=copy.deepcopy(gs);d=copy.deepcopy(ds)
        if input_area:
            for x in g:x['bbox']=[v*.5 for v in x['bbox']];x['area']=x['_area_input']
            for x in d:x['bbox']=[v*.5 for v in x['bbox']]
        for area in [[0,1e10],[0,math.nextafter(1024.,-math.inf)]]:
            reference=CORE['match_image'](d,g,area);got=matches(ds,gs,area,input_area)
            for key in ['scores','ids','matches','ignore','gt_ignore']:assert np.array_equal(reference[key],got[key]),key
    return True
def duplicate_at_threshold(records,images,cats,threshold):
    return sum(int(np.sum(r['duplicate_FP']&(r['scores']>=threshold))) for c in cats for im in images for r in [records[(c,im)]]) if threshold is not None else 0
def main():
    global STAGE,IMAGES,GROUPS,FIELDS
    assert not (R/'cache_result.json').exists() and not (R/'status.json').exists()
    with (R/'one_shot.lock').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'protocol_sha256':sha(R/'cache_protocol_FROZEN.json'),'runner_sha256':sha(__file__)},f)
    for name,h in CFG['input_hashes'].items():
        p=R/name if name in ['raw_annotations_verified.json','cache_components.py'] else {'A30_capture_metadata':D/'capture_metadata.json','A30_result':D/'result.json','A27_region_helper':N/'A27_查询扩容同预算任务诊断_v1'/'o27_metrics.py','A30_independent_metric_helper':A30/'independent_coco_reaggregation.py'}[name]
        assert sha(p)==h,name
    assert metric_preflight();region=region_functions();raw=read(R/'raw_annotations_verified.json')['cases'];a30cfg=read(A30/'native_dual_head_protocol_FROZEN.json');native=read(D/'Obox_Oscore_TOPK_predictions_original.json');mref=read(D/'Mbox_Mscore_NMS_predictions_original.json');native_by=collections.defaultdict(list);m_by=collections.defaultdict(list)
    for x in native:native_by[x['image_id']].append(x)
    for x in mref:m_by[x['image_id']].append(x)
    captured=read(D/'capture_metadata.json');canonical=read(Path(a30cfg['local_raw_root'])/'dataset/diagnostic_gt_orig.json');uidmap={g['stable_uid']:g for g in canonical['annotations'] if not g['iscrowd']}
    names=[g['id'] for g in CFG['groups']];assert len(names)==9;outputs={n:[] for n in names};annotations=[];counts=[];gt_counts=[];clamps=collections.defaultdict(int)
    STAGE='cache_transport';status()
    for case,cm in zip(raw,captured):
        guard();rows=np.asarray([r['values'] for r in case['rows']],dtype=float);integral=region['ignore_integral'](rows,case['height'],case['width']);selected=region['region_keep'](rows[:,:4],integral);valid=(rows[:,4]>0)&(rows[:,5]>=1)&(rows[:,5]<=10);keepGT=valid&selected;s=case['geometry']['ratio'];dropped=[]
        for i,record in enumerate(case['rows']):
            if not valid[i]:continue
            uid=case['file_name']+':'+str(record['source_row'])
            if not keepGT[i]:dropped.append(uid);continue
            old=uidmap[uid];g=dict(id=old['id'],image_id=case['image_id'],category_id=int(rows[i,5])-1,bbox=rows[i,:4].tolist(),area=float(rows[i,2]*rows[i,3]),iscrowd=0,stable_uid=uid,_area_input=float(rows[i,2]*rows[i,3]*s*s));annotations.append(g)
        gt_counts.append({'image_id':case['image_id'],'raw_valid_GT':int(valid.sum()),'kept_GT':int(keepGT.sum()),'dropped_uids':dropped,'class0_region_rectangles':int((rows[:,5]==0).sum())})
        current={'M_NATIVE_NMS_REFERENCE':copy.deepcopy(m_by[case['image_id']]),'O_CACHED_NATIVE':copy.deepcopy(native_by[case['image_id']])}
        with np.load(D/'captures'/('case_%03d.npz'%case['index']),allow_pickle=False) as data:
            corners=raw_original_corners(data['O_decoded_xyxy'],case);prob=probabilities(data['O_scores']);flows={}
            for alpha in [0.,1.,-1.,.5,-.5]:
                flows[alpha],clamp=score_flow(prob,data['anchors_px'],corners,case,[8,16,32],[[80,80],[40,40],[20,20]],alpha);FIELDS+=3;clamps[str(alpha)]+=clamp
            assert np.array_equal(flows[0.],prob)
            for group in CFG['groups']:
                if 'alpha' not in group:continue
                name=group['id'];scores=flows[group['alpha']]
                if group['pool']=='all8400':
                    a,c,p=topk(scores);current[name]=prediction_records(case['image_id'],corners,a,c,p)
                else:
                    records=copy.deepcopy(native_by[case['image_id']]);ids=cm['lineage']['Obox_Oscore_TOPK'];assert len(ids)==len(records)
                    for record,idx in zip(records,ids):record['score']=float(scores[record['category_id'],idx])
                    current[name]=sorted(records,key=lambda x:-x['score'])
            for name,records in current.items():
                before=len(records);assert before<=500;keep=region['region_keep']([x['bbox'] for x in records],integral);chosen=[x for x,k in zip(records,keep) if k];outputs[name].extend(chosen)
                counts.append({'image_id':case['image_id'],'group':name,'before_region':before,'dropped_by_region':int((~keep).sum()),'after_region':len(chosen)})
            assert len(current['CENTER_KEEP_POOL'])==len(current['O_CACHED_NATIVE'])==len(current['REVERSE_KEEP_POOL'])
        IMAGES+=1
        if IMAGES%20==0:status();print(json.dumps({'images':IMAGES,'elapsed':time.monotonic()-START}),flush=True)
    assert FIELDS==CFG['limits']['bilinear_level_evaluations'] and IMAGES==100
    gt={'images':canonical['images'],'categories':canonical['categories'],'annotations':annotations};write(R/'region_policy_GT.json',gt);write(R/'region_case_counts.json',counts);write(R/'region_GT_filter_audit.json',gt_counts)
    (R/'predictions').mkdir(exist_ok=False)
    for name,records in outputs.items():write(R/'predictions'/(name+'.json'),records)
    cats=list(range(10));images=sorted(x['id'] for x in gt['images']);gg=collections.defaultdict(list)
    for g in annotations:gg[(g['category_id'],g['image_id'])].append(g)
    small={g['id'] for g in annotations if g['_area_input']<1024};metrics={};bootstrap_records={};all_records={};STAGE='metrics';status()
    ratios={c['image_id']:c['geometry']['ratio'] for c in raw}
    for name in names:
        guard();pred=outputs[name];dp=collections.defaultdict(list)
        for i,d in enumerate(pred):d['_id']=i+1;d['_ratio']=ratios[d['image_id']];dp[(d['category_id'],d['image_id'])].append(d)
        ar={(c,im):matches(dp[(c,im)],gg[(c,im)],[0,1e10]) for c in cats for im in images}
        osmall={(c,im):matches(dp[(c,im)],gg[(c,im)],[0,math.nextafter(1024.,-math.inf)]) for c in cats for im in images}
        ismall={(c,im):matches(dp[(c,im)],gg[(c,im)],[0,math.nextafter(1024.,-math.inf)],True) for c in cats for im in images}
        ap,ap50=CORE['ap'](ar,images,cats);apso,_=CORE['ap'](osmall,images,cats);apsi,_=CORE['ap'](ismall,images,cats)
        matched=set();fp=0
        for rec in ar.values():
            take=(rec['scores']>=.25)&~rec['ignore'][0];fp+=int(np.sum(take&(rec['matches'][0]==0)));matched.update(rec['matches'][0,take&(rec['matches'][0]>0)].tolist())
        metrics[name]={'AP_original_all':ap,'AP50_original_all':ap50,'AP_original_small':apso,'AP_input_small':apsi,'at_conf025':{'FP':fp,'TP':len(matched),'smallTP':len(matched&small)},'retained_detections':len(pred)};all_records[name]=ar
        if name in ['O_CACHED_NATIVE','ZERO_GRID','CENTER_FLOW','REVERSE_FLOW','CENTER_KEEP_POOL']:bootstrap_records[name]=ismall
        GROUPS+=1;status();print(json.dumps({'evaluated_group':name,'AP_inputsmall':apsi,'elapsed':time.monotonic()-START}),flush=True)
    budget=metrics['O_CACHED_NATIVE']['at_conf025']['FP'];budgets=sorted(set(CFG['metrics']['secondary_FP_budgets']+[budget]))
    for name,records in all_records.items():
        operating=CORE['operating_points'](records,images,cats,small,budgets)
        for point in operating:point['duplicate_FP']=duplicate_at_threshold(records,images,cats,point['global_score_threshold'])
        metrics[name]['same_FP']=operating
    tolerance=CFG['zero_fidelity'];baseline=metrics['O_CACHED_NATIVE'];zero=metrics['ZERO_GRID']
    assert max(abs(baseline[k]-zero[k]) for k in ['AP_original_all','AP50_original_all','AP_original_small','AP_input_small'])<=tolerance['max_native_AP_abs_diff']
    op=lambda n:next(x for x in metrics[n]['same_FP'] if x['FP_budget']==budget)
    assert abs(op('O_CACHED_NATIVE')['small_TP']-op('ZERO_GRID')['small_TP'])<=tolerance['max_small_TP_fixed_FP_diff']
    bc=CFG['metrics']['bootstrap'];rng=np.random.default_rng(bc['seed']);draws=rng.integers(0,100,size=(bc['draws'],100));samples={a+'__minus__'+b:[] for a,b in bc['contrasts']};STAGE='bootstrap';status()
    for i,draw in enumerate(draws):
        guard();values={name:CORE['ap'](rec,images,cats,draw)[0] for name,rec in bootstrap_records.items()}
        for a,b in bc['contrasts']:samples[a+'__minus__'+b].append(values[a]-values[b])
        if (i+1)%200==0:status({'bootstrap_completed':i+1});print(json.dumps({'bootstrap_completed':i+1,'elapsed':time.monotonic()-START}),flush=True)
    cis={name:np.quantile(v,[.025,.975]).tolist() for name,v in samples.items()};np.savez_compressed(R/'bootstrap_samples.npz',draws=draws,**{k:np.asarray(v) for k,v in samples.items()})
    g=CFG['gates'];gates={'inputsmall_AP_gain':metrics['CENTER_FLOW']['AP_input_small']-zero['AP_input_small']>=g['CENTER_FLOW_inputsmall_AP_gain_min'],'all_AP_no_material_loss':metrics['CENTER_FLOW']['AP_original_all']-zero['AP_original_all']>=g['CENTER_FLOW_all_AP_gain_min'],'directional_specificity':metrics['CENTER_FLOW']['AP_input_small']-metrics['REVERSE_FLOW']['AP_input_small']>=g['CENTER_minus_REVERSE_inputsmall_AP_min'],'same_FP_smallTP_gain':op('CENTER_FLOW')['small_TP']-op('ZERO_GRID')['small_TP']>=g['CENTER_FLOW_smallTP_gain_at_primary_FP_min'],'fixed_pool_nonnegative_AP':metrics['CENTER_KEEP_POOL']['AP_input_small']-baseline['AP_input_small']>=g['CENTER_KEEP_POOL_inputsmall_AP_gain_min'],'bootstrap_center_vs_zero_positive':cis['CENTER_FLOW__minus__ZERO_GRID'][0]>0,'bootstrap_center_vs_reverse_positive':cis['CENTER_FLOW__minus__REVERSE_FLOW'][0]>0}
    assert GROUPS==CFG['limits']['evaluation_groups']
    outcome={'phase':'A31','status':'CACHED_SCORE_TRANSPORT_DIAGNOSTIC_COMPLETED','metric_policy':CFG['region_policy'],'images':IMAGES,'evaluation_groups':GROUPS,'bilinear_level_calls':FIELDS,'valid_GT_before_region':4491,'valid_GT_after_region':len(annotations),'small_input_GT_after_region':len(small),'primary_FP_budget':budget,'metrics':metrics,'bootstrap95':cis,'gates':gates,'mechanism_gate_pass':all(gates.values()),'new_algorithm_or_mainline_validated':False,'new_model_calls':0,'new_optimizer_updates':0,'boundary_clamp_counts':dict(clamps),'elapsed_seconds':time.monotonic()-START}
    write(R/'cache_result.json',outcome);STAGE='complete';status({'status':'COMPLETED'});print(json.dumps({'status':outcome['status'],'primary_FP_budget':budget,'GT_after_region':len(annotations),'gates':gates,'new_model_calls':0,'elapsed':time.monotonic()-START}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(R/'failure.json',{'status':'FAILED_STOPPED_NO_RETRY','stage':STAGE,'images':IMAGES,'groups':GROUPS,'error':repr(error),'traceback':traceback.format_exc(),'new_model_calls':0});traceback.print_exc();sys.exit(1)
