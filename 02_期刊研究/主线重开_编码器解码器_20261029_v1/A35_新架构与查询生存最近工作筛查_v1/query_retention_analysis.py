"""Frozen native-support attrition statistic. This does not simulate decoder pruning."""
from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def prefix_threshold(scores,fp,image,weights,budget):
    """Accept complete equal-score blocks in a single global prefix."""
    if not len(scores):return float('inf'),0
    order=np.argsort(-scores,kind='stable')
    s=scores[order];cost=fp[order]*weights[image[order]]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    sums=np.cumsum(cost,dtype=np.int64)[ends]
    allowed=np.flatnonzero(sums<=budget)
    if not len(allowed):return float('inf'),0
    j=int(allowed[-1]);return float(s[ends[j]]),int(sums[j])

def fixtures():
    s=np.array([.9,.8,.8,.7,.6]);fp=np.array([0,1,1,0,1]);im=np.zeros(5,dtype=np.int64)
    assert prefix_threshold(s,fp,im,np.array([1]),1)==(.9,0)
    assert prefix_threshold(s,fp,im,np.array([1]),2)==(.7,2)
    assert prefix_threshold(s,fp,im,np.array([2]),2)==(.9,0)
    assert not (.8 < .8), 'A surviving equal-score backup must prevent support attrition'
    assert (np.array([True,False]).mean()-np.array([False]).mean())==.5
    return dict(status='PASS_ANALYTIC_TIES_WEIGHTED_FP_AND_BACKUP_FIXTURES',model_calls=0)

def helper_functions():
    path=D/'pinned_o27_metrics.py';mod=ast.parse(path.read_text(encoding='utf-8'))
    names=['matlab_round','ignore_integral','region_keep','prepare_cases','iou_xywh']
    functions=[n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==len(names)
    ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
    return ns

def summarize_rows(rows,weights,threshold,depth):
    protected=rows['native_TP_score']>=threshold
    w=weights[rows['image']]*protected
    lost=rows[f'best_retained_score_depth{depth}']<threshold
    result={}
    for name,mask in [('small',rows['small']),('larger',~rows['small'])]:
        n=int(w[mask].sum());count=int((w[mask]*lost[mask]).sum())
        result[name]=dict(native_supported_GT=n,attrited_GT=count,rate=(count/n if n else None),
                          images=int(len(np.unique(rows['image'][mask&(w>0)]))))
    a=result['small']['rate'];b=result['larger']['rate']
    result['small_minus_larger']=(a-b if a is not None and b is not None else None)
    # Compare within image, class and final native confidence bin; this is predictive adjustment, not causal control.
    sid=(rows['image']*13+rows['label'])*5+rows['final_score_bin']
    small=rows['small'];larger=~small;length=12*13*5
    ns=np.bincount(sid[small],weights=w[small],minlength=length)
    nl=np.bincount(sid[larger],weights=w[larger],minlength=length)
    ls=np.bincount(sid[small],weights=w[small]*lost[small],minlength=length)
    ll=np.bincount(sid[larger],weights=w[larger]*lost[larger],minlength=length)
    both=(ns>0)&(nl>0);overlap=np.minimum(ns[both],nl[both])
    numerator=float(np.sum(overlap*(ls[both]/ns[both]-ll[both]/nl[both])))
    denominator=float(overlap.sum())
    eligible_small=int(ns[both].sum());eligible_larger=int(nl[both].sum())
    eligible_images=np.unique(np.flatnonzero(both)//(13*5));strata=int(both.sum())
    result['adjusted_small_minus_larger']=(numerator/denominator if denominator else None)
    result['adjustment_support']=dict(small=eligible_small,larger=eligible_larger,images=len(eligible_images),strata=strata)
    return result

def main():
    cfg=read(D/'native_prefix_protocol_FROZEN.json');E=D/'downloaded_evidence'
    assert hashlib.sha256((D/'query_retention_analysis.py').read_bytes()).hexdigest()==read(D/'execution_inputs_manifest.json')['analysis_sha256']
    result=read(E/'result.json');assert result['counts']['decoder_entries']==37
    assert read(D/'archive_verification.json')['status']=='PASS_ALL_ARCHIVE_HASHES_AND_NATIVE_PREFIX_FIDELITY'
    h=helper_functions();cases=h['prepare_cases'](read(D/'raw_annotations_verified.json')['cases'])
    raw=read(E/'raw_native_predictions_without_region_filter.json');assert len(raw)==len(cases)==12
    row_parts=[];events=[];case_summaries=[]
    for i,(c,pred) in enumerate(zip(cases,raw)):
        assert pred['index']==i and pred['file']==c['file']
        b=np.asarray(pred['boxes'],dtype=np.float64).reshape(-1,4)
        labels=np.asarray(pred['labels'],dtype=np.int64);scores=np.asarray(pred['scores'],dtype=np.float64)
        queries=np.asarray(pred['query_ids'],dtype=np.int64)
        keep=h['region_keep'](b,c['integral']);b=b[keep];labels=labels[keep];scores=scores[keep];queries=queries[keep]
        assert np.all((labels>=1)&(labels<=10)) and len(b)<=500
        overlap=h['iou_xywh'](b,c['gt_xywh']);gt=c['gt_xywh'];ngt=len(gt)
        assert ngt<=cfg['eligibility']['max_valid_GT_per_image']
        used=np.zeros(ngt,dtype=bool);tp_score=np.full(ngt,-np.inf)
        order=np.argsort(-scores,kind='stable')
        for j in order:
            eligible=np.flatnonzero((c['gt_labels']==labels[j])&~used);g=-1
            if len(eligible):
                best=eligible[np.argmax(overlap[j,eligible])]
                if overlap[j,best]>=.5:g=int(best);used[g]=True;tp_score[g]=scores[j]
            events.append((scores[j],int(g<0),i))
        arrays=dict(image=np.full(ngt,i,dtype=np.int64),label=c['gt_labels'],
                    raw_row=c['gt_row_indices'],small=c['gt_small'],native_TP_score=tp_score,
                    final_score_bin=np.digitize(tp_score,np.array(cfg['metrics']['confidence_bins'])[1:-1],right=False))
        with np.load(E/'captures'/('case%02d.npz'%i),allow_pickle=False) as z:
            for depth in [1,2]:
                rank=z[f'depth{depth}_rank_valid'];prob=z[f'depth{depth}_probabilities'][0,:,1:11].max(1)
                assert np.array_equal(rank,np.argsort(-prob,kind='stable'))
                allowed=np.isin(queries,rank[:250]);valid=(overlap>=.5)&(labels[:,None]==c['gt_labels'][None,:])&allowed[:,None]
                arrays[f'best_retained_score_depth{depth}']=np.max(np.where(valid,scores[:,None],-np.inf),axis=0)
        row_parts.append(arrays)
        case_summaries.append(dict(index=i,file=c['file'],valid_GT=ngt,small_GT=int(c['gt_small'].sum()),
                                   baseline_native_detections_after_filters=len(b),native_TP_before_global_FP_prefix=int(used.sum())))
    rows={k:np.concatenate([x[k] for x in row_parts]) for k in row_parts[0]}
    ev=np.asarray(events,dtype=np.float64);event_scores=ev[:,0];event_fp=ev[:,1].astype(np.int64);event_images=ev[:,2].astype(np.int64)
    weights=np.load(D/'bootstrap_draws_FROZEN.npz',allow_pickle=False)['weights']
    assert weights.shape==(12,1001) and np.array_equal(weights[:,0],np.ones(12)) and np.all(weights.sum(0)==12)
    budget=cfg['metrics']['global_FP_budget'];point={};samples={1:[],2:[]};thresholds=[];actual_fp=[]
    for j in range(weights.shape[1]):
        threshold,fp=prefix_threshold(event_scores,event_fp,event_images,weights[:,j],budget)
        thresholds.append(threshold);actual_fp.append(fp)
        for depth in [1,2]:
            s=summarize_rows(rows,weights[:,j],threshold,depth)
            if j==0:point[str(depth)]=s
            samples[depth].append(s['small_minus_larger'])
    intervals={}
    for depth in [1,2]:
        finite=np.array([v for v in samples[depth][1:] if v is not None],dtype=np.float64)
        intervals[str(depth)]=dict(valid_draws=len(finite),small_minus_larger_95=(np.quantile(finite,[.025,.975]).tolist() if len(finite) else None))
    primary=point['1'];adj=primary['adjustment_support'];ci=intervals['1']['small_minus_larger_95']
    gates=dict(sample_support=all(primary[k]['native_supported_GT']>=50 and primary[k]['images']>=6 for k in ['small','larger']),
               small_attrition_atleast_005=primary['small']['rate'] is not None and primary['small']['rate']>=.05,
               excess_atleast_003=primary['small_minus_larger'] is not None and primary['small_minus_larger']>=.03,
               excess_CI_lower_positive=intervals['1']['valid_draws']>=950 and ci is not None and ci[0]>0,
               adjusted_same_direction=adj['small']>=20 and adj['larger']>=20 and adj['images']>=6 and
                   primary['adjusted_small_minus_larger'] is not None and primary['adjusted_small_minus_larger']>=0)
    np.savez_compressed(D/'native_support_rows.npz',**rows,event_scores=event_scores,event_fp=event_fp,event_images=event_images)
    np.savez_compressed(D/'paired_support_bootstrap.npz',depth1=np.array([np.nan if x is None else x for x in samples[1]]),
                        depth2=np.array([np.nan if x is None else x for x in samples[2]]),thresholds=np.array(thresholds),actual_FP=np.array(actual_fp))
    out=dict(status='COMPLETED_FROZEN_NATIVE_SUPPORT_DIAGNOSIS',cases=case_summaries,GT_rows=len(rows['image']),
             global_FP_budget=budget,base_actual_FP=actual_fp[0],point=point,paired_image_intervals=intervals,
             gates=gates,question_worth_actual_pruned_decoder_followup=all(gates.values()),new_algorithm_validated=False,
             limits=cfg['interpretation_limits'])
    save(D/'query_retention_analysis_result.json',out)
    print(json.dumps({k:out[k] for k in ['status','GT_rows','base_actual_FP','point','paired_image_intervals','gates','question_worth_actual_pruned_decoder_followup']},ensure_ascii=False))

if __name__=='__main__':
    if '--fixture' in sys.argv:print(json.dumps(fixtures()))
    else:main()
