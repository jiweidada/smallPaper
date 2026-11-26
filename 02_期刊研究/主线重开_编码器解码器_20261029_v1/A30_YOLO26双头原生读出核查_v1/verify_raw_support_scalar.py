"""Verify all native positive counts and stratified scalar raw-support examples."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent;D=R/'downloaded_evidence'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    result=read(R/'raw_support_result.json');index={x['uid']:x for x in result['rows']};meta=read(D/'capture_metadata.json');cfg=read(R/'native_dual_head_protocol_FROZEN.json');locations={uid:(case,i) for case in meta for i,uid in enumerate(case['GT_uids'])}
    verified_counts=0
    for case in meta:
        data=np.load(D/'captures'/('case_%03d.npz'%case['index']),allow_pickle=False)
        for k in ['M','O']:
            fg=data[k+'_fg'].astype(bool);weight=data[k+'_target_scores'].sum(axis=1);eff=fg&(weight>0);g=data[k+'_target_gt_idx'];n=len(case['GT_uids']);nc=np.bincount(g[fg],minlength=n);ne=np.bincount(g[eff],minlength=n)
            for i,uid in enumerate(case['GT_uids']):
                assert int(nc[i])==index[uid][k]['native_fg_count'] and int(ne[i])==index[uid][k]['native_effective_positive_count'];verified_counts+=1
    chosen=[]
    for k in ['M','O']:
        groups=[lambda v:v['raw_cover_exists'] and v['best_cover_score']<.25 and v['native_effective_positive_count']==0,lambda v:v['raw_cover_exists'] and v['best_cover_score']<.25 and v['native_effective_positive_count']>0,lambda v:v['raw_cover_exists'] and v['best_cover_score']>=.25,lambda v:not v['raw_cover_exists'],lambda v:v['raw_cover_exists'] and not v['best_cover_inside_original_GT']]
        for predicate in groups:
            choices=sorted([x for x in result['rows'] if x['small'] and not x[k]['matched_small_at025'] and predicate(x[k])],key=lambda x:x['uid'])[:2]
            chosen.extend((k,x['uid']) for x in choices)
    chosen=sorted(set(chosen));assert len(chosen)>=8
    for k,uid in chosen:
        case,gi=locations[uid];data=np.load(D/'captures'/('case_%03d.npz'%case['index']),allow_pickle=False);z=cfg['cases'][case['index']];geo=z['geometry'];s=geo['ratio'];l,t=geo['pad_ltrb'][:2];gt=data['GT_xyxy'][gi].tolist();cls=int(data['GT_classes'][gi]);best=None;value=None
        for i,box in enumerate(data[k+'_decoded_xyxy']):
            a=box.tolist();a[0]=max(l,min(a[0],l+s*z['width']));a[2]=max(l,min(a[2],l+s*z['width']));a[1]=max(t,min(a[1],t+s*z['height']));a[3]=max(t,min(a[3],t+s*z['height']))
            inter=max(0,min(a[2],gt[2])-max(a[0],gt[0]))*max(0,min(a[3],gt[3])-max(a[1],gt[1]));union=(a[2]-a[0])*(a[3]-a[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-inter;overlap=inter/union if union>0 else 0
            if overlap<.5:continue
            p=1/(1+math.exp(-max(-80,min(80,float(data[k+'_scores'][cls,i])))))
            if value is None or p>value:best,value=i,p
        wanted=index[uid][k];assert best==wanted['best_cover_anchor'],(k,uid,best,wanted['best_cover_anchor'])
        assert value is None or abs(value-wanted['best_cover_score'])<1e-15
    for k in ['M','O']:
        small=[x[k] for x in result['rows'] if x['small']];miss=[x for x in small if not x['matched_small_at025']];summary=result['summary'][k]
        assert sum(x['native_effective_positive_count']==0 for x in small)==summary['small_zero_effective_native_positives']
        assert sum(x['raw_cover_exists'] and x['best_cover_score']<.25 and x['native_effective_positive_count']==0 for x in miss)==summary['missed_low_cover_score_and_zero_effective_native_positives']
    out={'status':'PASS_ALL8982_NATIVE_GT_COUNTS_AND_STRATIFIED_SCALAR_SUPPORT','native_GT_branch_rows_checked':verified_counts,'raw_support_scalar_rows_checked':len(chosen),'chosen_rows':chosen,'raw_result_sha256':hashlib.sha256((R/'raw_support_result.json').read_bytes()).hexdigest(),'model_calls':0,'scope':'Descriptive final-checkpoint support,not historical training assignments or new-mechanism validation.'}
    with (R/'raw_support_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='chosen_rows'},ensure_ascii=False))
if __name__=='__main__':main()
