"""Independent scalar-IoU verification of all missed-GT bins, no NumPy or models."""
from pathlib import Path
import collections,hashlib,json,math,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def scalar_iou(x,y):
    left=max(x[0],y[0]);top=max(x[1],y[1]);right=min(x[0]+x[2],y[0]+y[2]);bottom=min(x[1]+x[3],y[1]+y[3])
    intersection=max(0,right-left)*max(0,bottom-top)
    union=x[2]*x[3]+y[2]*y[3]-intersection
    return intersection/max(union,1e-12)
def main():
    p=read(R/'D0_cache_audit_protocol_FROZEN.json');result=read(R/'D0_cache_candidate_audit_result.json')
    gt=read(p['files']['GT']['path']);pred=read(p['files']['predictions']['path']);meta={x['image_id']:x['geometry'] for x in read(p['files']['metadata']['path'])['images']}
    metrics=read(p['files']['metrics']['path']);m=next(x for x in metrics['matches'] if x['confidence']==.25 and x['iou']==.5);matched=set(m['matched_small_uids'])
    grouped=collections.defaultdict(list)
    for x in pred:grouped[x['image_id']].append(x)
    indexed={x['stable_uid']:x for x in gt['annotations'] if not x['iscrowd']}
    assert len({x['uid'] for x in result['rows']})==len(result['rows'])
    expected_missing={uid for uid,x in indexed.items() if x['area']*meta[x['image_id']]['ratio']**2<1024}-matched
    assert expected_missing=={x['uid'] for x in result['rows']}
    counts=collections.Counter()
    for row in result['rows']:
        x=indexed[row['uid']];g=meta[x['image_id']];s=g['ratio'];l,t=g['pad_ltrb'][:2];a=x['bbox'];b=[a[0]*s+l,a[1]*s+t,a[2]*s,a[3]*s]
        same=[];any_cover=[];near=False;maxsame=0.
        for d in grouped[x['image_id']]:
            v=scalar_iou(b,d['bbox']);equal=d['category_id']==x['category_id']
            if equal:maxsame=max(maxsame,v)
            if v>=.5:
                any_cover.append(d['score'])
                if equal:same.append(d['score'])
            if equal and v>=.1 and d['score']>=.05:near=True
        ms=max(same) if same else None;ma=max(any_cover) if any_cover else None
        if ms is not None and ms>=.25:i=0
        elif ms is not None and ms>=.05:i=1
        elif ms is not None:i=2
        elif ma is not None and ma>=.05:i=3
        elif near:i=4
        elif ma is not None:i=5
        else:i=6
        label=p['exclusive_priority_bins'][i]['id'];assert label==row['bin'],row['uid']
        assert ms==row['best_same_class_cover_score'] and ma==row['best_any_class_cover_score']
        assert abs(maxsame-row['max_same_class_IoU'])<1e-12
        counts[label]+=1
    assert dict(counts)==result['exclusive_bin_counts']
    out={'status':'PASS_ALL_MISSED_GT_SCALAR_IOU_BIN_REAGGREGATION','GT_rows_checked':len(result['rows']),'scope':'Existing post-NMS candidate availability only; no algorithm, AP or causal validation.','model_calls':0,'result_sha256':hashlib.sha256((R/'D0_cache_candidate_audit_result.json').read_bytes()).hexdigest()}
    with (R/'D0_cache_independent_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
