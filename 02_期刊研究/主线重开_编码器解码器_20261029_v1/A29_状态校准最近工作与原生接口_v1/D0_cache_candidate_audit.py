"""Describe retained-candidate support for historically missed tiny GT; no model calls."""
from pathlib import Path
import collections,hashlib,json,sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def iou_one(b,boxes):
    lt=np.maximum(b[:2],boxes[:,:2]);rb=np.minimum(b[:2]+b[2:],boxes[:,:2]+boxes[:,2:])
    wh=np.maximum(rb-lt,0);inter=wh[:,0]*wh[:,1]
    return inter/np.maximum(b[2]*b[3]+boxes[:,2]*boxes[:,3]-inter,1e-12)
def main():
    cfg=read(R/'D0_cache_audit_protocol_FROZEN.json');out=R/'D0_cache_candidate_audit_result.json';assert not out.exists()
    for row in cfg['files'].values():
        h=hashlib.sha256()
        with Path(row['path']).open('rb') as f:
            for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
        assert h.hexdigest()==row['sha256'],row['path']
    data={k:read(v['path']) for k,v in cfg['files'].items() if k in ['predictions','metrics','GT','metadata']}
    metric=next(x for x in data['metrics']['matches'] if x['confidence']==.25 and x['iou']==.5)
    matched=set(metric['matched_small_uids']);meta={x['image_id']:x['geometry'] for x in data['metadata']['images']}
    grouped=collections.defaultdict(list)
    for i,p in enumerate(data['predictions']):grouped[p['image_id']].append((i,p))
    arrays={im:(np.asarray([p['bbox'] for _,p in ps],dtype=np.float64).reshape(-1,4),np.asarray([p['category_id'] for _,p in ps]),np.asarray([p['score'] for _,p in ps]),np.asarray([i for i,_ in ps])) for im,ps in grouped.items()}
    labels=[x['id'] for x in cfg['exclusive_priority_bins']];rows=[];valid=0;small=0;missed=0
    for gt in data['GT']['annotations']:
        if gt['iscrowd']:continue
        valid+=1;g=meta[gt['image_id']];r=g['ratio'];area=gt['area']*r*r
        if not area<1024:continue
        small+=1
        if gt['stable_uid'] in matched:continue
        missed+=1;x,y,w,h=gt['bbox'];l,t,*_=g['pad_ltrb'];b=np.asarray([x*r+l,y*r+t,w*r,h*r])
        boxes,cats,scores,ids=arrays.get(gt['image_id'],(np.empty((0,4)),np.empty(0),np.empty(0),np.empty(0,dtype=int)))
        overlaps=iou_one(b,boxes);same=cats==gt['category_id'];cover=overlaps>=.5
        same_score=float(scores[same&cover].max()) if np.any(same&cover) else None
        any_score=float(scores[cover].max()) if np.any(cover) else None
        conditions=[same_score is not None and same_score>=.25,same_score is not None and same_score>=.05,same_score is not None,any_score is not None and any_score>=.05,bool(np.any(same&(overlaps>=.1)&(scores>=.05))),any_score is not None,True]
        label=next(lab for lab,test in zip(labels,conditions) if test)
        rows.append({'uid':gt['stable_uid'],'image_id':gt['image_id'],'category_id':gt['category_id'],'input_area':area,'size_bin':'lt16sq' if area<256 else '16to32sq','bin':label,'best_same_class_cover_score':same_score,'best_any_class_cover_score':any_score,'max_same_class_IoU':float(overlaps[same].max()) if same.any() else 0.,'high_score_same_class_cover_prediction_ids':ids[same&cover&(scores>=.25)].tolist()})
    assert len(meta)==100 and valid==4491 and small==4129
    assert small-len(matched)==metric['small_FN']==missed==len(rows)
    counts={lab:sum(x['bin']==lab for x in rows) for lab in labels}
    by_size={size:{lab:sum(x['size_bin']==size and x['bin']==lab for x in rows) for lab in labels} for size in ['lt16sq','16to32sq']}
    result={'phase':'A29','status':'EXPLORATORY_EXISTING_SOURCE_PREDICTION_AUDIT_COMPLETED','host':cfg['host'],
      'images':len(meta),'saved_predictions':len(data['predictions']),'valid_GT':valid,'small_GT':small,'historical_small_TP':len(matched),'historical_small_FN':missed,'exclusive_bin_counts':counts,'by_size':by_size,
      'rows':rows,'reading_rules':cfg['reading_rules'],'new_model_calls':0,'new_optimizer_updates':0,'novel_mechanism_or_mainline_validated':False}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','reading_rules','host']},ensure_ascii=False))
if __name__=='__main__':main()
