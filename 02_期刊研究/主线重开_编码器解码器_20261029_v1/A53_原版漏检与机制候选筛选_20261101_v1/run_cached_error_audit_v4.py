"""One fixed descriptive audit of cached native B outputs. No torch/model imports."""
from pathlib import Path
from collections import defaultdict, Counter
import contextlib,csv,hashlib,io,json,sys,time

ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
A49=BASE/'A49_下一主线有界筛选_20261101_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    start=time.monotonic(); result_path=ROOT/'error_audit_result_v4.json'; assert not result_path.exists()
    plan=load(ROOT/'protocol_FIXED_v4.json')
    assert all(sha(Path(p))==h for p,h in plan['source_files_sha256'].items())
    sys.path.insert(0,str(A49/'postprocess_ignore_strata/deps'))
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    folder=A49/'completed_B_v1/arm_B'
    gt=load(folder/'epoch12_EMA_GT.json'); pred=load(folder/'epoch12_EMA_predictions.json')
    reference=load(folder/'epoch12_EMA_metrics.json')
    assert len(gt['images'])==548 and len(gt['annotations'])==38602
    ims={x['id']:x for x in gt['images']}; anns={x['id']:x for x in gt['annotations']}
    manifest=load(A49/'val_data_manifest.json')
    assert {r['id']:(r['file'],r['width'],r['height']) for r in manifest['rows']}=={k:(x['file_name'],x['width'],x['height']) for k,x in ims.items()}
    assert len(anns)==38602 and all(a['iscrowd']==0 and a['area']>0 for a in anns.values())
    bypred=defaultdict(list); bygt=defaultdict(list)
    for p in pred:bypred[p['image_id']].append(p)
    for a in anns.values():bygt[a['image_id']].append(a)
    assert set(bypred)<=set(ims) and all(len(v)<=300 for v in bypred.values())
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):
        api=COCO();api.dataset=gt;api.createIndex();dt=api.loadRes(pred)
        ev=COCOeval(api,dt,'bbox');ev.params.maxDets=[1,10,300]
        ev.evaluate();ev.accumulate()
    (ROOT/'COCO_evaluation_stdout_v4.txt').write_text(capture.getvalue(),encoding='utf-8')
    t50=int(np.argmin(abs(ev.params.iouThrs-.5)));t75=int(np.argmin(abs(ev.params.iouThrs-.75)))
    sm75=ev.eval['precision'][t75,:,:,1,2];sm75=sm75[sm75>=0]
    precision=ev.eval['precision'];recall=ev.eval['recall']
    def ap(area=0,iou=None):
        z=precision[:,:,:,area,-1]
        if iou is not None:z=z[np.isclose(ev.params.iouThrs,iou)]
        z=z[z>=0];return float(z.mean()*100) if len(z) else None
    def ar(area=0):
        z=recall[:,:,area,-1];z=z[z>=0];return float(z.mean()*100) if len(z) else None
    metrics={'AP':ap(),'AP50':ap(iou=.5),'AP75':ap(iou=.75),'APs':ap(1),'APs75':ap(1,.75),'APm':ap(2),'APl':ap(3),'ARs':ar(1)}
    differences={k:metrics[k]-reference[k] for k in metrics}
    assert max(abs(v) for v in differences.values())<=1e-8,differences
    matched={50:set(),75:set()}
    for item in ev.evalImgs:
        if item is None or item['aRng']!=ev.params.areaRng[0] or item['maxDet']!=300:continue
        for label,t in [(50,t50),(75,t75)]:
            for aid,did,ignored in zip(item['gtIds'],item['gtMatches'][t],item['gtIgnore']):
                if did>0 and not ignored:matched[label].add(int(aid))

    def boxarray(values):
        x=np.asarray(values,dtype=np.float64).reshape(-1,4).copy()
        assert np.isfinite(x).all() and (x[:,2:]>=0).all()
        x[:,2:]+=x[:,:2]
        return x
    def iou(a,b):
        if not len(a) or not len(b):return np.zeros((len(a),len(b)))
        wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
        inter=wh[:,:,0]*wh[:,:,1]
        area_a=np.prod(a[:,2:]-a[:,:2],axis=1);area_b=np.prod(b[:,2:]-b[:,:2],axis=1)
        return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)

    rows=[]
    for iid,im in sorted(ims.items()):
        aa=bygt[iid];pp=bypred[iid];gb=boxarray([a['bbox'] for a in aa]);pb=boxarray([p['bbox'] for p in pp])
        gc=np.asarray([a['category_id'] for a in aa]);pc=np.asarray([p['category_id'] for p in pp])
        ov=iou(gb,pb);same=gc[:,None]==pc[None,:]
        anymax=ov.max(axis=1) if len(pp) else np.zeros(len(aa))
        samemax=np.where(same,ov,0).max(axis=1) if len(pp) else np.zeros(len(aa))
        gg=iou(gb,gb);np.fill_diagonal(gg,0)
        ggsame=np.where(gc[:,None]==gc[None,:],gg,0).max(axis=1) if len(aa) else np.zeros(0)
        for j,a in enumerate(aa):
            mid=a['id'];small=a['area']<=1024
            minside=min(a['bbox'][2]*640/im['width'],a['bbox'][3]*640/im['height'])
            binname='lt2' if minside<2 else '2to4' if minside<4 else '4to8' if minside<8 else 'ge8'
            if mid in matched[50]:kind='matched50'
            elif samemax[j]>=.5:kind='correct_class_coverage_unmatched50'
            elif anymax[j]>=.5:kind='other_class_geometric_coverage50'
            elif anymax[j]>=.1:kind='partial_geometric_coverage_0_1_to_0_5'
            else:kind='no_final_geometric_coverage_ge0_1'
            rows.append({'GT_id':mid,'image_id':iid,'category_id':a['category_id'],'small':small,
                'original_area':a['area'],'input_min_side':minside,'input_min_side_bin':binname,
                'matched50':mid in matched[50],'matched75':mid in matched[75],
                'max_any_class_IoU':float(anymax[j]),'max_same_class_IoU':float(samemax[j]),
                'overlap_same_class_GT_ge0_1':bool(ggsame[j]>=.1),'category50':kind})
    assert len(rows)==38602
    def summarize(selected):
        n=len(selected);c=Counter(r['category50'] for r in selected)
        return {'GT':n,'matched50':sum(r['matched50'] for r in selected),'matched75':sum(r['matched75'] for r in selected),
            'recall50_percent':None if not n else sum(r['matched50'] for r in selected)*100/n,
            'recall75_percent':None if not n else sum(r['matched75'] for r in selected)*100/n,
            'category50_counts':dict(c)}
    small=[r for r in rows if r['small']]
    result={'status':'DESCRIPTIVE_AUDIT_COMPLETE_NOT_A_METHOD_PASS','goal_status':'active','mainline_confirmed':False,
        'protocol_sha256':sha(ROOT/'protocol_FIXED_v4.json'),'COCO_evaluate_calls':1,
        'metric_reproduction':metrics,'metric_minus_reference_pp':differences,
        'all_GT':summarize(rows),'small_GT':summarize(small),
        'small_by_input_min_side':{name:summarize([r for r in small if r['input_min_side_bin']==name]) for name in ['lt2','2to4','4to8','ge8']},
        'small_by_same_class_GT_overlap':{str(v):summarize([r for r in small if r['overlap_same_class_GT_ge0_1']==v]) for v in [False,True]},
        'small_by_class':{c['name']:summarize([r for r in small if r['category_id']==c['id']]) for c in gt['categories']},
        'new_model_forwards':0,'new_optimizer_updates':0,'new_formal_epochs':0,
        'additional_training_queued':False,'seconds':time.monotonic()-start,
        'interpretation_limits':plan['limits']}
    csvpath=ROOT/'GT_error_rows_v4.csv';assert not csvpath.exists()
    with csvpath.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result['GT_rows_csv_sha256']=sha(csvpath)
    assert all(sha(Path(p))==h for p,h in plan['source_files_sha256'].items())
    save(result_path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['small_by_class','interpretation_limits']},ensure_ascii=False))
if __name__=='__main__':main()
