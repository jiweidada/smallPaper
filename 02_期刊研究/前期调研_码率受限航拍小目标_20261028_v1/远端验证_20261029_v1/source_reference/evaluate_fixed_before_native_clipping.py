"""Unfused one2many fixed-checkpoint evaluation on the preselected dev diagnostic100 only."""
from pathlib import Path
from io import BytesIO,StringIO
from collections import defaultdict
import argparse,contextlib,copy,hashlib,json,math,time
import cv2,numpy as np,torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.utils import nms
from study import ROOT,json_write

def geometry(width,height):
    ratio=min(640/width,640/height);rw,rh=round(width*ratio),round(height*ratio)
    left,top=round((640-rw)/2-.1),round((640-rh)/2-.1)
    return dict(ratio=ratio,resized_wh=[rw,rh],pad_ltrb=[left,top,640-rw-left,640-rh-top],
                effective_resize_xy=[rw/width,rh/height])

def prepare_pixels(pixels,meta):
    rw,rh=meta['resized_wh'];l,t,r,b=meta['pad_ltrb']
    out=cv2.resize(pixels,(rw,rh),interpolation=cv2.INTER_LINEAR)
    out=cv2.copyMakeBorder(out,t,b,l,r,cv2.BORDER_CONSTANT,value=(114,114,114))
    assert out.shape==(640,640,3)
    return torch.from_numpy(np.ascontiguousarray(out.transpose(2,0,1))).unsqueeze(0).cuda().float()/255

def prepare_observations():
    gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text())
    folder=ROOT/'observations/q20';folder.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for image in gt['images']:
        source=ROOT/'dataset/train/images'/image['file_name'];target=folder/image['file_name']
        with Image.open(source) as im:rgb=im.convert('RGB')
        if not target.exists():
            rgb.save(target,format='JPEG',quality=20,subsampling=2,optimize=True,progressive=False)
        payload=target.read_bytes()
        metadata.append(dict(image_id=image['id'],file_name=image['file_name'],source=str(source),q20=str(target),
            width=image['width'],height=image['height'],geometry=geometry(image['width'],image['height']),
            jpeg_bytes=len(payload),jpeg_sha256=hashlib.sha256(payload).hexdigest(),source_bytes=source.stat().st_size))
    json_write(ROOT/'receipts/diagnostic_observations.json',dict(images=metadata,quality=20,subsampling=2,optimize=True,progressive=False,
        bpp=8*sum(x['jpeg_bytes'] for x in metadata)/sum(x['width']*x['height'] for x in metadata),
        same_payloads_for_both_detectors=True,source_is_jpeg_not_raw=True))
    return gt,metadata

def ground_truth_input(gt,metadata):
    new=copy.deepcopy(gt);mapping={r['image_id']:r for r in metadata}
    for image in new['images']:image.update(width=640,height=640)
    for ann in new['annotations']:
        meta=mapping[ann['image_id']]['geometry'];r=meta['ratio'];l,t,_,_=meta['pad_ltrb'];x,y,w,h=ann['bbox']
        ann['bbox']=[x*r+l,y*r+t,w*r,h*r];ann['area']=w*h*r*r
    return new

def coco_eval(gt,predictions):
    with contextlib.redirect_stdout(StringIO()):
        coco=COCO();coco.dataset=copy.deepcopy(gt);coco.createIndex()
        if predictions:dt=coco.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(images=copy.deepcopy(gt['images']),annotations=[],categories=copy.deepcopy(gt['categories']));dt.createIndex()
        ev=COCOeval(coco,dt,'bbox')
        ev.params.maxDets=[1,10,500]
        ev.params.areaRng=[[0,1e10],[0,math.nextafter(1024.,-math.inf)],[1024.,math.nextafter(9216.,-math.inf)],[9216.,1e10]]
        ev.params.areaRngLbl=['all','small','medium','large'];ev.evaluate();ev.accumulate()
    def ap(area,threshold=None):
        p=ev.eval['precision'][:,:,:,area,2]
        if threshold is not None:p=p[np.isclose(ev.params.iouThrs,threshold)]
        values=p[p>=0]
        return float(values.mean()) if values.size else None
    return ev,{'AP500':ap(0),'AP50_500':ap(0,.5),'AP_small_500':ap(1)}

def match_summary(ev,gt,confidence=.25,iou=.5):
    uid={a['id']:a.get('stable_uid',str(a['id'])) for a in gt['annotations'] if not a['iscrowd']}
    small={a['id'] for a in gt['annotations'] if not a['iscrowd'] and a['area']<1024}
    iou_index=int(np.flatnonzero(np.isclose(ev.params.iouThrs,iou))[0])
    tp=set();fp=0
    for entry in ev.evalImgs:
        if entry is None or entry['aRng']!=ev.params.areaRng[0] or entry['maxDet']!=500:continue
        for j,score in enumerate(entry['dtScores']):
            if score<confidence or entry['dtIgnore'][iou_index,j]:continue
            matched=int(entry['dtMatches'][iou_index,j])
            if matched:tp.add(matched)
            else:fp+=1
    valid={a['id'] for a in gt['annotations'] if not a['iscrowd']}
    return {'confidence':confidence,'iou':iou,'TP':len(tp),'FN':len(valid-tp),'FP':fp,
            'FP_per_image':fp/len(gt['images']),'small_TP':len(tp&small),'small_FN':len(small-tp),
            'small_recall':len(tp&small)/len(small) if small else None,
            'matched_small_uids':sorted(uid[i] for i in tp&small)}

def bootstrap_ap(ev,draw):
    # Same matching records as COCOeval, resampled by image; no model execution or GT selection.
    params=ev.params;image_ids=params.imgIds;cat_ids=params.catIds
    lookup={(e['category_id'],e['image_id']):e for e in ev.evalImgs if e is not None and e['aRng']==params.areaRng[1] and e['maxDet']==500}
    category_values=[]
    for category in cat_ids:
        selected=[lookup.get((category,image_ids[int(i)])) for i in draw];selected=[e for e in selected if e is not None]
        positives=sum(np.count_nonzero(np.asarray(e['gtIgnore'])==0) for e in selected)
        if not positives:continue
        scores=np.concatenate([e['dtScores'] for e in selected])
        order=np.argsort(-scores,kind='mergesort')
        match=np.concatenate([e['dtMatches'] for e in selected],axis=1)[:,order]
        ignore=np.concatenate([e['dtIgnore'] for e in selected],axis=1)[:,order]
        tp=np.cumsum((match>0)&~ignore,axis=1);fp=np.cumsum((match==0)&~ignore,axis=1)
        values=[]
        for row in range(len(params.iouThrs)):
            recall=tp[row]/positives;precision=tp[row]/(tp[row]+fp[row]+np.spacing(1))
            precision=np.maximum.accumulate(precision[::-1])[::-1]
            indices=np.searchsorted(recall,params.recThrs,side='left');q=np.zeros(len(indices));good=indices<len(precision)
            q[good]=precision[indices[good]];values.append(q.mean())
        category_values.append(np.mean(values))
    return float(np.mean(category_values)) if category_values else float('nan')

def run(arm,gt,metadata):
    complete=json.loads((ROOT/'results'/arm/'completed.json').read_text())
    path=Path(complete['primary_checkpoint'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==complete['primary_checkpoint_sha256']
    model=YOLO(str(path)).model.cuda().float().eval().requires_grad_(False)
    head=model.model[-1];assert head.cv2 is not None and head.one2one_cv2 is not None
    head.end2end=False;head.export=False
    result={};evaluators={};input_gt=ground_truth_input(gt,metadata)
    for condition in ['source','q20']:
        predictions=[];original_predictions=[];cap_hits=0
        for index,meta in enumerate(metadata):
            with Image.open(meta[condition]) as image:pixels=np.asarray(image.convert('RGB'))
            tensor=prepare_pixels(pixels,meta['geometry'])
            with torch.no_grad():
                decoded,raw=model(tensor)
                detections=nms.non_max_suppression(decoded,conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=500,nc=10,end2end=False)[0]
            cap_hits+=int(len(detections)==500)
            l,t,_,_=meta['geometry']['pad_ltrb'];ratio=meta['geometry']['ratio']
            for x1,y1,x2,y2,score,category in detections.cpu().tolist():
                predictions.append(dict(image_id=meta['image_id'],category_id=int(category),bbox=[x1,y1,x2-x1,y2-y1],score=score))
                ox1,oy1=max(0,min(meta['width'],(x1-l)/ratio)),max(0,min(meta['height'],(y1-t)/ratio))
                ox2,oy2=max(0,min(meta['width'],(x2-l)/ratio)),max(0,min(meta['height'],(y2-t)/ratio))
                original_predictions.append(dict(image_id=meta['image_id'],category_id=int(category),bbox=[ox1,oy1,ox2-ox1,oy2-oy1],score=score))
            if (index+1)%20==0:print(json.dumps({'arm':arm,'condition':condition,'images':index+1}),flush=True)
        ev,metrics=coco_eval(input_gt,predictions);_,original=coco_eval(gt,original_predictions)
        summaries=[match_summary(ev,input_gt,c,iou) for c in [.05,.1,.25,.5] for iou in [.5,.75]]
        result[condition]={'AP500':metrics['AP500'],'AP50_500':metrics['AP50_500'],'AP_small_input_500':metrics['AP_small_500'],
                           'AP_small_original_500':original['AP_small_500'],'cap_hits':cap_hits,'images':len(metadata),'matches':summaries}
        json_write(ROOT/'evaluation'/arm/(condition+'_predictions_input.json'),predictions)
        json_write(ROOT/'evaluation'/arm/(condition+'_metrics.json'),result[condition])
        evaluators[condition]=ev
    ref=next(x for x in result['source']['matches'] if x['confidence']==.25 and x['iou']==.5)
    comp=next(x for x in result['q20']['matches'] if x['confidence']==.25 and x['iou']==.5)
    a,b=set(ref['matched_small_uids']),set(comp['matched_small_uids'])
    result['paired_transitions']={'source_hit_then_jpeg_missed':sorted(a-b),'jpeg_hit_then_source_missed':sorted(b-a),'both_hit':len(a&b),
                                  'all_GT_never_detected_not_counted_as_compression_new_misses':True}
    result['remaining_small_AP_loss_pp']=100*(result['source']['AP_small_input_500']-result['q20']['AP_small_input_500'])
    json_write(ROOT/'evaluation'/arm/'summary.json',result)
    del model;torch.cuda.empty_cache()
    return result,evaluators

if __name__=='__main__':
    gt,metadata=prepare_observations();result={};cache={}
    for arm in ['D0','DA']:result[arm],cache[arm]=run(arm,gt,metadata)
    # Exact full-image draw must reproduce COCO's primary AP before any resampling intervals are trusted.
    draw=np.arange(len(metadata))
    for arm in ['D0','DA']:
        for condition in ['source','q20']:
            assert abs(bootstrap_ap(cache[arm][condition],draw)-result[arm][condition]['AP_small_input_500'])<1e-12
    samples=defaultdict(list);rng=np.random.default_rng(20261029)
    for iteration in range(2000):
        draw=rng.integers(0,len(metadata),len(metadata))
        for arm in ['D0','DA']:
            gap=100*(bootstrap_ap(cache[arm]['source'],draw)-bootstrap_ap(cache[arm]['q20'],draw));samples[arm].append(gap)
        if (iteration+1)%200==0:print(json.dumps({'paired_bootstrap':iteration+1,'total':2000}),flush=True)
    for arm in ['D0','DA']:result[arm]['gap_paired_image_bootstrap95_pp']=np.nanquantile(samples[arm],[.025,.975]).tolist()
    result['interpretation']={'checkpoint_conditional_only':True,'single_seed':True,'image_bootstrap_may_underestimate_related_frame_uncertainty':True,
       'P1_planning_signal_DA_gap_ge_1pp':result['DA']['remaining_small_AP_loss_pp']>=1,
       'P2_region_joint_necessity_validated':False,'P3_deployable_algorithm_validated':False,
       'no_strategy_training_or_GT_oracle_deployment_score':True,'not_official_VisDrone_challenge_score':True}
    json_write(ROOT/'evaluation/initial_diagnostic_result.json',result)
    print(json.dumps({'status':'DIAGNOSTIC_COMPLETED','D0_gap_pp':result['D0']['remaining_small_AP_loss_pp'],'DA_gap_pp':result['DA']['remaining_small_AP_loss_pp']}),flush=True)
