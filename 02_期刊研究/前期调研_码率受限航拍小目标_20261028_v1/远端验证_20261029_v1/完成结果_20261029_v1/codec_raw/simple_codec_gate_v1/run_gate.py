"""Bounded, frozen-checkpoint codec controls. No training or policy fitting."""
from pathlib import Path
from io import BytesIO, StringIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib, copy, hashlib, json, math, os, sys, time, traceback
import cv2, numpy as np, torch
from PIL import Image, features, __version__ as pillow_version

ROOT = Path('/root/c10_region_restore_20261029_v1')
OUT = ROOT / 'simple_codec_gate_v1'
sys.path.insert(0, str(ROOT / 'code'))
from evaluate_fixed import geometry, prepare_pixels, detection_records, ground_truth_input
from study import json_write
from ultralytics import YOLO
from ultralytics.utils import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

START = time.monotonic()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def event(stage, **kwargs):
    info = dict(stage=stage, elapsed_seconds=time.monotonic()-START, pid=os.getpid(), **kwargs)
    json_write(OUT / 'live.json', info)
    print(json.dumps(info), flush=True)

def encode(rgb, kind, quality):
    tic=time.monotonic()
    if kind == 'half_jpeg':
        a=np.asarray(rgb); h,w=a.shape[:2]
        rgb=Image.fromarray(cv2.resize(a,(max(1,round(w/2)),max(1,round(h/2))),interpolation=cv2.INTER_AREA))
    buf=BytesIO()
    if kind=='webp':
        rgb.save(buf,format='WEBP',quality=quality,lossless=False,method=4)
    else:
        rgb.save(buf,format='JPEG',quality=quality,subsampling=2,optimize=True,progressive=False)
    return buf.getvalue(),time.monotonic()-tic

def metrics(gt, predictions, cap):
    with contextlib.redirect_stdout(StringIO()):
        coco=COCO();coco.dataset=copy.deepcopy(gt);coco.createIndex()
        if predictions:
            dt=coco.loadRes(predictions)
        else:
            dt=COCO();dt.dataset=dict(images=copy.deepcopy(gt['images']),annotations=[],categories=copy.deepcopy(gt['categories']));dt.createIndex()
        ev=COCOeval(coco,dt,'bbox');ev.params.maxDets=[1,10,cap]
        ev.params.areaRng=[[0,1e10],[0,math.nextafter(1024.,-math.inf)],[1024.,math.nextafter(9216.,-math.inf)],[9216.,1e10]]
        ev.params.areaRngLbl=['all','small','medium','large'];ev.evaluate();ev.accumulate()
    def ap(area,thresh=None):
        p=ev.eval['precision'][:,:,:,area,2]
        if thresh is not None:p=p[np.isclose(ev.params.iouThrs,thresh)]
        valid=p[p>=0];return float(valid.mean()) if valid.size else None
    return ev,dict(AP=ap(0),AP50=ap(0,.5),AP_small=ap(1))

def matches(ev,gt,confidence=.25,iou=.5):
    valid={a['id'] for a in gt['annotations'] if not a['iscrowd']}
    small={a['id'] for a in gt['annotations'] if not a['iscrowd'] and a['area']<1024}
    uids={a['id']:a.get('stable_uid',str(a['id'])) for a in gt['annotations'] if not a['iscrowd']}
    k=int(np.flatnonzero(np.isclose(ev.params.iouThrs,iou))[0]);tp=set();fp=0
    for entry in ev.evalImgs:
        if entry is None or entry['aRng']!=ev.params.areaRng[0] or entry['maxDet']!=ev.params.maxDets[-1]:continue
        for j,score in enumerate(entry['dtScores']):
            if score<confidence or entry['dtIgnore'][k,j]:continue
            match=int(entry['dtMatches'][k,j])
            if match:tp.add(match)
            else:fp+=1
    return dict(confidence=confidence,iou=iou,TP=len(tp),FN=len(valid-tp),FP=fp,FP_per_image=fp/len(gt['images']),
                small_TP=len(tp&small),small_FN=len(small-tp),small_recall=len(tp&small)/len(small) if small else None,matched_small_uids=sorted(uids[i] for i in tp&small))

def fixture():
    meta=dict(image_id=1,width=959,height=539,geometry=geometry(959,539))
    a=np.zeros((270,480,3),dtype=np.uint8)
    # Both alternatives use the same original canvas and coordinate frame.
    tensor=prepare_pixels(a,meta['geometry'])
    assert tensor.shape==(1,3,640,640)
    r=meta['geometry']['ratio'];l,t,_,_=meta['geometry']['pad_ltrb']
    det=torch.tensor([[10*r+l,20*r+t,40*r+l,60*r+t,.9,0]])
    inp,orig=detection_records(det,meta)
    assert np.allclose(orig[0]['bbox'],[10,20,30,40],atol=1e-4)
    gt=dict(info={},categories=[dict(id=0,name='a')],images=[dict(id=1,width=959,height=539)],
            annotations=[dict(id=1,image_id=1,category_id=0,bbox=[10,20,30,40],area=1200,iscrowd=0)])
    for cap in [500,1000]:
        ev,ms=metrics(gt,orig,cap);assert abs(ms['AP']-1)<1e-12
        assert matches(ev,gt)['TP']==1
    return dict(status='PASS',half_geometry_roundtrip=True,perfect_AP_caps_500_1000=True,detector_forwards=0)

def calibration(protocol):
    # This stage uses image metadata and payload sizes; never detection scores or GT annotations.
    images=json.loads((ROOT/'dataset/tuning_gt_orig.json').read_text())['images']
    qualities=protocol['codec_calibration']['quality_grid']
    def one(im):
        with Image.open(ROOT/'dataset/train/images'/im['file_name']) as p:rgb=p.convert('RGB')
        ref,_=encode(rgb,'q20',20)
        row=dict(file_name=im['file_name'],reference_bytes=len(ref),sizes={})
        for kind in ['webp','half_jpeg']:
            row['sizes'][kind]={str(q):len(encode(rgb,kind,q)[0])+(8 if kind=='half_jpeg' else 0) for q in qualities}
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=protocol['codec_calibration']['workers']) as pool:
        for row in pool.map(one,images):
            rows.append(row)
            if len(rows)%10==0:event('BYTE_CALIBRATION',images=len(rows),total=len(images))
    ref=sum(r['reference_bytes'] for r in rows)
    totals={kind:{str(q):sum(r['sizes'][kind][str(q)] for r in rows) for q in qualities} for kind in ['webp','half_jpeg']}
    chosen={kind:min(qualities,key=lambda q:(abs(totals[kind][str(q)]-ref),q)) for kind in totals}
    result=dict(images=len(rows),reference_total_bytes=ref,grid_totals=totals,chosen_qualities=chosen,
                selected_relative_rate_drift={kind:totals[kind][str(q)]/ref-1 for kind,q in chosen.items()},
                per_image=rows,detector_queries=0,GT_utility_used=False)
    json_write(OUT/'calibration.json',result)
    return result

def observations(cal):
    meta=json.loads((ROOT/'receipts/diagnostic_observations.json').read_text())['images']
    payload_rows=[]
    for i,m in enumerate(meta):
        assert sha(m['q20'])==m['jpeg_sha256']
        with Image.open(m['source']) as p:rgb=p.convert('RGB')
        for kind,q in cal['chosen_qualities'].items():
            payload,encode_s=encode(rgb,kind,q)
            path=OUT/'payloads'/kind/(Path(m['file_name']).stem+('.webp' if kind=='webp' else '.jpg'))
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload);m[kind]=str(path)
            if kind=='half_jpeg':
                path.with_suffix('.dims').write_bytes(int(m['width']).to_bytes(4,'big')+int(m['height']).to_bytes(4,'big'))
            tic=time.monotonic()
            with Image.open(BytesIO(payload)) as image:
                image.load();size=image.size
            decode_s=time.monotonic()-tic
            payload_rows.append(dict(image_id=m['image_id'],kind=kind,quality=q,file=str(path),bytes=len(payload),
                 dimension_metadata_bytes=8 if kind=='half_jpeg' else 0,sha256=hashlib.sha256(payload).hexdigest(),
                 encoded_wh=size,encode_seconds=encode_s,decode_seconds=decode_s,reference_bytes=m['jpeg_bytes']))
        if (i+1)%20==0:event('DIAGNOSTIC_PAYLOADS',images=i+1)
    ref=sum(m['jpeg_bytes'] for m in meta)
    summary={}
    for kind in ['webp','half_jpeg']:
        subset=[r for r in payload_rows if r['kind']==kind]
        size=sum(r['bytes']+r['dimension_metadata_bytes'] for r in subset)
        summary[kind]=dict(total_bytes=size,reference_total_bytes=ref,relative_byte_drift=size/ref-1,
            per_image_relative_drift_quantiles=np.quantile([(r['bytes']+r['dimension_metadata_bytes'])/r['reference_bytes']-1 for r in subset],[0,.25,.5,.75,1]).tolist(),
            encode_total_seconds=sum(r['encode_seconds'] for r in subset),decode_total_seconds=sum(r['decode_seconds'] for r in subset))
    json_write(OUT/'observations.json',dict(images=meta,payloads=payload_rows,summary=summary))
    return meta,summary

def evaluate(meta,protocol):
    gt=json.loads((ROOT/'dataset/diagnostic_gt_orig.json').read_text());inp_gt=ground_truth_input(gt,meta)
    previous=json.loads((ROOT/'evaluation/initial_diagnostic_result.json').read_text())
    result={};forwards=0
    for arm in ['D0','DA']:
        complete=json.loads((ROOT/'results'/arm/'completed.json').read_text());assert sha(complete['primary_checkpoint'])==complete['primary_checkpoint_sha256']
        model=YOLO(complete['primary_checkpoint']).model.cuda().float().eval().requires_grad_(False)
        head=model.model[-1];assert head.cv2 is not None and head.one2one_cv2 is not None
        head.end2end=False;head.export=False
        result[arm]={}
        for condition in ['source','q20','webp','half_jpeg']:
            cap=1000 if condition in ['source','q20'] else 500
            ps=[];orig=[];ps500=[];orig500=[];cap_hits=0
            for i,m in enumerate(meta):
                with Image.open(m[condition]) as im:pixels=np.asarray(im.convert('RGB'))
                if condition=='half_jpeg':
                    header=Path(m[condition]).with_suffix('.dims').read_bytes()
                    assert len(header)==8 and int.from_bytes(header[:4],'big')==m['width'] and int.from_bytes(header[4:],'big')==m['height']
                tensor=prepare_pixels(pixels,m['geometry'])
                with torch.inference_mode():
                    decoded,raw=model(tensor)
                    detections=nms.non_max_suppression(decoded,conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,max_det=cap,nc=10,end2end=False)[0]
                forwards+=1;cap_hits+=int(len(detections)==cap)
                a,b=detection_records(detections,m);ps.extend(a);orig.extend(b)
                a,b=detection_records(detections[:500],m);ps500.extend(a);orig500.extend(b)
                if (i+1)%20==0:event('DETECTOR_FORWARD',arm=arm,condition=condition,images=i+1,total_detector_forwards=forwards)
            ev,mi=metrics(inp_gt,ps,cap);_,mo=metrics(gt,orig,cap)
            ev500,m500=metrics(inp_gt,ps500,500)
            entry=dict(cap=cap,cap_hits=cap_hits,AP=mo['AP'],AP50=mo['AP50'],AP_small_original=mo['AP_small'],
                        AP_small_input=mi['AP_small'],AP_input_crosscheck=mi['AP'],prefix500_AP_small_input=m500['AP_small'],
                        matches=[matches(ev,inp_gt,c,k) for c in [.05,.1,.25,.5] for k in [.5,.75]])
            assert abs(mo['AP']-mi['AP'])<1e-10
            if condition in ['source','q20']:
                delta=m500['AP_small']-previous[arm][condition]['AP_small_input_500']
                entry['prefix500_AP_reproduction_delta']=delta
                assert abs(delta)<protocol['cap_sensitivity']['prefix500_crosscheck_tolerance_AP'],(arm,condition,delta)
                old=next(s for s in previous[arm][condition]['matches'] if s['confidence']==.25 and s['iou']==.5)
                new=next(s for s in entry['matches'] if s['confidence']==.25 and s['iou']==.5)
                assert all(old[k]==new[k] for k in ['TP','FN','FP','small_TP','small_FN'])
                entry['cap500_vs1000_AP_small_delta_pp']=100*(mi['AP_small']-m500['AP_small'])
            result[arm][condition]=entry
            pred_path=OUT/'evaluation'/arm/(condition+'_predictions_input.json')
            json_write(pred_path,ps);json_write(OUT/'evaluation'/arm/(condition+'_metrics.json'),entry)
            del ev,ev500,ps,orig,ps500,orig500
        del model;torch.cuda.empty_cache()
    return result,forwards

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Exclusive scientific stage lock; never silently rerun or overwrite completed forwards.
    with (OUT/'one_shot.lock').open('x') as f:f.write(str(os.getpid()))
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert json.loads((ROOT/'receipts/queue_completed.json').read_text())['status']=='COMPLETED_BASELINES_AND_INITIAL_DIAGNOSTIC'
    assert features.check('webp')
    torch.set_num_threads(4);cv2.setNumThreads(0);torch.cuda.reset_peak_memory_stats()
    frozen=dict(protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),pid=os.getpid(),
                environment=dict(pillow=pillow_version,webp=features.version('webp'),jpeg=features.version('jpg'),opencv=cv2.__version__,torch=torch.__version__),
                original_science_files_untouched=True,training_updates=0)
    json_write(OUT/'frozen.json',frozen)
    json_write(OUT/'fixture.json',fixture());event('PREFLIGHT_PASS')
    cal=calibration(protocol);meta,rates=observations(cal);result,forwards=evaluate(meta,protocol)
    for arm in ['D0','DA']:
        result[arm]['source_q20_cap1000_gap_pp']=100*(result[arm]['source']['AP_small_input']-result[arm]['q20']['AP_small_input'])
    summary=dict(status='COMPLETED',frozen=frozen,chosen_qualities=cal['chosen_qualities'],rates=rates,metrics=result,
                 detector_forwards=forwards,training_updates=0,elapsed_seconds=time.monotonic()-START,
                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),P2_region_selection_validated=False,P3_deployable_algorithm_validated=False)
    json_write(OUT/'completed.json',summary);event('COMPLETED',detector_forwards=forwards)
    print(json.dumps(dict(status='COMPLETED',qualities=cal['chosen_qualities'],rates=rates,
              DA={c:result['DA'][c]['AP_small_input'] for c in ['source','q20','webp','half_jpeg']})),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        json_write(OUT/'failed.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START))
        raise
