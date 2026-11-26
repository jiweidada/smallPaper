"""Explicit metric policy: pinned-toolkit region filter, COCO AP@500.

No detector construction. Bootstrap uses fixed score-tie order and integer
image/prefix multiplicities. The legacy continuous-AP score is descriptive.
"""
import contextlib
import io
import numpy as np

ARMS=['BASE250','NATIVE500','TWOWAY500','ONEWAY500']

def matlab_round(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x>=0,np.floor(x+0.5),np.ceil(x-0.5)).astype(np.int64)

def ignore_integral(raw_rows,height,width):
    rows=np.asarray(raw_rows,dtype=np.float64).reshape(-1,8)
    regions=rows[rows[:,5]==0,:4]
    if not len(regions):return None
    bitmap=np.zeros((height,width),dtype=np.uint8)
    for rect in regions:
        assert np.equal(rect,np.floor(rect)).all(),'Pinned toolkit expects integer region coordinates'
        x,y,w,h=np.maximum(1,rect.astype(np.int64))
        assert x<=width and y<=height
        bitmap[y-1:min(height,y+h),x-1:min(width,x+w)]=1
    return bitmap.cumsum(axis=0,dtype=np.int64).cumsum(axis=1,dtype=np.int64)

def region_keep(xywh,integral):
    boxes=np.asarray(xywh,dtype=np.float64).reshape(-1,4)
    if integral is None:return np.ones(len(boxes),dtype=bool)
    height,width=integral.shape
    pos=np.maximum(1,matlab_round(boxes))
    x=np.clip(pos[:,0],1,width);y=np.clip(pos[:,1],1,height)
    w=pos[:,2];h=pos[:,3]
    x2=np.minimum(width,x+w);y2=np.minimum(height,y+h)
    covered=integral[y-1,x-1]+integral[y2-1,x2-1]-integral[y-1,x2-1]-integral[y2-1,x-1]
    return covered/(h*w)<0.5

def prepare_cases(raw_cases):
    cases=[]
    for i,raw in enumerate(raw_cases):
        rows=np.asarray(raw['rows'],dtype=np.float64).reshape(-1,8)
        integral=ignore_integral(rows,raw['height'],raw['width'])
        keep=(rows[:,5]!=0)&region_keep(rows[:,:4],integral)
        keep &= (rows[:,5]>=1)&(rows[:,5]<=10)&(rows[:,4]>0)
        ids=np.flatnonzero(keep)
        selected=rows[ids]
        assert np.all(selected[:,2:4]>0)
        cases.append({'id':i+1,'file':raw['file'],'prefix':raw['prefix'],
          'width':raw['width'],'height':raw['height'],'integral':integral,
          'gt_xywh':selected[:,:4].copy(),'gt_labels':selected[:,5].astype(np.int64),
          'gt_row_indices':ids,'gt_small':selected[:,2]*selected[:,3]<=1024,
          'raw_valid_GT':int(np.sum((rows[:,4]>0)&(rows[:,5]>=1)&(rows[:,5]<=10)))})
    return cases

def postprocess_fixed(boxes,logits,case,cap=500,argmax_only=False):
    import torch
    from torchvision.ops import box_convert
    b=torch.from_numpy(np.asarray(boxes,dtype=np.float32))
    l=torch.from_numpy(np.asarray(logits,dtype=np.float32))
    assert b.ndim==3 and b.shape[0]==1 and l.shape==(*b.shape[:2],12)
    scores=l.sigmoid()
    if argmax_only:
        score,label=scores.max(-1)
        index=torch.arange(b.shape[1])[None,:]
    else:
        score,flat=scores.flatten(1).topk(cap,dim=-1)
        label=flat%12;index=flat//12
    pixels=box_convert(b,in_fmt='cxcywh',out_fmt='xyxy')
    pixels=pixels*torch.tensor([[case['width'],case['height'],case['width'],case['height']]],dtype=torch.float32)[:,None,:]
    pixels=pixels.gather(1,index[:,:,None].expand(-1,-1,4))
    xywh=box_convert(pixels,in_fmt='xyxy',out_fmt='xywh')[0].numpy().astype(np.float64)
    label=label[0].numpy();score=score[0].numpy().astype(np.float64);index=index[0].numpy()
    valid=(label>=1)&(label<=10)
    keep=valid&region_keep(xywh,case['integral'])
    return {'boxes':xywh[keep],'labels':label[keep],'scores':score[keep],
            'query_ids':index[keep],
            'counts':{'before_label_and_region_filter':len(label),'invalid_labels':int((~valid).sum()),
                      'valid_label_boxes_removed_in_regions':int((valid&~region_keep(xywh,case['integral'])).sum()),
                      'retained':int(keep.sum())}}

def iou_xywh(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1,4);b=np.asarray(b,dtype=np.float64).reshape(-1,4)
    end_a=a[:,:2]+a[:,2:];end_b=b[:,:2]+b[:,2:]
    wh=np.maximum(0,np.minimum(end_a[:,None,:],end_b[None,:,:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    intersection=wh[:,:,0]*wh[:,:,1]
    union=(a[:,2]*a[:,3])[:,None]+(b[:,2]*b[:,3])[None,:]-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)

def coverage_counts(case,base,native,oneway):
    gt=case['gt_xywh'];labels=case['gt_labels']
    matrix=iou_xywh(base['boxes'],gt)
    used=np.zeros(len(gt),dtype=bool)
    for d in np.argsort(-base['scores'],kind='mergesort'):
        eligible=np.flatnonzero((labels==base['labels'][d])&~used)
        if len(eligible):
            g=eligible[np.argmax(matrix[d,eligible])]
            if matrix[d,g]>=0.5:used[g]=True
    protected=np.flatnonzero(used&case['gt_small'])
    lost=[]
    for detections in [native,oneway]:
        overlap=iou_xywh(detections['boxes'],gt)
        lost.append(sum(not np.any((detections['labels']==labels[g])&(overlap[:,g]>=0.5)) for g in protected))
    return {'base_correct_small':len(protected),'native_uncompensated_small_losses':int(lost[0]),
            'oneway_uncompensated_small_losses':int(lost[1])}

def coco_artifacts(cases,detections):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images=[{'id':c['id'],'width':c['width'],'height':c['height']} for c in cases]
    cats=[{'id':i,'name':str(i)} for i in range(1,11)]
    annotations=[]
    for case in cases:
        for box,label in zip(case['gt_xywh'],case['gt_labels']):
            annotations.append({'id':len(annotations)+1,'image_id':case['id'],'category_id':int(label),
                                'bbox':box.tolist(),'area':float(box[2]*box[3]),'iscrowd':0})
    records=[]
    for case,dt in zip(cases,detections):
        for box,label,score in zip(dt['boxes'],dt['labels'],dt['scores']):
            records.append({'image_id':case['id'],'category_id':int(label),'bbox':box.tolist(),'score':float(score)})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset={'info':{},'images':images,'categories':cats,'annotations':annotations};gt.createIndex()
        if records:dt=gt.loadRes(records)
        else:
            dt=COCO();dt.dataset={'info':{},'images':images,'categories':cats,'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=[c['id'] for c in cases];ev.params.catIds=list(range(1,11))
        ev.params.maxDets=[1,10,500];ev.evaluate();ev.accumulate()
    blocks=[]
    for area_index in [0,1]:
        area=ev.params.areaRng[area_index]
        for cat in range(1,11):
            entries=[x for x in ev.evalImgs if x and x['category_id']==cat and x['aRng']==area]
            by_image={x['image_id']:x for x in entries}
            scores=[];tps=[];fps=[];sources=[];ngt=[];presence=[]
            for index,case in enumerate(cases):
                x=by_image.get(case['id'])
                if x is None:
                    ngt.append(0);presence.append(0);continue
                ngt.append(int(np.count_nonzero(np.asarray(x['gtIgnore'])==0)))
                presence.append(int(len(x['gtIds'])>0))
                scores.extend(x['dtScores']);sources.extend([index]*len(x['dtScores']))
                tps.append((x['dtMatches']>0)&~x['dtIgnore'])
                fps.append((x['dtMatches']==0)&~x['dtIgnore'])
            scores=np.asarray(scores,dtype=np.float64);order=np.argsort(-scores,kind='mergesort')
            tp=np.concatenate(tps,axis=1) if tps else np.zeros((10,0),dtype=bool)
            fp=np.concatenate(fps,axis=1) if fps else np.zeros((10,0),dtype=bool)
            blocks.append({'area_index':area_index,'cat':cat,'tp':tp[:,order],'fp':fp[:,order],
                           'image_indices':np.asarray(sources,dtype=np.int64)[order],
                           'ngt':np.asarray(ngt),'presence':np.asarray(presence)})
    reference={}
    for name,area in [('AP',0),('APsmall',1)]:
        values=ev.eval['precision'][:,:,:,area,2]
        reference[name]=float(values[values>-1].mean()) if np.any(values>-1) else 0.0
    values=ev.eval['precision'][0,:,:,0,2]
    reference['AP50']=float(values[values>-1].mean()) if np.any(values>-1) else 0.0
    return blocks,reference

def metrics_from_blocks(blocks,weights):
    weights=np.asarray(weights,dtype=np.int64)
    results={0:[],1:[]};ap50=[];legacy=[];presence_weights=[]
    recall_grid=np.linspace(0,1,101)
    for block in blocks:
        ngt=int(block['ngt']@weights)
        assert ngt>=0
        if ngt==0:continue
        multiplicity=weights[block['image_indices']]
        active=multiplicity>0
        tp=np.cumsum(block['tp'][:,active]*multiplicity[active],axis=1)
        fp=np.cumsum(block['fp'][:,active]*multiplicity[active],axis=1)
        rc=tp/ngt
        pr=tp/(tp+fp+np.spacing(1))
        if pr.shape[1]:pr=np.maximum.accumulate(pr[:,::-1],axis=1)[:,::-1]
        sampled=np.zeros((10,101),dtype=np.float64);continuous=[]
        for threshold in range(10):
            if not rc.shape[1]:continuous.append(0.0);continue
            idx=np.searchsorted(rc[threshold],recall_grid,side='left');valid=idx<rc.shape[1]
            sampled[threshold,valid]=pr[threshold,idx[valid]]
            increments=np.diff(np.r_[0.0,rc[threshold]])
            continuous.append(float(np.sum(increments*pr[threshold])))
        results[block['area_index']].append(float(sampled.mean()))
        if block['area_index']==0:
            ap50.append(float(sampled[0].mean()));legacy.append(float(np.mean(continuous)))
            presence_weights.append(int(block['presence']@weights))
    return {'AP':float(np.mean(results[0])) if results[0] else 0.0,
            'APsmall':float(np.mean(results[1])) if results[1] else 0.0,
            'AP50':float(np.mean(ap50)) if ap50 else 0.0,
            'legacy_presence_weighted_continuous_AP':float(np.average(legacy,weights=presence_weights)) if sum(presence_weights) else 0.0}

def decision(point,interval,coverage):
    gates={
      'baseline_small_exposure':coverage['base_correct_small']>=24 and coverage['prefixes_with_base_small_TP']>=6,
      'native_uncompensated_harm':point['harm_rate']>=0.02 and interval['harm_rate'][0]>0,
      'APsmall_vs_native':point['APsmall_O_minus_N']>=0.005 and interval['APsmall_O_minus_N'][0]>0,
      'AP_no_material_loss':point['AP_O_minus_N']>=0 and interval['AP_O_minus_N'][0]>=-0.002,
      'specificity_vs_group_mask':point['APsmall_O_minus_T']>=0.002 and interval['APsmall_O_minus_T'][0]>0,
      'coverage_vs_base':point['APsmall_O_minus_B']>=0.003 and interval['APsmall_O_minus_B'][0]>0,
    }
    return {'gates':gates,'pilot_all_gates_pass':all(gates.values()),'novelty_cleared':False,'mainline_validated':False}
