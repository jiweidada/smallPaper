"""317/342 fusion, with every coordinate operation in shared geom342."""
import importlib
C=importlib.import_module('343_context')
import torch
G=C.load_geometry(C.HERE/'343_geometry'/'geom342.py')

def fuse(entry,new_boxes=None,new_logits=None):
    gl=entry['final300_logits'];gb=entry['final300_boxes'];wh=entry['orig_wh'];W=entry['window_xyxy_640']
    coords=G.global_queries_to_orig_xyxy(gb,wh)
    logits=gl
    if new_boxes is not None:
        assert new_boxes.shape==(4,4) and new_logits.shape==(4,10)
        coords=torch.cat((coords,G.local_cxcywh_to_orig_xyxy(new_boxes,W,wh)))
        logits=torch.cat((logits,new_logits))
    score,flat=logits.sigmoid().flatten().topk(300)
    query=flat//10;label=flat%10;pre={'scores':score,'flat_index':flat,'labels':label,'query_index':query}
    keep=(score>.001).nonzero().flatten().tolist();counts={};final=[]
    for i in keep:
        c=int(label[i]);n=counts.get(c,0)
        if n<100:final.append(i);counts[c]=n+1
    ix=torch.tensor(final,dtype=torch.long)
    out={'scores':score[ix],'labels':label[ix],'boxes':coords[query[ix]],'flat_index':flat[ix],
         'query_index':query[ix],'is_new':query[ix]>=300}
    G._boxes(out['boxes'])
    return out,pre

def coco_rows(image_id,out):
    xywh=G.orig_xyxy_to_coco_xywh(out['boxes'])
    return [{'image_id':int(image_id),'category_id':int(c),'bbox':b.tolist(),'score':float(s)}
            for c,b,s in zip(out['labels'],xywh,out['scores'])]

def tiny_geo0_group(entry,iou_threshold=.5,tiny_only=True):
    assert len(entry['gt_annotation_ids'])==len(entry['gt_short_side_640_all'])==len(entry['gt_boxes_640'])
    raw=G.raw_query_cxcywh_to_640_xyxy(entry['final300_boxes'])
    iou=G.pairwise_iou(raw,entry['gt_boxes_640'],allow_raw_a=True)
    geo=(iou.max(0).values<iou_threshold)
    result=geo&entry['G_W']
    if tiny_only:result=result&(entry['gt_short_side_640_all']<4)
    return result
