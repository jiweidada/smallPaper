"""Offline description of actual native selector locations, never dense box inference."""
import numpy as np

SHAPES=((80,80),(40,40),(20,20))
LEVELS=('P3','P4','P5','all_levels')
RADII=(0,1)
THRESHOLDS=(.5,.75)

def verify_grid_centers(record, checkpoint_anchors=None):
    """Exact savedEMA equality, then validate nominal feature-cell assignment."""
    anchors=np.asarray(record['dense_anchor_logits'])
    assert checkpoint_anchors is not None, 'Require a checkpoint-bound actual EMA anchor buffer'
    assert anchors.shape==(8400,4) and anchors.dtype==np.float32
    assert np.array_equal(anchors,checkpoint_anchors), 'Observed buffer differs from bound savedEMA'
    valid=np.isfinite(anchors).all(1);invalid=np.isposinf(anchors).all(1);assert (valid|invalid).all() and valid.any()
    expected=[];sizes=[]
    for height,width in SHAPES:
        row,col=np.meshgrid(np.arange(height),np.arange(width),indexing='ij')
        expected.append(np.stack([(col+.5)/width,(row+.5)/height],-1).reshape(-1,2))
        sizes.append(np.tile([width,height],(height*width,1)))
    expected=np.concatenate(expected);sizes=np.concatenate(sizes)
    xy=1/(1+np.exp(-anchors[valid,:2].astype(np.float64)))
    assert np.array_equal(np.floor(xy*sizes[valid]),np.floor(expected[valid]*sizes[valid])), 'EMA center moved into a different nominal cell'
    return float(np.max(np.abs(xy-expected[valid])))


def validate_record(record):
    required={'dense_class_logits','dense_anchor_logits','selected_indices','selected_anchor_logits',
              'selected_bbox_offsets','original_hw','input_hw','image_id'}
    assert set(record)==required
    logits=np.asarray(record['dense_class_logits']);anchors=np.asarray(record['dense_anchor_logits'])
    ids=np.asarray(record['selected_indices'])
    assert logits.shape==(8400,10) and logits.dtype.kind=='f' and np.isfinite(logits).all()
    assert anchors.shape==(8400,4) and anchors.dtype.kind=='f'
    finite=np.isfinite(anchors).all(1);invalid=np.isposinf(anchors).all(1)
    assert (finite|invalid).all(), 'Only finite priors or native +inf-invalid priors supported'
    assert ids.shape==(300,) and ids.dtype==np.dtype(np.int64)
    assert ((0<=ids)&(ids<8400)).all() and len(np.unique(ids))==300
    assert record['selected_anchor_logits'].shape==(300,4)
    assert record['selected_anchor_logits'].dtype==anchors.dtype
    assert np.array_equal(anchors[ids],record['selected_anchor_logits'])
    offsets=np.asarray(record['selected_bbox_offsets'])
    assert offsets.shape==(300,4) and offsets.dtype.kind=='f' and np.isfinite(offsets).all()
    assert np.asarray(record['input_hw']).tolist()==[640,640]
    assert np.asarray(record['original_hw']).shape==(2,) and (np.asarray(record['original_hw'])>0).all()
    assert np.asarray(record['image_id']).shape==()
    selected=np.zeros(8400,dtype=bool);selected[ids]=True
    score=logits.max(1)
    cutoff=score[ids].min()
    assert (score[~selected]<=cutoff).all(), 'Recorded native indices violate actual max-class TopK'
    return logits,finite,ids,selected,score,float(cutoff)

def neighborhood(cx,cy,radius):
    """Nearest cell, or its fixed3x3 window, separately at each native level.

    A midpoint chooses the upper cell; edges clip to the image. This spatial
    neighborhood is a prespecified proxy, not GT ownership or IoU coverage.
    """
    assert radius in RADII and np.isfinite([cx,cy]).all()
    out=[];offset=0
    for height,width in SHAPES:
        col=int(np.clip(np.floor(cx*width),0,width-1))
        row=int(np.clip(np.floor(cy*height),0,height-1))
        rr=np.arange(max(0,row-radius),min(height,row+radius+1),dtype=np.int64)
        cc=np.arange(max(0,col-radius),min(width,col+radius+1),dtype=np.int64)
        out.append(offset+(rr[:,None]*width+cc[None,:]).ravel())
        offset+=height*width
    return out+[np.concatenate(out)]

def image_description(record,annotations,initial_iou):
    logits,valid,ids,selected,score,cutoff=validate_record(record)
    hw=np.asarray(record['original_hw']);height,width=hw.tolist()
    n=len(annotations);initial_iou=np.asarray(initial_iou,dtype=np.float64)
    assert initial_iou.shape==(n,300) and np.isfinite(initial_iou).all()
    assert ((0<=initial_iou)&(initial_iou<=1+1e-12)).all()
    shape=(n,len(RADII),len(LEVELS))
    result={name:np.zeros(shape,dtype=np.int64) for name in
            ('neighborhood_sites','valid_neighborhood_sites','selected_neighborhood_sites',
             'valid_selected_neighborhood_sites','best_max_class_rank_min','best_max_class_rank_max')}
    for name in ('best_max_class_margin_to_cutoff','best_true_class_logit','best_max_class_logit',
                 'other_class_advantage_at_best_max_class_site'):
        result[name]=np.full(shape,np.nan,dtype=np.float64)
    result['selected_local_initial_support_count']=np.zeros((*shape,2),dtype=np.int64)
    result['initial_any_support']=np.stack([(initial_iou>=t).any(1) for t in THRESHOLDS],1)
    for g,a in enumerate(annotations):
        assert int(a['image_id'])==int(record['image_id'])
        category=int(a['category_id']);assert 0<=category<10
        x,y,w,h=map(float,a['bbox']);assert w>0 and h>0 and np.isfinite([x,y,w,h]).all()
        cx,cy=(x+w/2)/width,(y+h/2)/height
        for ri,radius in enumerate(RADII):
            for li,sites in enumerate(neighborhood(cx,cy,radius)):
                k=(g,ri,li);good=sites[valid[sites]]
                result['neighborhood_sites'][k]=len(sites)
                result['valid_neighborhood_sites'][k]=len(good)
                result['selected_neighborhood_sites'][k]=selected[sites].sum()
                result['valid_selected_neighborhood_sites'][k]=selected[good].sum()
                local_rows=np.isin(ids,sites)
                for ti,t in enumerate(THRESHOLDS):
                    result['selected_local_initial_support_count'][(*k,ti)]=((initial_iou[g]>=t)&local_rows).sum()
                if not len(good):continue
                maximum=score[good].max()
                # Inclusive rank interval avoids inventing a tie ordering.
                result['best_max_class_rank_min'][k]=1+(score>maximum).sum()
                result['best_max_class_rank_max'][k]=(score>=maximum).sum()
                result['best_max_class_logit'][k]=maximum
                result['best_max_class_margin_to_cutoff'][k]=float(maximum)-cutoff
                result['best_true_class_logit'][k]=logits[good,category].max()
                peak=good[np.flatnonzero(score[good]==maximum)]
                # Preserve all tied maxima: the weakest other-class advantage
                # at those sites; no purported actual TopK tie reconstruction.
                other=np.delete(logits[peak].astype(np.float64),category,axis=1).max(1)
                result['other_class_advantage_at_best_max_class_site'][k]=max(0.,float((other-logits[peak,category]).min()))
    audit=dict(dense_sites=8400,selected_sites=300,selected_invalid_priors=int((~valid[ids]).sum()),
               native_score_cutoff=cutoff,native_cutoff_tied_sites=int((score==cutoff).sum()),
               selected_ids_used_without_reranking=True,dense_predicted_boxes_available=False)
    return result,audit

