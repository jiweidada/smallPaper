"""Pure NumPy synthetic tests; never runs an old baseline or recomputes native AP."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import geometry_core as a

HERE=Path(__file__).resolve().parent
checks=[]


def check(name,condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejects(name,call):
    try:
        call()
    except (ValueError,RuntimeError):
        checks.append(name)
        return
    raise AssertionError('Expected rejection: '+name)


def observed(initial,pre,final,hw):
    count=len(initial)
    logits=np.zeros((count,10),dtype=np.float32)
    logits[:,2]=2
    return dict(zip(a.STAGES,[np.array(v,dtype=np.float32) for v in (initial,pre,final)]),
                original_hw=np.array(hw,dtype=np.int64),
                pre_after_first_layer_logits=logits.copy(),final_logits=logits.copy())


def main():
    dt=np.array([[0,0,2,2],[1,0,2,2],[10,10,2,2],[.5,.5,1,1]],dtype=float)
    expected=np.array([[1,1/3,0,.25],[.25,.25,0,1/16]])
    check('independently specified rectangle IoU matrix',
          np.allclose(a.iou_matrix([[0,0,2,2],[0,0,4,4]],dt),expected,rtol=0,atol=1e-15))
    check('degenerate boxes give zero support',np.array_equal(a.iou_matrix([[0,0,0,0]],[[0,0,0,0]]),[[0]]))
    check('rectangular original dimensions restore x and y correctly',
          np.array_equal(a.pixel_xywh([[.5,.5,.5,.5]],[100,200]),[[50,25,100,50]]))
    check('boxes outside the image are not silently clipped',
          np.array_equal(a.pixel_xywh([[0,0,.5,.5]],[100,200]),[[-50,-25,100,50]]))
    rejects('negative predicted width rejected',lambda:a.pixel_xywh([[.5,.5,-.2,.2]],[100,200]))
    rejects('nonfinite coordinates rejected',lambda:a.iou_matrix([[np.nan,0,1,1]],[[0,0,1,1]]))
    annotation={'images':[{'id':0,'width':200,'height':100},{'id':1,'width':80,'height':160}],
        'annotations':[{'id':0,'image_id':0,'category_id':0,'bbox':[0,0,25,12.5],'area':312.5},
            {'id':2,'image_id':0,'category_id':1,'bbox':[100,50,25,12.5],'area':312.5},
            {'id':3,'image_id':0,'category_id':2,'bbox':[100,50,25,12.5],'area':312.5},
            {'id':7,'image_id':1,'category_id':1,'bbox':[20,40,20,40],'area':800}]}
    first=[.0625,.0625,.125,.125]
    same=[.5625,.5625,.125,.125]
    bad=[.875,.125,.125,.125]
    background=[.875,.875,.125,.125]
    tall=[.375,.375,.25,.25]
    control_images={0:observed([first,bad,background],[first,same,background],
                               [first,same,background],[100,200]),
                    1:observed([bad,background],[bad,background],[bad,background],[160,80])}
    candidate_images=deepcopy(control_images)
    candidate_images[0]['encoder_initial_boxes'][1]=same
    candidate_images[1]['encoder_initial_boxes'][0]=tall
    def flags(rows):
        values=np.zeros((4,10),dtype=bool)
        values[rows]=True
        return {'annotation_id':np.array([0,2,3,7]),'category_id':np.array([0,1,2,1]),'hits':values}
    control_flags=flags([2])
    candidate_flags=flags([1])
    original_arrays={i:{k:v.copy() for k,v in data.items()} for i,data in candidate_images.items()}
    control=a.collect(annotation,control_images.__getitem__,control_flags)
    candidate=a.collect(annotation,candidate_images.__getitem__,candidate_flags)
    check('original annotation zero and native false flag preserved',
          candidate['annotation_id'][0]==0 and not candidate['native_TP'][0].any())
    check('shared candidate support does not become two independent TPs',
          candidate['best_query_within_run'][1,2]==candidate['best_query_within_run'][2,2]
          and candidate['native_TP'].sum(0).tolist()==[1]*10)
    check('argmax class does not overwrite native multiclass-output TP',
          candidate['final_best_class_argmax'][1]!=candidate['category_id'][1] and candidate['native_TP'][1,0])
    pair=a.pair_metrics(control,candidate)
    check('early geometry gain without final TP explicitly retained',
          pair['initial_support_gained_but_still_not_TP']==[2]*10)
    check('new and lost native TP both reported',
          pair['new_native_TP']==[1]*10 and pair['lost_native_TP']==[1]*10)
    check('new native TP with early support gain linked by annotation ID',
          pair['new_TP_with_initial_support_gain']==[1]*10
          and pair['transition_annotation_ids']['0.50']['new_TP_with_initial_support_gain']==[2])
    check('lost native TP can coexist with improved early geometry',
          pair['transition_annotation_ids']['0.50']['lost_TP']==[3]
          and pair['lost_TP_with_initial_support_loss']==[0]*10)
    check('analysis never mutates input stage arrays',all(np.array_equal(data[k],original_arrays[i][k])
          for i,data in candidate_images.items() for k in data))
    shuffled={k:v[[3,1,0,2]].copy() for k,v in candidate_flags.items()}
    joined=a.collect(annotation,candidate_images.__getitem__,shuffled)
    check('shuffled native flag rows joined by GT identity',
          all(np.array_equal(joined[k],candidate[k]) for k in joined))
    permuted=deepcopy(candidate_images)
    for data in permuted.values():
        for key in (*a.STAGES,'pre_after_first_layer_logits','final_logits'):
            data[key]=data[key][::-1].copy()
    reordered=a.collect(annotation,permuted.__getitem__,candidate_flags)
    delta=a.pair_metrics(candidate,reordered)
    check('cross-run query permutations do not create geometric or TP changes',
          delta['mean_max_IoU_delta_by_stage']==[0.,0.,0.] and delta['new_native_TP']==[0]*10
          and delta['lost_native_TP']==[0]*10)
    check('query identities can change despite equal per-GT geometry',
          not np.array_equal(candidate['best_query_within_run'],reordered['best_query_within_run']))
    one_ann={'images':[annotation['images'][1]],'annotations':[annotation['annotations'][3]]}
    changed_query=observed([tall,bad],[tall,bad],[bad,tall],[160,80])
    one_flags={'annotation_id':np.array([7]),'category_id':np.array([1]),'hits':np.ones((1,10),dtype=bool)}
    tracked=a.collect(one_ann,lambda _:changed_query,one_flags)
    check('within-run initial-best query can fail while another preserves coverage',
          a.describe(tracked)['initial_best_query_loses_but_other_query_retains_support']==[1]*10)
    invalid_query=observed([tall,bad],[tall,bad],[tall,[.375,.375,.25,-.25]],[160,80])
    invalid_copy=invalid_query['final_boxes'].copy()
    invalid_metrics=a.image_metrics(annotation['images'][1],[annotation['annotations'][3]],invalid_query)
    check('negative-size raw query contributes zero support while valid query survives',
          invalid_metrics['max_iou'][0,2]==1.0 and invalid_metrics['best_query_within_run'][0,2]==0)
    check('invalid raw boxes preserved without absolute values or clipping',
          np.array_equal(invalid_query['final_boxes'],invalid_copy))
    check('empty class group produces null means and zero counts',
          a.describe(candidate,candidate['category_id']==9)['mean_max_IoU_by_stage']==[None]*3
          and a.pair_metrics(control,candidate,candidate['category_id']==9)['new_native_TP']==[0]*10)
    mismatched=deepcopy(candidate_images)
    mismatched[0]['original_hw']=np.array([200,100])
    rejects('WH/HW mixup rejected before metrics',lambda:a.collect(annotation,mismatched.__getitem__,candidate_flags))
    dup=deepcopy(candidate_flags);dup['annotation_id'][1]=0
    rejects('duplicate TP identities rejected',lambda:a.collect(annotation,candidate_images.__getitem__,dup))
    incomplete={k:v[:-1] for k,v in candidate_flags.items()}
    rejects('missing native GT row rejected',lambda:a.collect(annotation,candidate_images.__getitem__,incomplete))
    wrong=deepcopy(candidate_flags);wrong['category_id'][1]=2
    rejects('wrong native GT class rejected',lambda:a.collect(annotation,candidate_images.__getitem__,wrong))
    smaller=deepcopy(candidate);smaller['annotation_id']=np.array([0,2,3,9])
    rejects('different cross-run populations rejected',lambda:a.pair_metrics(control,smaller))
    nonfinite=deepcopy(candidate_images);nonfinite[0]['final_logits'][0,0]=np.nan
    rejects('nonfinite class logits rejected',lambda:a.collect(annotation,nonfinite.__getitem__,candidate_flags))
    impossible=deepcopy(candidate);impossible['native_TP'][3]=True
    check('TP conflicting with recomputed final geometry flagged without changing TP',
          a.describe(impossible)['native_TP_below_recomputed_final_support']==[1]*10
          and impossible['native_TP'][3].all())
    absent_early=deepcopy(candidate);absent_early['max_iou'][:,0]=control['max_iou'][:,0]
    check('final gain without early crossing remains a separate explanation',
          a.pair_metrics(control,absent_early)['new_TP_without_initial_support_gain']==[1]*10)
    check('synthetic analysis imports no torch or evaluator',
          'torch' not in sys.modules and not any(k.startswith('pycocoeval') for k in sys.modules))
    out=HERE/'engineering_synthetic'
    out.mkdir(exist_ok=False)
    a.write(out/'report.json',{'status':'PASS_PURE_SYNTHETIC_GEOMETRY_AND_IDENTITY_JOIN',
        'checks':checks,'check_count':len(checks),'real_baseline_arrays_read':0,
        'real_research_images_read':0,'model_calls':0,'evaluator_calls':0,
        'native_TP_flags_recomputed':False,'training_started':False,
        'formal_fixed12_pair_execution_verified':False,
        'code_sha256':{'geometry_core.py':a.sha(HERE/'geometry_core.py'),
                       'test_geometry_core.py':a.sha(__file__)}})
    print(json.dumps({'status':'PASS_PURE_SYNTHETIC_GEOMETRY_AND_IDENTITY_JOIN','checks':len(checks)}))


if __name__=='__main__':
    main()
