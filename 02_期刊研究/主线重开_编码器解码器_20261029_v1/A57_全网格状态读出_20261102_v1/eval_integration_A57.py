"""Exactly one group1 batch per arm; no AP computation or optimization."""
import gc
import hashlib
import os
import traceback
import numpy as np
import torch
from common_A57 import (ROOT,MODEL_COUNTER,make_model,validate_files,save_json,
    VisDroneDataset,collate,seed_all,input_digest,assert_output,digest,set_event_journal)
from evaluate_A57 import bind_ground_truth,postprocess


def main():
    plan=validate_files('eval_integration_scope_FROZEN_v1.json')
    with (ROOT/'eval_integration_once.lock').open('x') as f:f.write(str(os.getpid()))
    set_event_journal(ROOT/'eval_integration_events.jsonl')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    data=VisDroneDataset('val');gt=bind_ground_truth(data)
    # Independent rectangular-image coordinate check; no detector forward.
    logits=torch.full((1,400,10),-80.);logits[0,0,0]=10.
    boxes=torch.zeros(1,400,4);boxes[0,0]=torch.tensor([.5,.25,.2,.1])
    result=postprocess({'pred_logits':logits,'pred_boxes':boxes},torch.tensor([[1920,1080]]))
    assert torch.allclose(result[0]['boxes'][0],torch.tensor([768.,216.,1152.,324.]),rtol=0,atol=3e-4)
    pairs=[data[i] for i in range(4)];images,targets=collate(pairs)
    input_sha=input_digest(images,targets)
    sizes=torch.stack([t['orig_size'] for t in targets])
    assert sizes[0].tolist()==[gt['images'][0]['width'],gt['images'][0]['height']]
    rows=[]
    for arm in ['B','M','Q']:
        save_json(ROOT/'eval_integration_progress.json',{'status':'CONSTRUCTING','arm':arm,
            'model_counter':dict(MODEL_COUNTER),'pid':os.getpid()})
        model=make_model(arm,3).cuda().eval();seed_all(20261101)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            outputs=model(images.cuda())
        assert outputs['num_groups']==1;assert_output(outputs,4,400)
        decoded=postprocess(outputs,sizes.cuda())
        # Use independent scalar/NumPy box conversion at the selected query indices.
        indices=outputs['pred_logits'].float().sigmoid().flatten(1).topk(300,dim=1).indices.cpu().numpy()
        rawboxes=outputs['pred_boxes'].float().cpu().numpy()
        worst=0.0
        for i,row in enumerate(decoded):
            assert len(row['scores'])==len(row['boxes'])==300
            qid=indices[i]//10;xywh=rawboxes[i,qid].astype(np.float64)
            w,h=sizes[i].tolist()
            ref=np.column_stack(((xywh[:,0]-xywh[:,2]/2)*w,(xywh[:,1]-xywh[:,3]/2)*h,
                                 (xywh[:,0]+xywh[:,2]/2)*w,(xywh[:,1]+xywh[:,3]/2)*h))
            got=row['boxes'].detach().cpu().numpy();err=float(np.max(np.abs(ref-got)));worst=max(worst,err)
            assert err<1e-3,(arm,i,err)
            assert np.array_equal(row['labels'].cpu().numpy(),indices[i]%10)
        rows.append({'arm':arm,'initial_state_exact':True,'group_count':1,'logits_shape':list(outputs['pred_logits'].shape),
                     'boxes_shape':list(outputs['pred_boxes'].shape),'decoded_per_image':300,
                     'input_and_target_sha256':input_sha,'max_box_conversion_error_pixels':worst,
                     'finite_outputs':True,'optimizer_updates':0})
        del model,outputs,decoded,rawboxes
        gc.collect();torch.cuda.empty_cache()
    assert MODEL_COUNTER['entries']==MODEL_COUNTER['completions']==3
    validate_files('eval_integration_scope_FROZEN_v1.json')
    save_json(ROOT/'eval_integration_result.json',{
        'status':'PASS_EVAL_INTEGRATION_ONLY','whole_detector_forwards':3,'optimizer_updates':0,
        'COCO_AP_evaluations':0,'model_counter':dict(MODEL_COUNTER),'arms':rows,
        'validation_GT_sha256':digest(ROOT/'validation_GT_FROZEN_v1.json'),
        'identical_to_frozen_prior_GT':True,'images':548,'evaluated_GT':38602,'ignored_valid_GT':157,
        'orig_size_convention':'width,height; verified on rectangular1920x1080 image',
        'scope_sha256':digest(ROOT/'eval_integration_scope_FROZEN_v1.json'),
        'scope':'Initialization/eval-group/decode/GT integration only; no AP or learning claim',
        'formal_training_launched':False})


if __name__=='__main__':
    try:main()
    except BaseException:
        save_json(ROOT/'eval_integration_failure.json',{'status':'FAILED_NO_AUTOMATIC_RETRY',
            'traceback':traceback.format_exc(),'model_counter':dict(MODEL_COUNTER),'optimizer_updates':0})
        raise
