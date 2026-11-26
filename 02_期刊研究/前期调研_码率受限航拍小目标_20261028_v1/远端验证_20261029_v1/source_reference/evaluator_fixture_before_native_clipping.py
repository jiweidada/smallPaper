"""Artificial AP/crowd/boundary/empty-image fixtures. No detector forward or training."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[2]/'tmp/c10_eval_test_deps'))
from evaluate_fixed import coco_eval,match_summary,bootstrap_ap
from ultralytics.data.augment import LetterBox
from ultralytics.utils.instance import Instances
from ultralytics.utils import ops
import torch

# Native validation metadata must reverse the same single letterbox used for pixels and GT.
for height,width in [(540,960),(997,473),(240,320)]:
    labels={'img':np.zeros((height,width,3),dtype=np.uint8),'ratio_pad':(1.,1.),
            'instances':Instances(np.array([[25/width,40/height,30/width,40/height]],dtype=np.float32),
                                  segments=np.zeros((0,1000,2),dtype=np.float32),bbox_format='xywh',normalized=True),
            'cls':np.zeros((1,1),dtype=np.float32)}
    mapped=LetterBox(new_shape=(640,640),auto=False,scaleup=True,center=True)(labels)
    box=torch.from_numpy(mapped['instances'].bboxes.copy())
    restored=ops.scale_boxes((640,640),box,(height,width),ratio_pad=mapped['ratio_pad'])
    assert torch.allclose(restored,torch.tensor([[10.,20.,40.,60.]]),atol=1e-4)
gt={'info':{},'licenses':[],'categories':[{'id':0,'name':'a'},{'id':1,'name':'b'}],
    'images':[{'id':1,'file_name':'artificial1','width':100,'height':100},{'id':2,'file_name':'empty2','width':100,'height':100}],
    'annotations':[{'id':1,'image_id':1,'category_id':0,'bbox':[10,10,4,4],'area':16,'iscrowd':0,'stable_uid':'tiny'},
                   {'id':2,'image_id':1,'category_id':1,'bbox':[30,30,40,40],'area':1600,'iscrowd':0,'stable_uid':'medium'},
                   {'id':3,'image_id':1,'category_id':0,'bbox':[10,50,32,32],'area':1024,'iscrowd':0,'stable_uid':'boundary'},
                   {'id':4,'image_id':1,'category_id':0,'bbox':[60,60,30,30],'area':900,'iscrowd':1,'stable_uid':'ignored'}]}
pred=[{'image_id':1,'category_id':0,'bbox':[10,10,4,4],'score':.9},
      {'image_id':1,'category_id':1,'bbox':[30,30,40,40],'score':.85},
      {'image_id':1,'category_id':0,'bbox':[10,50,32,32],'score':.99},
      {'image_id':1,'category_id':0,'bbox':[70,70,5,5],'score':.8},
      {'image_id':2,'category_id':0,'bbox':[20,20,5,5],'score':.95}]
ev,metrics=coco_eval(gt,pred);matches=match_summary(ev,gt)
assert abs(metrics['AP_small_500']-.5)<1e-12
assert matches['TP']==3 and matches['FN']==0 and matches['FP']==1,matches
assert matches['small_TP']==1 and matches['small_FN']==0 and matches['matched_small_uids']==['tiny']
assert abs(bootstrap_ap(ev,np.arange(2))-metrics['AP_small_500'])<1e-12
assert abs(bootstrap_ap(ev,np.array([0,0]))-1)<1e-12
empty,empty_metrics=coco_eval(gt,[]);empty_matches=match_summary(empty,gt)
assert empty_metrics['AP_small_500']==0 and empty_matches['TP']==0 and empty_matches['FN']==3 and empty_matches['FP']==0
r={'status':'PASS','fixture_kind':'synthetic software evaluation contract, no scientific effect estimate',
   'native_single_letterbox_roundtrip':True,
   'crowd_detection_excluded_from_FP':True,'empty_image_FP_counted':True,'area1024_excluded_from_small':True,
   'exact_full_image_bootstrap_matches_COCO':True,'duplicated_image_bootstrap_checked':True,'empty_predictions_checked':True,
   'AP_small_fixture':metrics['AP_small_500'],'matches':matches,'detector_forwards':0,'training_updates':0}
(ROOT/'evaluator_fixture_result.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r),flush=True)
