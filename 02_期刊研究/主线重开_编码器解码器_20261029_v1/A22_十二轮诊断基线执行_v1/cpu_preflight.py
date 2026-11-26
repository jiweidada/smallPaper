"""Critical data/metric/optimizer CPU checks only: no detector forward or optimizer step."""
import contextlib,hashlib,importlib,importlib.metadata,io,json,sys
import numpy as np
from runtime_common import ROOT,host,torch,rd,sha,atomic_json,utc,tensor_hash
from data_pipeline import TrainDataset,pixel_edits
from train_pilot import optimizer_for,warmup_factor
from prespecified_metrics import ARMS,proxies,paired_decision,coco_metrics

def main():
    torch.set_num_threads(4)
    model,criterion,_,receipt=host.build_host()
    assert receipt['host_state_digest']=='8e27a3a06df1f52d91c4190c1f903e4a27566e9d8aaa7b4ce95d30e5f3c76229'
    opt=optimizer_for(model)
    assert abs(warmup_factor(0)-.001)<1e-15 and warmup_factor(999)==warmup_factor(1000)==1
    d0=TrainDataset(1,False);d1=TrainDataset(1,True)
    for index in [0,6470]:
        im,t,m=d0[index];fl,ft,fm=d1[index]
        assert torch.equal(fl,torch.flip(im,[-1])) and torch.equal(t['labels'],ft['labels'])
        assert torch.equal(ft['boxes'][:,0],1-t['boxes'][:,0]) and torch.equal(ft['boxes'][:,1:],t['boxes'][:,1:])
        assert bool((t['boxes'][:,2:]>0).all()) and 1<=int(t['labels'].min())<=int(t['labels'].max())<=10
    # A query cannot own both disjoint GT boxes in the one-to-one pair metric.
    case={'category':1,'target_box800_exact':['0','0','8','8'],'neighbour_box800_exact':['16','0','24','8']}
    boxes=np.zeros((500,4));boxes[0]=[4/800,4/800,8/800,8/800]
    logits=np.zeros((500,12));out={'pred_boxes':boxes,'pred_logits':logits}
    p=proxies(out,case);assert abs(p['J']-1)<1e-14 and abs(p['P']-.5)<1e-14
    boxes[1]=[20/800,4/800,8/800,8/800];p=proxies(out,case);assert abs(p['P']-1)<1e-14
    zero_cases=[{'prefix':str(i),'arms':{a:{'J':.5,'P':.5} for a in ARMS}} for i in range(27)]
    decision=paired_decision(zero_cases);assert not decision['all_gates_pass'] and decision['contrasts']['D']['CI95']==[0.0,0.0]
    for c in zero_cases:
        c['arms']['original']['J']=.8;c['arms']['both_N']['J']=.7;c['arms']['both_B']['J']=.5
        c['arms']['normstate_N']['J']=.8;c['arms']['normstate_B']['J']=.6
    wrong_gain=paired_decision(zero_cases)
    assert wrong_gain['contrasts']['D']['gate'] and not wrong_gain['contrasts']['G']['gate'] and not wrong_gain['all_gates_pass']
    from evaluate_fixed_endpoint import state_norm_controls
    base={'state':torch.zeros(2),'mean':torch.zeros(1)}
    sn,sb,fidelity=state_norm_controls(base,{'state':torch.tensor([3.,4.])},{'state':torch.tensor([0.,2.])})
    assert torch.allclose(sn['state'],torch.tensor([1.2,1.6])) and fidelity['engineering_fidelity_pass']
    gt={'info':{},'images':[{'id':1,'width':200,'height':100}],
        'categories':[{'id':i,'name':str(i)} for i in range(1,11)],
        'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[10,10,8,8],'area':64,'iscrowd':0}]}
    # True positive ranked151 must survive AP@500; AP@100 would miss it.
    det=[{'image_id':1,'category_id':1,'bbox':[100,50,8,8],'score':1-i*.001} for i in range(150)]
    det.append({'image_id':1,'category_id':1,'bbox':[10,10,8,8],'score':.5})
    metric=coco_metrics(gt,det);assert metric['AP50']>0 and metric['APsmall']>0 and metric['maxDets']==500
    empty=coco_metrics(gt,[]);assert empty['AP50']==0
    post=importlib.import_module('src.zoo.dome.postprocessor').DomePostProcessor(num_classes=12,use_focal_loss=True)
    tx=torch.zeros(1,500,12);tx[0,0,1]=10
    tb=torch.zeros(1,500,4);tb[0,0]=torch.tensor([.5,.5,.2,.4])
    decoded=post({'pred_logits':tx,'pred_boxes':tb},torch.tensor([[200,100]]))[0]
    assert int(decoded['labels'][0])==1 and torch.allclose(decoded['boxes'][0],torch.tensor([80.,30.,120.,70.]))
    cohort=rd(ROOT/'scientific_cohort_asset_manifest.json')['cases'];prepared={c['image']:c for c in rd(ROOT/'cohort_pixel_control_preparation.json')['cases']}
    val=rd(ROOT/'validation_asset_manifest.json');transform=d0.transform;from PIL import Image
    for row in val['files']:
        assert sha(ROOT/'val_inputs'/row['image'])==row['image_sha256']
        assert sha(ROOT/'val_inputs'/row['annotation'])==row['annotation_sha256']
    for c in cohort:
        with Image.open(ROOT/'val_inputs'/c['image']) as im:x=transform(im.convert('RGB')).squeeze(0)
        edited,e=pixel_edits(x,c);v=prepared[c['image']]
        assert tensor_hash(x)==v['original_resized_tensor_sha256']
        assert tensor_hash(edited[0])==v['neighbour_edited_tensor_sha256'] and tensor_hash(edited[1])==v['background_edited_tensor_sha256']
    assert not torch.cuda.is_initialized()
    atomic_json(ROOT/'cpu_preflight_result.json',{'utc':utc(),'status':'CPU_DATA_METRIC_AND_CONSTRUCTION_CHECKS_PASS',
      'detector_complete_forwards':0,'optimizer_steps':0,'source_postprocessor_fixture_calls':1,
      'classification_and_host_strict_load':True,'optimizer_groups':len(opt.param_groups),
      'trainable_parameters':sum(p.numel() for g in opt.param_groups for p in g['params']),
      'paired_metric_distinct_query_check':True,'zero_effect_bootstrap_rejected':True,'AP500_rank151_true_positive_retained':True,
      'background_damage_false_positive_rejected':True,'convex_state_norm_control_fixture_pass':True,
      'native_postprocess_WH_scaling_verified':True,'cohort_pixel_hashes_match_all33':True,'validation_assets_hash_verified':548,
      'CUDA_initialized':False,'synthetic_fixture_only_no_real_detection_outcomes':True,
      'versions':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','scipy','Pillow','pycocotools']},
      'installed_COCOeval_source_sha256':sha(importlib.import_module('pycocotools.cocoeval').__file__)})
    print(json.dumps(rd(ROOT/'cpu_preflight_result.json')))
if __name__=='__main__':main()
