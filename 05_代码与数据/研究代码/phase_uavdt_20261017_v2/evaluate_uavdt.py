"""Independent native AP70 evaluation of the selected EMA; compressed raw predictions."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from uavdt_protocol import NativeScorer,PROTOCOL,install_native_adapter

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);install_native_adapter()
    from engine.core import YAMLConfig
    from engine.solver.det_engine import _build_resize_pad_metadata
    cfg=YAMLConfig(a.config);settings=cfg.yaml_cfg['val_dataloader']
    ann_path=a.data/'annotations'/(a.split+'.json');annotation=json.loads(ann_path.read_text())
    settings['dataset'].update(img_folder=str(a.data/'images'),ann_file=str(ann_path));settings.update(num_workers=2,total_batch_size=4)
    model=cfg.model;checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert isinstance(checkpoint.get('ema'),dict) and 'module' in checkpoint['ema']
    model.load_state_dict(checkpoint['ema']['module'],strict=True);model=model.cuda().eval();post=cfg.postprocessor.cuda().eval()
    scorer=NativeScorer(annotation);predictions=[];started=time.monotonic();seen=set()
    with torch.inference_mode():
        for samples,targets in cfg.val_dataloader:
            samples=samples.cuda(non_blocking=True);sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
            results=post(model(samples),sizes,for_eval=True,resize_pad=_build_resize_pad_metadata(targets,samples))
            for target,result in zip(targets,results):
                image_id=int(target['image_id'].item());assert image_id not in seen;seen.add(image_id)
                boxes=result['boxes'].cpu().numpy().astype(np.float64);scores=result['scores'].cpu().numpy().astype(np.float64)
                assert len(scores)<=300 and (result['labels'].cpu().numpy()==0).all()
                dt=np.column_stack([boxes[:,:2],boxes[:,2:]-boxes[:,:2],scores]);scorer.update(image_id,dt)
                predictions.append(np.column_stack([np.full(len(dt),image_id,dtype=np.float64),dt]))
    named,recall,precision=scorer.finish();assert seen==set(scorer.images)
    np.savez_compressed(a.out/'predictions.npz',detections=np.concatenate(predictions),columns=np.array(['image_id','x','y','w','h','score']))
    np.savez_compressed(a.out/'native_PR.npz',recall=recall,precision=precision)
    result={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':'UAVDT_v1','split':a.split,
            'images':len(seen),'GT':scorer.count,'checkpoint':str(a.checkpoint),
            'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'checkpoint_epoch_zero_based':checkpoint['last_epoch'],
            'best_stat':checkpoint.get('best_stat'),'selection':'held-out training sequences validation AP70; epoch84..95 stage2 EMA',
            'ground_truth':str(ann_path),'ground_truth_sha256':hashlib.sha256(ann_path.read_bytes()).hexdigest(),
            'metrics_raw_0_to_1':named,'metrics_percent':{k:v*100 for k,v in named.items()},
            'protocol':dict(PROTOCOL,input=[640,640],whole_image_stretch=True,tiling=False,TTA=False,natural_model_queries=300),
            'ignored_predictions':scorer.ignored,'elapsed_seconds':time.monotonic()-started}
    with (a.out/'metrics.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'split':a.split,'metrics':named,'out':str(a.out)}),flush=True)

if __name__=='__main__':main()
