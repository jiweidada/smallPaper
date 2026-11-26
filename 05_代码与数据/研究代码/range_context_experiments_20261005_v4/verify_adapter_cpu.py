"""Real-sample optimizer/evaluation adapter verification, not GPU acceptance."""
import argparse,json
from pathlib import Path
import torch
import bootstrap
from factory import build_model,build_criterion,load_common_init,load_config,verify_snapshot
from dataset import PairedCocoImages,paired_collate
from run_candidate import one_step,optimizer_for,seed_all
from evaluate import infer_original
from factory import build_postprocessor
from dataset import OriginalCocoImages

def run(a):
    torch.set_num_threads(4)
    seed_all(1);verify_snapshot()
    cfg=load_config(a.config);m=build_model(cfg).cpu().train()
    init=load_common_init(m,a.init,'85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91')
    ds=PairedCocoImages(a.images,a.annotations,augment=False)
    x,t=paired_collate([ds[0]])
    c=build_criterion();c.set_model(m)
    result=one_step(m,c,optimizer_for(m),torch.amp.GradScaler('cpu',enabled=False),x,t,cfg,torch.device('cpu'))
    assert result['optimizer_step_calls']==1
    m.eval();original=OriginalCocoImages(a.images,a.annotations)[0][0]
    prediction,sources,views=infer_original(m,original,cfg,build_postprocessor(),torch.device('cpu'))
    assert len(sources)==views and len(prediction['scores'])<=300 and torch.isfinite(prediction['boxes']).all()
    assert not torch.cuda.is_initialized()
    receipt={'status':'real_CPU_adapter_step_and_original_image_evaluation_pass_not_GPU_capacity',
             'optimizer_step':result,'initialization':init,'evaluation_views_first_image':views,
             'predictions_first_image':len(prediction['scores']),'CUDA_initialized':False,
             'no_trained_checkpoint_saved':True}
    Path(a.output).write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['config','init','images','annotations','output']:p.add_argument('--'+n,required=True)
    run(p.parse_args())
