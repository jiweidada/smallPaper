"""Once-only recorder preflight on an existing final checkpoint, never training."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from artifact_recorder import ArtifactRecorder, sha

def save(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return [torch.get_rng_state().clone()]+[s.clone() for s in torch.cuda.get_rng_state_all()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native',required=True)
    ap.add_argument('--args',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--checkpoint-sha',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--images',default='0,1')
    args=ap.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    save(output/'start.json',dict(status='RUNNING_ENGINEERING_RECORDER_PREFLIGHT',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_epochs=0,optimizer_steps=0))
    recorder=None
    try:
        native=Path(args.native).resolve()
        source_sha={p.relative_to(native).as_posix():sha(p) for p in sorted((native/'engine').rglob('*.py'))}
        sys.path.insert(0,str(native))
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        from engine.deim.postprocessor import PostProcessor
        from engine.data.transforms._transforms import ConvertPILImage
        from engine.deim.dfine_utils import distance2bbox
        from torchvision.transforms import v2 as T
        from PIL import Image
        saved=json.loads(Path(args.args).read_text())
        cfg=copy.deepcopy(saved['yaml_cfg'])
        if sha(args.checkpoint)!=args.checkpoint_sha: raise ValueError('Wrong checkpoint bytes')
        ann_path=Path(cfg['val_dataloader']['dataset']['ann_file'])
        assert sha(ann_path)=='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
        ann=json.loads(ann_path.read_text())
        assert all(not x.get('iscrowd',0) for x in ann['annotations'])
        info={x['id']:x for x in ann['images']}
        image_ids=list(map(int,args.images.split(',')))
        assert len(image_ids)==len(set(image_ids)) and 1<=len(image_ids)<=8
        backbone=HGNetv2(**cfg['HGNetv2'])
        encoder=HybridEncoder(**cfg['HybridEncoder'],eval_spatial_size=[640,640])
        decoder=DFINETransformer(**cfg['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640])
        model=DEIM(backbone,encoder,decoder,num_classes=10)
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert payload['last_epoch']==131
        model.load_state_dict(payload['ema']['module'],strict=True)
        del payload
        model=model.cuda().eval()
        post=PostProcessor(num_classes=10,num_top_queries=300,use_focal_loss=True,remap_mscoco_category=False,score_threshold=.001).cuda().eval()
        transform=T.Compose([T.Resize([640,640]),ConvertPILImage(dtype='float32',scale=True)])
        initial_state_sha=state_sha(model)
        recorder=ArtifactRecorder(model)
        results=[]
        image_root=Path(cfg['val_dataloader']['dataset']['img_folder'])
        for image_id in image_ids:
            meta=info[image_id]; image_path=image_root/meta['file_name']
            with Image.open(image_path) as im:
                im=im.convert('RGB'); assert im.size==(meta['width'],meta['height'])
                x=transform(im).as_subclass(torch.Tensor).unsqueeze(0).cuda()
            gt=[g for g in ann['annotations'] if g['image_id']==image_id]
            identity=dict(image_id=image_id,file_name=meta['file_name'],source_image_sha256=sha(image_path),original_wh=[meta['width'],meta['height']],input_hw=[640,640],GT_annotation_ids=[g['id'] for g in gt],checkpoint_sha256=args.checkpoint_sha,epoch_one_based=132,weights='EMA',seed=int(saved['seed']),mode='eval_float32',augmented=False,query_identity='same-forward original300 query order only; no cross-checkpoint query identity assumed')
            before_rng=rng_state()
            torch.cuda.synchronize(); tick=time.monotonic()
            with torch.no_grad(): plain=model(x)
            torch.cuda.synchronize(); plain_time=time.monotonic()-tick
            assert all(torch.equal(a,b) for a,b in zip(before_rng,rng_state()))
            recorder.begin(identity,x)
            torch.cuda.reset_peak_memory_stats(); tick=time.monotonic()
            with torch.no_grad(): recorded=model(x)
            torch.cuda.synchronize(); capture_forward_time=time.monotonic()-tick
            assert set(plain)==set(recorded)
            assert all(torch.equal(plain[k],recorded[k]) for k in plain)
            assert all(torch.equal(a,b) for a,b in zip(before_rng,rng_state()))
            # Reconstruct intermediate geometry from integral outputs already
            # produced by this forward. No unexecuted score head is evaluated.
            data=recorder.data
            pre=data['decoder.decoder@0.value.4']
            refs=torch.from_numpy(pre).cuda()
            for i in range(3):
                offsets=torch.from_numpy(data[f'decoder.integral@{i}.value']).cuda()
                box=distance2bbox(refs,offsets,model.decoder.reg_scale)
                recorder.put(f'analysis.FDR{i}.boxes_from_native_integral',box)
                if i==2: assert torch.equal(box,recorded['pred_boxes'])
            for k,v in recorded.items(): recorder.put('final.'+k,v)
            wh=torch.tensor([[meta['width'],meta['height']]],device='cuda')
            predictions=post(recorded,wh)[0]
            for k,v in predictions.items(): recorder.put('postprocessed.'+k,v)
            sample=output/f'image_{image_id:06d}'
            trace=recorder.end(sample)
            save(sample/'GT_original.json',dict(image=meta,annotations=gt,coordinate_format='original pixel xywh; categories0..9',annotation_sha256=sha(ann_path)))
            save(sample/'GT_manifest.json',dict(file='GT_original.json',sha256=sha(sample/'GT_original.json')))
            results.append(dict(image_id=image_id,plain_recorded_output_equal=True,RNG_unchanged=True,GT_count=len(gt),called_modules=len(trace['called_module_counts']),module_invocations=len(trace['invocations']),tensors=trace['tensor_count'],tensor_bytes=trace['tensor_bytes_uncompressed'],npz_bytes=(sample/'tensors.npz').stat().st_size,plain_forward_seconds=plain_time,capture_forward_seconds=capture_forward_time,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,head_calls_added=0))
            print(json.dumps(dict(stage='IMAGE_CAPTURED',**results[-1])),flush=True)
        recorder.close(); recorder=None
        assert state_sha(model)==initial_state_sha
        assert {p.relative_to(native).as_posix():sha(p) for p in sorted((native/'engine').rglob('*.py'))}==source_sha
        completion=dict(status='PASS_EXISTING_CHECKPOINT_RECORDER_PREFLIGHT',results=results,source_python_sha256=source_sha,checkpoint_sha256=args.checkpoint_sha,checkpoint_path=args.checkpoint,checkpoint_last_epoch=131,model_state_sha256_before_after=initial_state_sha,model_buffers_and_parameters_unchanged=True,native_sources_unchanged=True,training_epochs_added=0,optimizer_steps=0,model_forwards=2*len(image_ids),scientific_root_cause_identified=False,training_history_recorded=False,all_leaf_raw_tensors_saved=False,every_named_module_output_summarized=True,structural_raw_tensors_saved=True,extra_prediction_head_calls=0,elapsed_seconds=time.monotonic()-started,limits='Engineering sample only, not full-validation AP or causal attribution. Historical gradients/matching cannot be reconstructed from final weights. Eval intermediate heads unexecuted by native model remain unexecuted. Capture times include synchronous observation overhead and are not a fair latency benchmark.',code_sha256={p.name:sha(p) for p in Path(__file__).resolve().parent.glob('*.py')},environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
        save(output/'completion.json',completion)
        save(output/'manifest.json',dict(files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))
        print(json.dumps(dict(status=completion['status'],elapsed_seconds=completion['elapsed_seconds'],training_epochs=0)),flush=True)
    except BaseException:
        save(output/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=traceback.format_exc(),training_epochs=0))
        raise
    finally:
        if recorder is not None: recorder.close()

if __name__=='__main__': main()
