"""Exercise actual DEIM paired forward/loss/cache/COCO on synthetic CPU data."""
import argparse
import copy
import datetime
import hashlib
import json
import contextlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import bootstrap
from bootstrap import ROOT
from factory import build_model, build_criterion, load_config, load_common_init, verify_snapshot
from dual_view import DualViewDEIM, paired_loss
from dataset import PairedCocoImages, paired_collate
from evaluate import evaluate_dataset
from run_candidate import optimizer_for, one_step, implementation_hash


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(4); torch.manual_seed(73)
    result=dict(status='running_CPU_only',checks=[],synthetic_data_only=True,
                GPU_capacity_measured=False,APs_improvement_demonstrated=False,
                created=datetime.datetime.now().astimezone().isoformat(),torch=torch.__version__)
    with args.output.open('x',encoding='utf8') as f: json.dump(result,f,indent=2)
    def check(name,condition,**measurements):
        row=dict(name=name,passed=bool(condition),**measurements)
        result['checks'].append(row); args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)
        if not condition: raise AssertionError(name)
    try:
        verify_snapshot()
        def fake_criterion(outputs, targets):
            return {key:torch.tensor(value) for key,value in outputs}
        ordered=paired_loss(fake_criterion,{'global':[('z',2.),('a',4.)], 'local':[('m',6.)]}, {'global':None,'local':None})
        check('paired loss order is deterministic and missing terms stay half-weighted',list(ordered)==['a','m','z'] and [v.item() for v in ordered.values()]==[2.,3.,1.])
        cfg=load_config(ROOT/'configs/C_query_gap.yml')
        materials=args.output.parent/(args.output.stem+'_synthetic_materials')
        materials.mkdir(exist_ok=False)
        result['synthetic_materials']=str(materials.resolve())
        with contextlib.nullcontext(materials) as tmp:
            tmp=Path(tmp)
            (tmp/'SYNTHETIC_TEST_ONLY.txt').write_text('Random image and random model. These COCO numbers are interface tests, not VisDrone results. Preserve inputs and any failures.\n',encoding='utf8')
            # Object extends across a possible crop boundary: keep old clipped-GT rule.
            pixels=np.random.default_rng(73).integers(0,256,(700,900,3),dtype=np.uint8)
            Image.fromarray(pixels).save(tmp/'0.png')
            ann={'images':[{'id':0,'file_name':'0.png','width':900,'height':700}],
                 'categories':[{'id':i,'name':str(i)} for i in range(10)],
                 'annotations':[{'id':0,'image_id':0,'category_id':1,'bbox':[270,200,12,16],'area':192,'iscrowd':0}], 'info':{}}
            (tmp/'ann.json').write_text(json.dumps(ann),encoding='utf8')
            dataset=PairedCocoImages(tmp,tmp/'ann.json',augment=False)
            samples,targets=paired_collate([dataset[0]])
            model=build_model(cfg).eval()
            baseline=DualViewDEIM(copy.deepcopy(model.backbone),copy.deepcopy(model.encoder),copy.deepcopy(model.decoder),mode='independent').eval()
            baseline.interaction.load_state_dict(model.interaction.state_dict())
            with torch.no_grad():
                native=baseline(samples)
                proposed=model(samples)
            check('zero-init matches independent paired native detector exactly',all(torch.equal(native[v][k],proposed[v][k]) for v in ['global','local'] for k in ['pred_logits','pred_boxes']))
            del baseline, native, proposed
            # The native common state can be loaded strictly; extra prototype tensors only.
            native_state={k:v for k,v in model.state_dict().items() if not k.startswith(('interaction.','context_interaction.'))}
            torch.save({'model':native_state},tmp/'synthetic_common.pth')
            receipt=load_common_init(model,tmp/'synthetic_common.pth')
            check('common initialization exact with only interaction extras',receipt['common_tensors_exact']==len(native_state) and all(k.startswith(('interaction.','context_interaction.')) for k in receipt['new_tensors']))
            # Test actual learned (nonzero) interaction, not a vacuous zero-init cache comparison.
            with torch.no_grad(): model.context_interaction.output.weight.normal_(0,.01)
            with torch.no_grad():
                paired=model(samples)
                cached,global_output=model.cache_global(samples['global'])
                local_output=model.predict_local(samples['local'],samples['crop_norm'],cached)
            global_error=max((paired['global'][k]-global_output[k]).abs().max().item() for k in ['pred_boxes','pred_logits'])
            local_error=max((paired['local'][k]-local_output[k]).abs().max().item() for k in ['pred_boxes','pred_logits'])
            check('nonzero context cached evaluation agrees with paired evaluation',global_error<=3e-5 and local_error<=3e-5,max_global_error=global_error,max_local_error=local_error)
            with torch.no_grad():
                old=model.context_mode; model.context_mode='independent'; disabled=model(samples); model.context_mode=old
            check('local pre-detection interaction changes actual predictions',not torch.equal(paired['local']['pred_logits'],disabled['local']['pred_logits']))
            check('local interaction leaves same-weight global inference unchanged',torch.equal(paired['global']['pred_logits'],disabled['global']['pred_logits']))
            # Reset candidate output to zero before the actual loss/optimizer test.
            with torch.no_grad(): model.context_interaction.output.weight.zero_()
            model.context_interaction.trace_enabled=True
            criterion=build_criterion(); criterion.set_epoch(7); criterion.set_model(model); model.train()
            optimizer=optimizer_for(model); scaler=torch.amp.GradScaler('cpu',enabled=False)
            before=model.context_interaction.output.weight.detach().clone()
            step=one_step(model,criterion,optimizer,scaler,samples,targets,cfg,torch.device('cpu'))
            trace=model.context_interaction.last_trace
            check('real native paired loss and optimizer step execute',step['optimizer_step_calls']==1 and np.isfinite(step['loss']),loss=step['loss'])
            check('actual detection loss updates context projection',not torch.equal(before,model.context_interaction.output.weight))
            check('normal plus denoising queries traverse same new block',trace['rho'].shape[1]>300,actual_queries=trace['rho'].shape[1])
            empty=copy.deepcopy(targets)
            for k in ['boxes','labels','area','annotation_ids','iscrowd']: empty['local'][0][k]=empty['local'][0][k][:0]
            step=one_step(model,criterion,optimizer,scaler,samples,empty,cfg,torch.device('cpu'))
            check('empty crop preserves native denoising/loss behavior',step['optimizer_step_calls']==1 and np.isfinite(step['loss']))
            check('context inner projections learn after first output update',model.context_interaction.value.weight.grad is not None and model.context_interaction.value.weight.grad.abs().sum().item()>0)
            model.context_interaction.trace_enabled=False
            eval_result=evaluate_dataset(model,cfg,tmp,tmp/'ann.json',tmp/'eval',torch.device('cpu'))
            check('original-image evaluator produces full twelve-metric schema',len(eval_result['metrics'])==12 and len(eval_result['per_class_APs'])==10)
            check('all fixed windows plus global preserved in evaluation',eval_result['forward_views']==5 and eval_result['images']==1)
            counts={}
            for config in sorted((ROOT/'configs').glob('*.yml')):
                arm_cfg=load_config(config); arm_model=build_model(arm_cfg)
                counts[arm_cfg['arm']]={'active':sum(p.numel() for p in arm_model.parameters() if p.requires_grad),
                                       'stored':sum(p.numel() for p in arm_model.parameters()),
                                       'context_active':sum(p.numel() for p in arm_model.context_interaction.parameters() if p.requires_grad)}
                del arm_model
            check('full-size attention controls have equal active parameter counts',len({counts[k]['active'] for k in ['global','gap','local','support','gap_unit']})==1,counts=counts)
            check('all newly prepared GPU arms remain paused',all(load_config(p)['execution_status']=='paused_design_only' for p in (ROOT/'configs').glob('*.yml')))
        upstream=json.loads((ROOT/'upstream_snapshot.json').read_text(encoding='utf8'))
        unchanged=all(hashlib.sha256((Path(upstream['source'])/row['path']).read_bytes()).hexdigest()==row['sha256'] for row in upstream['files'])
        check('every original v2 source/config/result file remains unchanged',unchanged,files=len(upstream['files']))
        result['status']='passed_CPU_native_integration_only'
    except BaseException as e:
        result.update(status='technical_failure_preserved',error=repr(e)); raise
    finally:
        result['implementation_sha256']=implementation_hash()
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__': main()
