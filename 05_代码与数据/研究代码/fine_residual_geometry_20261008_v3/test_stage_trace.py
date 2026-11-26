"""Synthetic, parameter-free CPU hook tests; never constructs a detector or optimizer."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import results
import stage_trace
from stage_trace import StageTrace,TracedLoader

HERE=Path(__file__).resolve().parent


class Layer(nn.Module):
    def __init__(self):
        super().__init__(); self.calls=0
    def forward(self,target,reference):
        self.calls+=1
        return target


class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.ModuleList([Layer()]); self.calls=0
        self.skip_layer=False
    def forward(self,samples):
        self.calls+=1
        b=samples.shape[0]
        reference=samples.mean((1,2,3)).reshape(b,1,1).expand(b,300,4).sigmoid()
        target=torch.zeros(b,300,2)
        if not self.skip_layer: self.layers[0](target,reference.unsqueeze(2))
        pre=reference*.9+.02
        logits=torch.arange(10,dtype=torch.float32).reshape(1,1,10).expand(b,300,10)
        return (pre+.001).unsqueeze(0),logits.unsqueeze(0),None,None,pre,logits-.1


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.decoder=nn.Module(); self.decoder.decoder=Decoder()
        self.calls=0; self.mismatch=False
    def forward(self,samples):
        self.calls+=1
        out=self.decoder.decoder(samples)
        return dict(pred_boxes=out[0][-1]+(.1 if self.mismatch else 0),pred_logits=out[1][-1])


class Loader:
    dataset=object()
    def __init__(self,batches):self.batches=batches
    def __iter__(self):return iter(self.batches)
    def __len__(self):return len(self.batches)


def main():
    out=HERE/'engineering_stages'
    out.mkdir(exist_ok=False)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:fn()
        except ValueError:checks.append(name)
        else:raise AssertionError(name+' accepted')
    annotation=out/'synthetic_annotation.json'
    results.write(annotation,{'images':[{'id':i,'height':8,'width':8} for i in (0,1)],'annotations':[]})
    identity={'arm':'baseline','seed':1,'val_annotation_sha256':results.sha(annotation)}
    batches=[(torch.full((1,3,8,8),float(i)),[{'image_id':torch.tensor(i),'orig_size':torch.tensor([8,8])}]) for i in (0,1)]
    original=Loader(batches)
    model=ToyModel().eval()
    check('toy model has no parameters',len(list(model.parameters()))==0)
    with torch.inference_mode():baseline=[model(b[0]) for b in batches]
    before=(model.calls,model.decoder.decoder.calls,model.decoder.decoder.layers[0].calls)
    rng=torch.get_rng_state().clone()
    trace=StageTrace(model,identity,annotation,mode='synthetic_engineering')
    wrapped=TracedLoader(original,trace)
    check('loader length and dataset are preserved',len(wrapped)==len(original) and wrapped.dataset is original.dataset)
    with torch.inference_mode():
        for i,batch in enumerate(wrapped):
            check(f'batch{i} object identity preserved',batch is batches[i])
            output=model(batch[0])
            check(f'batch{i} exact output equality',all(torch.equal(output[k],baseline[i][k]) for k in output))
    after=(model.calls,model.decoder.decoder.calls,model.decoder.decoder.layers[0].calls)
    check('no extra model decoder or layer invocation',tuple(b-a for a,b in zip(before,after))==(2,2,2))
    check('observer consumes no random numbers',torch.equal(torch.get_rng_state(),rng))
    check('observer does not change mode',not model.training and not model.decoder.decoder.training)
    manifest=trace.finish(out/'synthetic_trace')
    trace.close()
    check('all hooks removed and close is idempotent',not model._forward_hooks and not model.decoder.decoder._forward_hooks and not model.decoder.decoder.layers[0]._forward_pre_hooks)
    report=results.read(manifest.parent/'summary.json')
    check('synthetic result explicitly labelled',report['mode']=='synthetic_engineering')
    check('trace identifies pre as after first layer',report['stage_names']==['encoder_initial','pre_after_first_layer','final'])
    check('timing contamination disclosed',report['timing_includes_observation_overhead'])
    with np.load(manifest.parent/'image_000000.npz',allow_pickle=False) as z:
        check('captures actual native-equivalent initial references',np.array_equal(z['encoder_initial_boxes'],np.full((300,4),.5,dtype=np.float32)))
        check('final boxes and logits preserve exact model output',np.array_equal(z['final_boxes'],baseline[0]['pred_boxes'][0].numpy()) and np.array_equal(z['final_logits'],baseline[0]['pred_logits'][0].numpy()))
        check('all expected stage shapes retained',z['pre_after_first_layer_boxes'].shape==(300,4) and z['pre_after_first_layer_logits'].shape==(300,10))
    rejects('synthetic trace cannot become formal evidence',lambda:stage_trace.verify_trace(manifest.parent,identity))
    calls=0
    for label,configure,targets in [
        ('missing stage',lambda m:setattr(m.decoder.decoder,'skip_layer',True),batches),
        ('model decoder disagreement',lambda m:setattr(m,'mismatch',True),batches),
        ('training mode',lambda m:m.train(),batches),
        ('duplicate image',lambda m:None,[batches[0],batches[0]]),
        ('incomplete image coverage',lambda m:None,[batches[0]]),
    ]:
        model=ToyModel().eval(); configure(model)
        observer=StageTrace(model,identity,annotation,mode='synthetic_engineering')
        with torch.inference_mode():
            for samples,target in TracedLoader(Loader(targets),observer):
                model(samples)
        calls+=model.calls
        check(label+' did not suppress native forward',model.calls==len(targets))
        rejects(label+' cannot complete trace',lambda:observer.finish(out/label.replace(' ','_')))
        check(label+' removes hooks after failed finish',not model._forward_hooks)

    # Execute only the two adapter methods with an inert native parent. No fit.
    tree=ast.parse((HERE/'experiment.py').read_text(encoding='utf-8'))
    cls=deepcopy(next(x for x in ast.walk(tree) if isinstance(x,ast.ClassDef) and x.name=='ExperimentSolver'))
    cls.body=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('_train_one_epoch_with_amp_rollback','save_best_model')]
    events=[]
    class NativeStub:
        def _train_one_epoch_with_amp_rollback(self,*args):events.append('native_train_return'); return 'train_result'
        def save_best_model(self,*args):
            check('loader restored before native checkpoint method',self.val_dataloader is original)
            events.append('native_save');return 'native_saved'
    class ObserverStub:
        def __init__(self,*args):events.append('observer_install')
        def close(self):events.append('observer_close')
        def finish(self,*args):events.append('observer_finish');return out/'unused_manifest.json'
    context={'DetSolver':NativeStub,'receipt':{'identity':identity},'paths':{'val_ann':annotation},
        'torch':SimpleNamespace(__version__='stub',version=SimpleNamespace(cuda=None),cuda=SimpleNamespace(get_device_name=lambda _: 'stub'))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'isolated_adapter','exec'),context)
    solver=context['ExperimentSolver']()
    solver.ema=SimpleNamespace(module=object());solver.val_dataloader=original
    solver.output_dir=out;solver.device='stub';solver.evaluator=SimpleNamespace(coco_eval={'bbox':object()})
    with patch.object(stage_trace,'StageTrace',ObserverStub),patch.object(results,'capture',return_value={}),patch.dict(sys.modules,{'torchvision':SimpleNamespace(__version__='stub')}):
        check('nonfinal epoch returns native result',solver._train_one_epoch_with_amp_rollback(130,None)=='train_result')
        check('no nonfinal observer installed',events==['native_train_return'])
        events.clear();solver._train_one_epoch_with_amp_rollback(131,None)
        check('observer installed only after native training returns',events==['native_train_return','observer_install'])
        check('only validation loader wrapped',isinstance(solver.val_dataloader,TracedLoader) and solver.val_dataloader.original is original)
        check('successful adapter returns native save result',solver.save_best_model({}, {},131)=='native_saved')
        check('observer removed before native checkpoint then exported',events[-3:]==['observer_close','native_save','observer_finish'])
        check('successful adapter reports completed capture',solver.endpoint_capture_complete)
        del solver.endpoint_capture_complete
        solver._train_one_epoch_with_amp_rollback(131,None)
        with patch.object(ObserverStub,'finish',side_effect=ValueError('deliberate trace export failure')):
            check('trace export failure still returns to native fit',solver.save_best_model({}, {},131)=='native_saved')
            check('trace export failure remains explicit',not getattr(solver,'endpoint_capture_complete',False) and 'deliberate trace export failure' in solver.endpoint_capture_error)
    check('no CUDA initialization',not torch.cuda.is_initialized())
    check('no evaluator imported',not any(x in sys.modules for x in ('pycocoeval','pycocotools')))
    result=dict(status='PASS_SYNTHETIC_PARAMETER_FREE_CPU_AND_ADAPTER_STUBS',check_count=len(checks),checks=checks,
        parameter_free_toy_model_calls=4+calls,trained_detector_calls=0,optimizer_constructions=0,parameter_updates=0,
        checkpoint_loads=0,native_training_calls=0,native_full_runtime_integration_verified=False,
        capture_code_sha256=results.sha(HERE/'stage_trace.py'),adapter_code_sha256=results.sha(HERE/'experiment.py'),
        test_code_sha256=results.sha(__file__))
    results.write(out/'report.json',result)
    print(json.dumps({'status':result['status'],'checks':len(checks),'toy_calls':4+calls},ensure_ascii=False))


if __name__=='__main__':main()
