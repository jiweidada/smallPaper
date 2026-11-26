"""Four fresh-model CPU engineering forwards; no trained weights or parameter changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from runtime import HERE,activate_snapshot,verify_sources
from results import read,write,sha
from stage_trace import StageTrace,TracedLoader


def state_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    activate_snapshot()
    from geometry import ResidualGeometryDEIM
    torch.set_num_threads(4)
    out=HERE/'engineering_actual_stage_interface'
    out.mkdir(exist_ok=False)
    annotation=out/'synthetic_annotation.json'
    write(annotation,{'images':[{'id':0,'height':640,'width':640}],'annotations':[]})
    sample=torch.linspace(0,1,3*640*640).reshape(1,3,640,640)
    batches=[(sample,[{'image_id':torch.tensor(0),'orig_size':torch.tensor([640,640])}])]
    checks=[];states={}
    for arm in ('baseline','fine_residual'):
        plan=read(HERE/f'configs/{arm}_seed1.json')
        assert plan['model_recipe']['HGNetv2']['pretrained'] is False
        torch.manual_seed(810218)
        model=ResidualGeometryDEIM(plan['model_recipe'],arm).cpu().eval()
        before=state_digest(model);rng=torch.get_rng_state().clone()
        with torch.inference_mode(): reference=model(sample)
        identity={'arm':arm,'seed':1,'val_annotation_sha256':sha(annotation)}
        trace=StageTrace(model,identity,annotation,mode='synthetic_engineering')
        with torch.inference_mode():
            for inputs,targets in TracedLoader(batches,trace): observed=model(inputs)
        assert set(reference)==set(observed) and all(torch.equal(reference[k],observed[k]) for k in reference)
        checks.append(arm+': native output bitwise unchanged by observation')
        path=trace.finish(out/arm)
        with np.load(path.parent/'image_000000.npz',allow_pickle=False) as saved:
            assert saved['encoder_initial_boxes'].shape==(300,4)
            assert saved['pre_after_first_layer_boxes'].shape==(300,4)
            assert np.array_equal(saved['final_boxes'],observed['pred_boxes'][0].numpy())
            assert np.array_equal(saved['final_logits'],observed['pred_logits'][0].numpy())
        checks.append(arm+': real decoder interface and final equality verified')
        assert not model._forward_hooks and not model.decoder.decoder._forward_hooks and not model.decoder.decoder.layers[0]._forward_pre_hooks
        checks.append(arm+': all observation hooks removed')
        assert state_digest(model)==before
        checks.append(arm+': all model parameters and persistent buffers unchanged')
        assert torch.equal(torch.get_rng_state(),rng)
        checks.append(arm+': evaluation and observation consume no initialization RNG')
        states[arm]=before
        del model,reference,observed,trace
    assert not torch.cuda.is_initialized()
    sources=verify_sources()
    report=dict(status='PASS_FRESH_CPU_MODEL_INTERFACE_ONLY',checks=checks,check_count=len(checks),
        full_fresh_detector_forwards=4,trained_checkpoint_forwards=0,checkpoint_loads=0,
        gradient_calls=0,optimizer_constructions=0,parameter_updates_after_initialization=0,
        CUDA_initialized=False,training_started=False,native_training_environment_verified=False,
        snapshot='local_candidates_20261004_v2 diagnostic engine; not full training solver',
        source_files_verified=len(sources),initial_state_sha256=states,
        code_sha256={name:sha(HERE/name) for name in ('stage_trace.py','geometry.py','residual.py','check_actual_stage_interface.py')})
    write(out/'report.json',report)
    print(json.dumps({'status':report['status'],'checks':len(checks),'fresh_forwards':4},ensure_ascii=False))


if __name__=='__main__':main()
