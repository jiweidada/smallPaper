"""CPU tensor-data audit; no model class imports or detector calls."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import sys
import time
import traceback

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def save_new(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)

def audit(arm):
    scope=json.loads((P/'fixed12_audit_scope_FROZEN_v1.json').read_text('utf-8'))
    assert sha(Path(__file__))==scope['local_checkpoint_script_sha256']
    folder=P/('fixed12_'+arm+'_remote_v1')
    manifest=json.loads((folder/'bundle_manifest.json').read_text('utf-8'))
    # Caller downloads and verifies the full terminal bundle and this manifest first.
    d=folder/('arm_'+arm)
    result=json.loads((d/'result.json').read_text())
    assert result['status']=='COMPLETED' and result['epochs']==12 and result['optimizer_updates']==9708
    assert result['full_model_forwards']==10159 and result['protocol_sha256']==scope['formal_protocol_sha256']
    ckpt_path=d/'epoch12.pth';row=manifest['files']['arm_'+arm+'/epoch12.pth']
    assert ckpt_path.stat().st_size==row['bytes'] and sha(ckpt_path)==row['sha256']
    assert sha(P/'common_init_augmented_v1.pth')==scope['augmented_initialization_sha256']
    marker=P/('fixed12_'+arm+'_checkpoint_audit_once.json')
    save_new(marker,dict(arm=arm,checkpoint_loads_allowed=1,initialization_loads_allowed=1,model_constructors=0,model_forwards=0))
    start=time.monotonic()
    try:
        sys.path.insert(0,str(P.parents[1]/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
        import torch
        torch.set_num_threads(4)
        assert not torch.cuda.is_initialized()
        initial=torch.load(P/'common_init_augmented_v1.pth',map_location='cpu',weights_only=True)['model']
        ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=True)
        assert ckpt['arm']==arm and ckpt['epoch']==12 and ckpt['optimizer_updates']==9708
        assert ckpt['protocol_sha256']==scope['formal_protocol_sha256']
        assert ckpt['common_initialization_sha256']==scope['augmented_initialization_sha256']
        assert ckpt['inference_mode']==arm and ckpt['ema']['updates']==9708
        assert len(initial)==674
        checks={}
        sites=['encoder.fpn_blocks.0.cv4.norm','encoder.fpn_blocks.1.cv4.norm',
               'encoder.pan_blocks.0.cv4.norm','encoder.pan_blocks.1.cv4.norm']
        for label,state in [('raw',ckpt['model']),('EMA',ckpt['ema']['module'])]:
            assert set(state)==set(initial)
            changed=[];structural=[];bn={};extra_changed=[]
            for key,value in state.items():
                base=initial[key]
                assert value.shape==base.shape and value.dtype==base.dtype,key
                if value.is_floating_point():
                    assert not torch.isnan(value).any(),key
                    if torch.isinf(value).any():
                        assert key=='decoder.anchors' and torch.equal(value,base),key
                        assert torch.equal(torch.isposinf(value),~state['decoder.valid_mask'].bool().expand_as(value))
                        structural.append(key)
                if not torch.equal(value,base):
                    changed.append(key)
                    if key.startswith('encoder.a58_context.'):extra_changed.append(key)
                if key.endswith('.num_batches_tracked'):
                    bn[key]=int(value)-int(base)
            for site in sites:assert bn[site+'.num_batches_tracked']==9708,(label,site)
            assert changed
            if arm=='B':
                assert float(state['encoder.a58_context.gate'])==0
                if label=='raw':assert not extra_changed
            checks[label]=dict(state_tensors=len(state),changed_state_tensors=len(changed),
                documented_unchanged_inf_buffers=structural,BN_training_increment_histogram=dict(collections.Counter(bn.values())),
                four_encoder_BN_increments={s:bn[s+'.num_batches_tracked'] for s in sites},
                context_gate=float(state['encoder.a58_context.gate']),changed_context_state_keys=extra_changed)
        optim=ckpt['optimizer'];steps=collections.Counter()
        for value in optim['state'].values():
            for key,tensor in value.items():
                if torch.is_tensor(tensor) and tensor.is_floating_point():assert torch.isfinite(tensor).all(),key
            if 'step' in value:steps[int(value['step'])]+=1
        assert steps and max(steps)==9708 and min(steps)>0 and all(s<=9708 for s in steps)
        assigned=[i for g in optim['param_groups'] for i in g['params']]
        assert len(assigned)==len(set(assigned)) and set(optim['state'])<=set(assigned)
        assert not torch.cuda.is_initialized()
        out=dict(status='PASS_FIXED12_CPU_TENSOR_DATA_AUDIT',arm=arm,epoch=12,EMA_updates=9708,
            states=checks,optimizer_step_histogram=dict(steps),optimizer_states=len(optim['state']),
            model_constructors=0,model_forwards=0,optimizer_updates_added=0,
            checkpoint_loads=1,initialization_loads=1,monotonic_seconds=time.monotonic()-start,
            checkpoint_sha256=sha(ckpt_path),scope_sha256=sha(P/'fixed12_audit_scope_FROZEN_v1.json'),
            limits='CPU saved tensor/state integrity only; not a new detector run, causal explanation, efficacy or novelty proof')
        save_new(P/('fixed12_'+arm+'_checkpoint_local_verification_v1.json'),out)
        print(json.dumps(out),flush=True)
    except BaseException:
        save_new(P/('fixed12_'+arm+'_checkpoint_local_failure_v1.json'),dict(status='FAILED_DATA_AUDIT_NO_RETRY',
            arm=arm,model_forwards=0,optimizer_updates_added=0,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['B','M','Q'],required=True)
    audit(parser.parse_args().arm)
