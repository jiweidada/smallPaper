"""Terminal-checkpoint strict load/optimizer checks; CPU only, no model forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,gc,hashlib,json,pathlib
import torch
from runtime_common import R,MODEL_COUNTER,create_components,make_optimizer,save_json,digest

def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main(arm):
    torch.set_num_threads(2)
    d=R/('arm_'+arm);result=json.loads((d/'result.json').read_text())
    assert result['status']=='COMPLETED' and result['epochs']==12
    plan=json.loads((R/'execution_protocol_FROZEN.json').read_text())
    for n,h in plan['files_sha256'].items():assert digest(R/n)==h,n
    assert not torch.cuda.is_initialized()
    checkpoint=torch.load(d/'epoch12.pth',map_location='cpu',weights_only=True)
    assert checkpoint['epoch']==12 and checkpoint['arm']==arm and checkpoint['optimizer_updates']==9708
    assert checkpoint['ema']['updates']==9708
    assert checkpoint['protocol_sha256']==digest(R/'execution_protocol_FROZEN.json')
    model,criterion,post=create_components(arm)
    initial={k:v.detach().clone()for k,v in model.state_dict().items()}
    info=json.loads((d/'initialization.json').read_text())
    assert info['common_init_sha256']==plan['common_init_sha256']
    if arm!='B':
        assert tensor_digest(model.decoder.decoder.detail_reader.state_dict())==info['branch_initial_tensor_digest']
    checked={}
    for name,state in [('raw',checkpoint['model']),('EMA',checkpoint['ema']['module'])]:
        model.load_state_dict(state,strict=True)
        assert all(torch.isfinite(p).all()for p in model.parameters())
        anchors=state['decoder.anchors'];valid=state['decoder.valid_mask'].expand_as(anchors)
        assert torch.isfinite(anchors[valid]).all() and torch.isposinf(anchors[~valid]).all()
        bad=[k for k,v in state.items()if v.is_floating_point()and not torch.isfinite(v).all()]
        assert set(bad)<={'decoder.anchors'}
        changed=sum(not torch.equal(v,initial[k])for k,v in state.items()if v.is_floating_point())
        checked[name]={'strict_load':True,'all_parameters_finite':True,'expected_masked_anchor_infinity_only':True,'changed_float_state_keys_from_init':changed}
        if arm!='B':
            reader=model.decoder.decoder.detail_reader
            checked[name]['reader_last_weight_nonzero']=int(torch.count_nonzero(reader.mlp[-1].weight))
            assert checked[name]['reader_last_weight_nonzero']>0
    optimizer=make_optimizer(model);optimizer.load_state_dict(checkpoint['optimizer'])
    steps=[int(s['step'])for s in optimizer.state.values()if 'step'in s]
    assert steps and max(steps)==9708 and min(steps)>0
    reader_steps={}
    if arm!='B':
        for n,p in model.decoder.decoder.detail_reader.named_parameters():
            step=int(optimizer.state[p]['step']);assert step==9708;reader_steps[n]=step
    assert MODEL_COUNTER['forwards']==0 and not torch.cuda.is_initialized()
    out={'status':'PASS','arm':arm,'checkpoint_sha256':digest(d/'epoch12.pth'),'raw_and_EMA':checked,'optimizer_steps_min':min(steps),'optimizer_steps_max':max(steps),'reader_optimizer_steps':reader_steps,'reconstructed_initial_reader_matches_saved_digest':True if arm!='B' else None,'new_whole_model_forward_calls':0,'new_optimizer_updates':0,'cuda_initialized':False,'scope':'CPU model construction/strict checkpoint loading, tensor and saved optimizer-state checks only. No inference, backward or optimizer step.'}
    save_json(R/('independent_checkpoint_'+arm+'.json'),out);print(json.dumps(out))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--arm',choices=['B','S','L'],required=True);main(a.parse_args().arm)
