"""Five untrained-host calls solely to validate mediator instrumentation, not effects."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parent
A20=ROOT.parent/'a20_vssd_host_capacity_20261030_v1'
sys.path.insert(0,str(A20))
import a20_host as host
import torch
from PIL import Image
from state_path_bridge import StateBridge,attach_recorder,compile_native_bridge,fingerprint,plain

CALLS=0
STAGE='before_initialization'
RESULTS=[]
def utc():return datetime.now(timezone.utc).isoformat()
def event(x):
    x={'utc':utc(),**x}
    with (ROOT/'engineering_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(x),flush=True)
def enter(m,args):
    global CALLS
    assert CALLS<5;CALLS+=1;event({'event':'COMPLETE_FORWARD_ENTER','index':CALLS,'stage':STAGE})
def outputs_equal(a,b):
    assert set(a)==set(b)
    for k in a:assert torch.equal(a[k],b[k]),('Output differs',k)
def main():
    global STAGE
    protocol=json.loads((ROOT/'engineering_protocol_FROZEN.json').read_text(encoding='utf-8'))
    assert protocol['status']=='FROZEN_BEFORE_ANY_A21_FORWARD'
    for row in protocol['files']:assert host.sha(ROOT/row['path'])==row['sha256'],row['path']
    fd=os.open(str(ROOT/'engineering_one_shot.lock'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    os.write(fd,json.dumps({'pid':os.getpid(),'protocol_sha256':host.sha(ROOT/'engineering_protocol_FROZEN.json')}).encode());os.close(fd)
    torch.set_num_threads(4)
    model,criterion,transform,receipt=host.build_host()
    assert receipt['host_state_digest']==protocol['expected_initial_state_digest']
    del criterion
    # Budget-controlled diagnostic mode. All first500 proposals are protected by native code.
    model.decoder.min_num_select=model.decoder.max_num_select=500
    model.eval();model.register_forward_pre_hook(enter)
    attention=model.backbone.native.layers[0].blocks[0].attn
    assert attention.ssd_linear_norm and attention.linear_attn_duality and attention.kwargs['exp_da']
    native_function=type(attention).non_casual_linear_attn
    patched,source_audit=compile_native_bridge(attention,host.SOURCE/'mamba2.py')
    bridge=StateBridge();attention._state_path_bridge=bridge
    data=json.loads((A20/'capacity_inputs/input_manifest.json').read_text())['cases'][0]
    image=A20/'capacity_inputs'/data['image_file'];assert host.sha(image)==data['image_sha256']
    with Image.open(image) as im:original=transform(im.convert('RGB'))
    altered=original.clone();altered[:,:,400:408,400:408]=1.0-altered[:,:,400:408,400:408]
    assert not torch.equal(original,altered)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    free,total=torch.cuda.mem_get_info();assert free>=16*1024**3
    model=model.cuda();original=original.cuda();altered=altered.cuda()
    operator_records=[];module_inputs=[];saved={};mediators={}
    attention.register_forward_pre_hook(lambda m,a:module_inputs.append(fingerprint(a[0])))
    plan=[('native_original',False,False),('noop_original',False,False),('identity_state_and_mean',True,True),
          ('altered_input_capture',False,False),('original_with_altered_mediators',True,True)]
    for name,rs,rm in plan:
        STAGE=name;bridge.replace_state=rs;bridge.replace_mean=rm
        if name=='identity_state_and_mean':bridge.donor=mediators['original']
        if name=='original_with_altered_mediators':bridge.donor=mediators['altered']
        attach_recorder(attention,native_function if name=='native_original' else patched,operator_records)
        x=altered if name=='altered_input_capture' else original
        with torch.inference_mode():out=model(x)
        assert list(out['pred_logits'].shape)==[1,500,12]
        assert list(out['pred_boxes'].shape)==[1,500,4]
        for k in ['pred_logits','pred_boxes']:assert bool(torch.isfinite(out[k]).all())
        saved[name]={k:plain(out[k]) for k in ['pred_logits','pred_boxes']}
        if name=='noop_original':mediators['original']={k:v.clone() for k,v in bridge.captured.items()}
        if name=='altered_input_capture':mediators['altered']={k:v.clone() for k,v in bridge.captured.items()}
        RESULTS.append({'stage':name,'input':fingerprint(x),'logits':fingerprint(out['pred_logits']),
             'boxes':fingerprint(out['pred_boxes']),'operator':operator_records[-1],
             'state_observer':None if name=='native_original' else bridge.records[-1]})
        event({'event':'STAGE_COMPLETE','stage':name,'complete_forward_index':CALLS,'queries':500})
        del out
    outputs_equal(saved['native_original'],saved['noop_original'])
    outputs_equal(saved['native_original'],saved['identity_state_and_mean'])
    assert operator_records[0]['inputs']==operator_records[1]['inputs']==operator_records[2]['inputs']==operator_records[4]['inputs']
    assert module_inputs[0]==module_inputs[1]==module_inputs[2]==module_inputs[4]
    for i in [1,2]:assert operator_records[0]['merged_state_plus_local_skip']==operator_records[i]['merged_state_plus_local_skip']
    for k in ['fixed_read_Q_pre','fixed_rotated_Q','fixed_local_V','fixed_local_D']:
        assert bridge.records[0][k]==bridge.records[1][k]==bridge.records[3][k],k
    assert bridge.records[3]['applied_state']==bridge.records[2]['native_state']
    assert bridge.records[3]['applied_mean']==bridge.records[2]['native_mean']
    assert bridge.records[3]['applied_state']!=bridge.records[0]['native_state'] or bridge.records[3]['applied_mean']!=bridge.records[0]['native_mean']
    assert operator_records[4]['merged_state_plus_local_skip']!=operator_records[0]['merged_state_plus_local_skip']
    assert CALLS==5 and len(operator_records)==5 and len(bridge.records)==4
    torch.save({'outputs':saved,'mediators':mediators},ROOT/'engineering_raw_outputs.pt')
    result={'utc':utc(),'status':'STATE_MEDIATOR_INSTRUMENTATION_PASS_NOT_SCIENTIFIC_EFFECT',
      'protocol_sha256':host.sha(ROOT/'engineering_protocol_FROZEN.json'),'source_audit':source_audit,
      'complete_model_forwards':CALLS,'optimizer_updates':0,'formal_epochs':0,
      'native_noop_full_outputs_bitwise_equal':True,'identity_state_mean_full_outputs_bitwise_equal':True,
      'hybrid_original_attention_inputs_bitwise_equal':True,'hybrid_original_module_input_bitwise_equal':True,
      'hybrid_Q_V_D_unchanged':True,'hybrid_donor_state_and_mean_exact':True,'hybrid_intervention_active':True,
      'fixed_query_count_all_arms':500,'state_block':'backbone.native.layers.0.blocks.0.attn',
      'training_image_only':data['image_file'],'model_is_initial_untrained_detector':True,
      'scientific_outcomes_estimated':False,'GT_or_val_predictions_read':False,
      'native_layer_state_shapes':{k:list(v.shape) for k,v in mediators['original'].items()},
      'stages':RESULTS,'native_calls':len(operator_records),'observer_calls':len(bridge.records),
      'raw_outputs_sha256':host.sha(ROOT/'engineering_raw_outputs.pt'),
      'no_followup_job_or_training':True}
    host.write(ROOT/'engineering_result.json',result)
    event({'event':'ENGINEERING_TERMINAL_PASS','calls':CALLS,'optimizer_updates':0})
if __name__=='__main__':
    try:main()
    except Exception:
        host.write(ROOT/'engineering_result.json',{'utc':utc(),'status':'ENGINEERING_FAILED_STOP_NO_RETRY',
          'stage':STAGE,'complete_model_forwards':CALLS,'optimizer_updates':0,'stages':RESULTS,'traceback':traceback.format_exc()})
        raise
