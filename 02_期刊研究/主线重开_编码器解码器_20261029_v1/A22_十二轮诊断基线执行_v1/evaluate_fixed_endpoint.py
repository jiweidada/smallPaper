"""Prespecified endpoint/11-arm analysis; requires a separate post-training allocation."""
from pathlib import Path
import importlib,json,os,sys,traceback,types
from runtime_common import ROOT,A21,host,torch,rd,sha,atomic_json,event,utc,verify_execution_manifest,author_transform,tensor_hash
from data_pipeline import pixel_edits
from prespecified_metrics import ARMS,coco_metrics,learning_gate,proxies,paired_decision
from state_path_bridge import StateBridge,compile_native_bridge,plain,fingerprint

CALLS=0;PHASE='before_evaluation';CAP=0
def enter(m,args):
    global CALLS
    assert CALLS<CAP;CALLS+=1
    event('evaluation_forward_events.jsonl',{'event':'DETECTOR_FORWARD_ENTER','index':CALLS,'phase':PHASE})
def state_norm_controls(base,n,b):
    dn=n['state'].double()-base['state'].double();db=b['state'].double()-base['state'].double()
    nn=float(torch.linalg.vector_norm(dn));nb=float(torch.linalg.vector_norm(db));energy=min(nn,nb)
    ns=(base['state'].double()+dn*(energy/nn if nn else 0.0)).float()
    bs=(base['state'].double()+db*(energy/nb if nb else 0.0)).float()
    an=float(torch.linalg.vector_norm(ns.double()-base['state'].double()))
    ab=float(torch.linalg.vector_norm(bs.double()-base['state'].double()))
    rel=abs(an-ab)/energy if energy else (0.0 if an==ab==0 else float('inf'))
    return {'state':ns,'mean':base['mean']},{'state':bs,'mean':base['mean']}, {
      'raw_norm_N':nn,'raw_norm_B':nb,'ideal_common_norm':energy,'actual_norm_N':an,'actual_norm_B':ab,
      'relative_energy_mismatch':rel,'engineering_fidelity_pass':rel<=.01,
      'scope':'NativeFP32 quantized convex state control; pre-outcome1% energy-fidelity guard, not an effect cutoff'}
def main():
    global PHASE,CAP
    verify_execution_manifest()
    allocation=rd(ROOT/'evaluation_allocation_FROZEN.json')  # deliberately absent during training
    assert allocation['checkpoint_epoch']==12 and allocation['max_complete_forwards']==878
    CAP=878
    assert allocation['source_eval_sha256']==sha(Path(__file__))
    assert allocation['scientific_design_sha256']==sha(ROOT/'scientific_design_PRESPECIFIED.json')
    result=rd(ROOT/'training_result.json');assert result['status']=='COMPLETED_12_EPOCH_PILOT'
    checkpoint=Path(result['fixed_checkpoint']);assert sha(checkpoint)==allocation['checkpoint_sha256']==result['fixed_checkpoint_sha256']
    fd=os.open(str(ROOT/'evaluation_one_shot.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,b'fixed12 endpoint only');os.close(fd)
    torch.set_num_threads(4)
    model,criterion,transform,receipt=host.build_host();del criterion
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert ck['completed_epoch']==12 and ck['counters']['forward_entries']==38832
    assert ck['execution_manifest_sha256']==sha(ROOT/'execution_manifest_FROZEN.json')
    loaded=model.load_state_dict(ck['model'],strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
    del ck
    model.requires_grad_(False);model.eval();model.decoder.min_num_select=model.decoder.max_num_select=500
    model.register_forward_pre_hook(enter)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    free,total=torch.cuda.mem_get_info();assert free>=12*1024**3;model=model.cuda()
    attention=model.backbone.native.layers[0].blocks[0].attn
    native=type(attention).non_casual_linear_attn;patched,patch_audit=compile_native_bridge(attention,host.SOURCE/'mamba2.py')
    bridge=StateBridge();attention._state_path_bridge=bridge
    post=importlib.import_module('src.zoo.dome.postprocessor').DomePostProcessor(num_classes=12,use_focal_loss=True,remap_mscoco_category=False).eval()
    val=rd(ROOT/'validation_asset_manifest.json');cohort=rd(ROOT/'scientific_cohort_asset_manifest.json')['cases']
    cases={x['image']:x for x in cohort};pixel={x['image']:x for x in rd(ROOT/'cohort_pixel_control_preparation.json')['cases']}
    raw={};base_mediators={};base_reads={};detections=[]
    from PIL import Image
    def run(x,mode,donor=None):
        bridge.records.clear();bridge.captured=None;bridge.donor=donor
        bridge.replace_state=mode in ['state','both'];bridge.replace_mean=mode in ['mean','both']
        attention.non_casual_linear_attn=types.MethodType(patched if mode!='native' else native,attention)
        with torch.inference_mode():outputs=model(x.unsqueeze(0).cuda())
        assert list(outputs['pred_boxes'].shape)==[1,500,4] and list(outputs['pred_logits'].shape)==[1,500,12]
        assert bool(torch.isfinite(outputs['pred_boxes']).all()) and bool(torch.isfinite(outputs['pred_logits']).all())
        out={k:plain(outputs[k]).squeeze(0) for k in ['pred_boxes','pred_logits']}
        med={k:v.clone() for k,v in bridge.captured.items()} if bridge.captured is not None else None
        obs=bridge.records[-1] if bridge.records else None
        return out,med,obs,outputs
    for row in val['files']:
        PHASE='endpoint_original:'+row['image']
        p=ROOT/'val_inputs'/row['image'];assert sha(p)==row['image_sha256']
        with Image.open(p) as im:
            assert im.size==(row['width'],row['height']);x=transform(im.convert('RGB')).squeeze(0)
        is_case=row['image'] in cases
        out,med,obs,outputs=run(x,'capture' if is_case else 'native')
        raw[row['image']]=out
        if is_case:
            assert tensor_hash(x)==pixel[row['image']]['original_resized_tensor_sha256']
            base_mediators[row['image']]=med;base_reads[row['image']]=obs
        # Author source uses [W,H] for xyxy scaling; no score threshold, no refill.
        with torch.inference_mode():decoded=post(outputs,torch.tensor([[row['width'],row['height']]],device='cuda'))[0]
        for label,box,score in zip(decoded['labels'].cpu().tolist(),decoded['boxes'].cpu().tolist(),decoded['scores'].cpu().tolist()):
            if 1<=label<=10:
                x1,y1,x2,y2=box;detections.append({'image_id':row['id'],'category_id':label,'bbox':[x1,y1,x2-x1,y2-y1],'score':score})
        del outputs,x
    assert CALLS==548 and len(raw)==548 and len(base_mediators)==33
    metrics=coco_metrics(rd(ROOT/'validation_valid_GT_COCO.json'),detections)
    gate=learning_gate(metrics,raw,cohort)
    atomic_json(ROOT/'fixed12_endpoint_metrics.json',{'utc':utc(),'metrics':metrics,'learning_gate':gate,
       'checkpoint_sha256':result['fixed_checkpoint_sha256'],'complete_model_calls':CALLS,'AP_not_official_VisDrone_benchmark':True})
    torch.save({'outputs':raw,'mediators':base_mediators,'read_fingerprints':base_reads},ROOT/'fixed12_raw_outputs_and_states.pt')
    atomic_json(ROOT/'fixed12_native_top500_detections.json',detections)
    if not gate['qualified']:
        atomic_json(ROOT/'evaluation_result.json',{'utc':utc(),'status':'LEARNING_QUALIFICATION_FAILED_NO_ARCHITECTURE_VERDICT',
          'complete_model_forwards':CALLS,'optimizer_updates':0,'metrics':metrics,'learning_gate':gate,
          'intervention_calls':0,'mainline_validated':False,'no_automatic_training_extension':True});return
    assay=[]
    for case in cohort:
        name=case['image'];p=ROOT/'val_inputs'/name
        with Image.open(p) as im:x=transform(im.convert('RGB')).squeeze(0)
        edits,energy=pixel_edits(x,case)
        assert tensor_hash(edits[0])==pixel[name]['neighbour_edited_tensor_sha256']
        assert tensor_hash(edits[1])==pixel[name]['background_edited_tensor_sha256']
        data={'original':raw[name]};meds={'original':base_mediators[name]};records={}
        for arm,edited in zip(['pixels_N','pixels_B'],edits):
            PHASE=name+':'+arm;out,med,obs,tmp=run(edited,'capture');del tmp
            data[arm]=out;meds[arm]=med;records[arm]=obs
        ns,bs,fidelity=state_norm_controls(meds['original'],meds['pixels_N'],meds['pixels_B'])
        plan=[('state_N','state',meds['pixels_N']),('mean_N','mean',meds['pixels_N']),('both_N','both',meds['pixels_N']),
              ('state_B','state',meds['pixels_B']),('mean_B','mean',meds['pixels_B']),('both_B','both',meds['pixels_B']),
              ('normstate_N','state',ns),('normstate_B','state',bs)]
        for arm,mode,donor in plan:
            PHASE=name+':'+arm;out,med,obs,tmp=run(x,mode,donor);del tmp
            for k in ['fixed_read_Q_pre','fixed_rotated_Q','fixed_local_V','fixed_local_D','native_state','native_mean']:
                assert obs[k]==base_reads[name][k],(name,arm,k)
            data[arm]=out;records[arm]=obs
        assert set(data)==set(ARMS)
        item={'image':name,'prefix':case['prefix'],'primary':not case['filename_prefix_also_in_train'],
          'arms':{arm:proxies(out,case) for arm,out in data.items()},'state_norm_fidelity':fidelity,'pixel_common_energy':energy}
        assay.append(item)
        torch.save({'outputs':data,'read_and_mediator_fingerprints':records},ROOT/('assay_'+Path(name).stem+'.pt'))
        atomic_json(ROOT/'assay_case_results.json',assay)
    assert CALLS==878
    primary=[x for x in assay if x['primary']];secondary=[x for x in assay if not x['primary']]
    assert len(primary)==27 and len(secondary)==6
    decision=paired_decision(primary);sensitivity=paired_decision(secondary)
    fidelity=all(x['state_norm_fidelity']['engineering_fidelity_pass'] for x in primary)
    atomic_json(ROOT/'evaluation_result.json',{'utc':utc(),'status':'PRESPECIFIED_PILOT_ASSAY_COMPLETE',
      'complete_model_forwards':CALLS,'optimizer_updates':0,'primary':decision,'sensitivity':sensitivity,
      'primary_state_norm_control_fidelity':fidelity,'screening_gates_pass':decision['all_gates_pass'] and fidelity,
      'metrics':metrics,'learning_gate':gate,'checkpoint_sha256':result['fixed_checkpoint_sha256'],
      'patch_audit':patch_audit,'mainline_validated':False,
      'scope':'Conditional pilot mechanism screening only; no trained increment or detector AP improvement demonstrated'})

if __name__=='__main__':
    assert sys.argv[1:]==['--run']
    try:main()
    except BaseException:
        atomic_json(ROOT/'evaluation_failure.json',{'utc':utc(),'phase':PHASE,'complete_forward_entries':CALLS,
          'optimizer_updates':0,'traceback':traceback.format_exc(),'no_automatic_retry_or_retuning':True});raise
