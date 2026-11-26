from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;V=P/'capacity_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not V.exists();prior=json.loads((P/'capacity_v1_result.json').read_text())
    assert prior['status']=='FAILED' and prior['whole_model_forwards']==1 and prior['capacity_only_optimizer_updates']==0
    assert "KeyError: 'pred_corners'" in prior['traceback'];V.mkdir()
    text=(P/'capacity_A54.py').read_text();text=text.replace('import gc,hashlib,json,os,time,traceback','import gc,hashlib,json,os,time,traceback\nfrom pathlib import Path')
    text=text.replace('from runtime_A54 import ROOT,A49,ASSETS,native,components,validate_files','from runtime_A54 import ROOT as PROJECT_ROOT,A49,ASSETS,native,components,validate_files\nROOT=Path(__file__).resolve().parent')
    text=text.replace("validate_files('capacity_protocol_FROZEN_v1.json')","validate_files('capacity_v2/capacity_protocol_FROZEN_v2.json')")
    text=text.replace("        model=model.cuda();criterion=criterion.cuda();model.eval()", """        zero_identity=[]
        if module is not None:
            original_correct=module.correct
            def observed_correct(fmap,refs,logits):
                corrected=original_correct(fmap,refs,logits)
                if float(module.gain.detach())==0.:
                    same=torch.equal(corrected,refs)
                    zero_identity.append(bool(same));assert same,'Nonidentity zero-gain references'
                return corrected
            module.correct=observed_correct
        model=model.cuda();criterion=criterion.cuda();model.eval()""")
    old="""        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):outputs=model(eval_images)
        assert_finite_output(outputs);error={}
        if reference is None:reference={k:outputs[k].clone() for k in ['pred_logits','pred_boxes']}
        else:
            error={k:float((outputs[k]-reference[k]).abs().max()) for k in reference}
            assert max(error.values())<=1e-6,error
        native.save_json(ROOT/'capacity_progress.json',{'status':'RUNNING','stage':arm+'_INITIAL_EVAL','pid':os.getpid(),'whole_model_forwards':native.MODEL_COUNTER['forwards'],'optimizer_updates':UPDATES,'initial_error':error})
        del outputs"""
    new="""        error={}
        if arm!='Q':
            with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):outputs=model(eval_images)
            # Native eval intentionally has no pred_corners. Do not use a training-only checker.
            assert all(torch.isfinite(outputs[k]).all() for k in ['pred_logits','pred_boxes'])
            if reference is None:reference={k:outputs[k].clone() for k in ['pred_logits','pred_boxes']}
            else:
                error={k:float((outputs[k]-reference[k]).abs().max()) for k in reference}
                assert max(error.values())<=1e-6,error
            del outputs
        native.save_json(ROOT/'capacity_progress.json',{'status':'RUNNING','stage':arm+'_INITIAL_STATE','pid':os.getpid(),'whole_model_forwards':native.MODEL_COUNTER['forwards'],'optimizer_updates':UPDATES,'initial_error':error})"""
    assert old in text;text=text.replace(old,new)
    text=text.replace("        peak=torch.cuda.max_memory_allocated();assert peak<24*(1<<30)","        assert zero_identity and all(zero_identity)\n        peak=torch.cuda.max_memory_allocated();assert peak<24*(1<<30)")
    text=text.replace("'updates':3,'losses':loss_log", "'updates':3,'zero_gain_native_reference_identity':list(zero_identity),'full_FP32_output_comparison_performed':arm=='M','losses':loss_log")
    text=text.replace("native.digest(ROOT/'capacity_protocol_FROZEN_v1.json')","native.digest(ROOT/'capacity_protocol_FROZEN_v2.json')")
    text=text.replace("'capacity_only_optimizer_updates':UPDATES,\n        'new_module_initial_tensors_match'", "'capacity_only_optimizer_updates':UPDATES,'prior_harness_failure_forwards':1,'cumulative_A54_capacity_forwards':native.MODEL_COUNTER['forwards']+1,\n        'new_module_initial_tensors_match'")
    text=text.replace("result['whole_model_forwards']==9 and UPDATES==6","result['whole_model_forwards']==8 and UPDATES==6")
    ast.parse(text);(V/'capacity_A54_v2.py').write_text(text,encoding='utf-8')
    launcher=(P/'launch_capacity_A54.py').read_text().replace("P==Path('/root/uav_a54_20261101_v1')","P==Path('/root/uav_a54_20261101_v1/capacity_v2')")
    launcher=launcher.replace("env['PYTHONDONTWRITEBYTECODE']='1'","env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(P.parent)")
    launcher=launcher.replace("capacity_A54.py","capacity_A54_v2.py").replace("capacity_protocol_FROZEN_v1.json","capacity_protocol_FROZEN_v2.json")
    ast.parse(launcher);(V/'launch_capacity_A54_v2.py').write_text(launcher,encoding='utf-8')
    (V/'observe_capacity_A54_v2.py').write_bytes((P/'observe_capacity_A54.py').read_bytes())
    oldplan=json.loads((P/'capacity_protocol_FROZEN_v1.json').read_text());own={k:v for k,v in oldplan['files_sha256'].items() if k!='capacity_A54.py'}
    own['capacity_v2/capacity_A54_v2.py']=sha(V/'capacity_A54_v2.py');own['capacity_v2/launch_capacity_A54_v2.py']=sha(V/'launch_capacity_A54_v2.py')
    obj=dict(oldplan,scope='CORRECTED_CAPACITY_HARNESS_SAME_ALGORITHM',maximum_whole_model_forwards_this_process=8,maximum_whole_model_forwards=9,previous_counted_forwards=1,
        correction='Eval output checker handles only native eval keys. B/M full FP32 output comparison; both M/Q zero-gain native reference identity checked inside actual calls. Q redundant initial eval omitted to retain cumulative9 budget, retaining all three dense train-capacity batches for each arm.',
        previous_capacity_result_sha256=sha(P/'capacity_v1_result.json'),files_sha256=own)
    obj['external_sha256']=dict(oldplan['external_sha256']);obj['external_sha256']['/root/uav_a54_20261101_v1/capacity_result.json']=sha(P/'capacity_v1_result.json')
    (V/'capacity_protocol_FROZEN_v2.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'new_forward_limit':8,'previous_forward_count':1,'total_phase_limit':9,'update_limit':6,'algorithm_module_unchanged':sha(P/'scene_scale_prior.py')==oldplan['files_sha256']['scene_scale_prior.py'],'v2_protocol_sha256':sha(V/'capacity_protocol_FROZEN_v2.json')}))
if __name__=='__main__':main()
