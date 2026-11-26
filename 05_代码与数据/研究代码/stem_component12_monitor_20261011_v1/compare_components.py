"""Compare verified full Stem and all four verified controls; CPU only."""
import argparse,importlib.util,json,sys
from pathlib import Path
from verify_terminal_local import ROOT,CORE,PASS,local_protocol,read,sha,write,native_endpoint,pure_module

def original_reader():
    file=ROOT/'英文小论文/code/original_stem12_monitor_20261011_v1/verify_terminal_local.py'
    spec=importlib.util.spec_from_file_location('original_stem_frozen_reader',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.local_protocol()

def assert_report(report,p,passtag):
    assert report['status']==passtag and report['fixed_endpoint']==p['endpoint']
    assert report['verification_script_sha256']==p['reader_files_sha256']['verify_terminal_local.py']
    assert report['identity']['implementation']==p['implementation_sha256']
    assert report['all548_stage_arrays_reloaded_exact'] and report['first100_inputs_and_pre_forward_RNG_exact']
    assert report['new_model_forwards']==report['new_evaluator_calls']==report['new_optimizer_updates']==report['new_training_epochs']==0

def compare(specpath):
    spec=read(specpath);protocol=local_protocol();old,fullprotocol=original_reader()
    full={int(x['seed']):x for x in spec['full']}
    assert set(full)=={1,2} and len(spec['full'])==2
    controls=spec['controls'];arms=protocol['arms']
    assert len(controls)==8 and {(x['arm'],x['seed']) for x in controls}=={(a,s) for a in arms for s in (1,2)}
    endpoints={};fullreports={};fingerprints=[]
    for seed,item in full.items():
        report=read(item['report']);assert_report(report,fullprotocol,old.PASS)
        assert report['identity']['arm']=='original_stem' and report['identity']['seed']==seed
        assert report['local_reader_protocol_sha256']==sha(old.PROTOCOL)
        packet=Path(item['packet']);assert sha(packet/'terminal_manifest.json')==report['terminal_manifest_sha256']
        ann=read(Path(report['baseline_source_packet'])/'reference/val_annotations.json')
        summary,hits,_=native_endpoint(packet/'run/endpoint_epoch12',ann)
        assert summary['identity']==report['identity'] and summary['overall_percent']==report['candidate_overall_percent']
        endpoints[seed]=(summary,hits,ann);fullreports[seed]=report
        fingerprints.append(dict(path=item['report'],sha256=sha(item['report'])))
    pairing=pure_module('stem_control_result_pairing','results.py');evaluator=pure_module('stem_control_evaluator','evaluator_binding.py')
    rows=[]
    for item in controls:
        report=read(item['report']);assert_report(report,protocol,PASS)
        assert report['local_reader_protocol_sha256']==sha(CORE.parent/'stem_component12_monitor_20261011_v1/local_reader_protocol.json')
        seed=item['seed'];arm=item['arm'];assert report['identity']['seed']==seed and report['identity']['arm']==arm
        packet=Path(report['downloaded_packet']);assert sha(packet/'terminal_manifest.json')==report['terminal_manifest_sha256']
        fullsummary,fullhits,ann=endpoints[seed]
        summary,hits,_=native_endpoint(packet/'run/endpoint_epoch12',ann)
        assert summary['identity']==report['identity'] and summary['overall_percent']==report['candidate_overall_percent']
        for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
            assert summary['identity'][key]==fullsummary['identity'][key]
        evaluator.same_result_evaluator(summary,fullsummary)
        pair=pairing.pair(summary,fullsummary,hits,fullhits)
        rows.append(dict(seed=seed,arm=arm,full=fullsummary['overall_percent'],control=summary['overall_percent'],
            full_minus_control_pp=pair['delta_AP_metrics_pp'],paired_GT_and_class_changes=pair,
            control_minus_BASE12_pp=report['delta_AP_metrics_pp'],full_minus_BASE12_pp=fullreports[seed]['delta_AP_metrics_pp']))
        fingerprints.append(dict(path=item['report'],sha256=sha(item['report'])))
    fullpasses=all(r['native_gate_passed_this_seed'] for r in fullreports.values())
    evidence=[]
    for arm in arms:
        r=[x for x in rows if x['arm']==arm];assert len(r)==2
        evidence.append(dict(arm=arm,consistent_positive_APs_conditional_difference=all(x['full_minus_control_pp']['APs']>0 for x in r),
            AP_nonnegative_both=all(x['full_minus_control_pp']['AP']>=0 for x in r),
            deltas=[dict(seed=x['seed'],**x['full_minus_control_pp']) for x in r]))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return dict(status='VERIFIED_FIXED12_FULL_MINUS_SINGLE_FACTOR_CONTROLS',full_passes_BASE12_both_seeds=fullpasses,
        rows=rows,conditional_evidence=evidence,reports=fingerprints,spec_sha256=sha(specpath),
        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
        limitation='Conditional differences in this matched12-epoch configuration only. Factors can interact; no additive contribution, statistical significance or unique root claim. If full Stem fails BASE12, these differences cannot establish an overall improvement.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=compare(a.spec);write(a.output,result)
    print(json.dumps(dict(status=result['status'],full_passes_BASE12_both_seeds=result['full_passes_BASE12_both_seeds'])))
