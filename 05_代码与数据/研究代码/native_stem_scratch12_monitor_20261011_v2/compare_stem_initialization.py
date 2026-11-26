"""Compare verified enhanced Stem, native scratch Stem and cached pretrained BASE12."""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from verify_terminal_local import ROOT,HERE,PROTOCOL,PASS,local_protocol,read,sha,write,native_endpoint

def full_reader():
    file=ROOT/'英文小论文/code/original_stem12_monitor_20261011_v1/verify_terminal_local.py'
    spec=importlib.util.spec_from_file_location('enhanced_stem_fixed12_reader',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.local_protocol()

def validate_report(report,protocol,tag,protocol_file):
    assert report['status']==tag and report['fixed_endpoint']==protocol['endpoint']
    assert report['identity']['implementation']==protocol['implementation_sha256']
    assert report['verification_script_sha256']==protocol['reader_files_sha256']['verify_terminal_local.py']
    assert report['local_reader_protocol_sha256']==sha(protocol_file)
    assert report['all548_stage_arrays_reloaded_exact'] and report['first100_inputs_and_pre_forward_RNG_exact']
    assert report['new_model_forwards']==report['new_evaluator_calls']==report['new_optimizer_updates']==report['new_training_epochs']==0

def compare(path):
    spec=read(path);full,fp=full_reader();rp=local_protocol()
    assert len(spec['full'])==len(spec['reference'])==2
    fm={x['seed']:x for x in spec['full']};rm={x['seed']:x for x in spec['reference']}
    assert set(fm)==set(rm)=={1,2}
    rows=[];files=[]
    for seed in (1,2):
        left,right=read(fm[seed]['report']),read(rm[seed]['report'])
        validate_report(left,fp,full.PASS,full.PROTOCOL)
        validate_report(right,rp,PASS,PROTOCOL)
        assert left['identity']['seed']==right['identity']['seed']==seed
        assert left['identity']['arm']=='original_stem' and right['identity']['arm']=='native_stem_scratch'
        for key in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
            assert left['identity'][key]==right['identity'][key],key
        annotation=read(Path(left['baseline_source_packet'])/'reference/val_annotations.json')
        packets=[Path(fm[seed]['packet']),Path(right['downloaded_packet'])]
        summaries=[];hits=[]
        for item,report,reader in zip(packets,[left,right],[full.native_endpoint,native_endpoint]):
            assert sha(item/'terminal_manifest.json')==report['terminal_manifest_sha256']
            summary,flags,_=reader(item/'run/endpoint_epoch12',annotation)
            assert summary['identity']==report['identity'] and summary['overall_percent']==report['candidate_overall_percent']
            summaries.append(summary)
            order=np.argsort(flags['annotation_id']);hits.append({k:flags[k][order] for k in ('annotation_id','category_id','hits')})
        assert np.array_equal(hits[0]['annotation_id'],hits[1]['annotation_id'])
        assert np.array_equal(hits[0]['category_id'],hits[1]['category_id'])
        assert left['baseline_overall_percent']==right['baseline_overall_percent']
        delta={k:summaries[0]['overall_percent'][k]-summaries[1]['overall_percent'][k] for k in summaries[0]['overall_percent']}
        rows.append(dict(seed=seed,enhanced=summaries[0]['overall_percent'],native_scratch=summaries[1]['overall_percent'],
            cached_pretrained_BASE=left['baseline_overall_percent'],enhanced_minus_native_scratch_pp=delta,
            enhanced_minus_pretrained_BASE_pp=left['delta_AP_metrics_pp'],native_scratch_minus_pretrained_BASE_pp=right['delta_AP_metrics_pp'],
            enhanced_new_TP_vs_scratch=(hits[0]['hits']&~hits[1]['hits']).sum(0).tolist(),
            enhanced_lost_TP_vs_scratch=(~hits[0]['hits']&hits[1]['hits']).sum(0).tolist()))
        files.extend(dict(path=str(Path(x).resolve()),sha256=sha(x)) for x in (fm[seed]['report'],rm[seed]['report']))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return dict(status='VERIFIED_FIXED12_ENHANCED_VS_NATIVE_SCRATCH_REFERENCE',seeds=rows,reports=files,spec_sha256=sha(path),
        enhanced_beats_scratch_APs_both_and_AP_nonnegative=all(r['enhanced_minus_native_scratch_pp']['APs']>0 and r['enhanced_minus_native_scratch_pp']['AP']>=0 for r in rows),
        enhanced_passes_original_pretrained_BASE_goal_gate=all(r['enhanced_minus_pretrained_BASE_pp']['APs']>0 and r['enhanced_minus_pretrained_BASE_pp']['AP']>=0 for r in rows),
        new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,new_training_epochs=0,
        limitation='Same-seed default-initialization and shared outside-Stem reference comparison. Different architectures have different random tensors; this does not establish a unique mechanism, statistical significance or an additive contribution. Native scratch is a new reference; the cached pretrained BASE comparison is retained.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=compare(a.spec);write(a.output,result);print(json.dumps(dict(status=result['status'])))
