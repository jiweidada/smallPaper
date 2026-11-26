"""Verify each real seed2 tensor packet; do not relax the six-record aggregate contract."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    output=ROOT/'outputs/native_selection_collection12_20261011_v2';spec=read(output/'tensor_reader_protocol.json');assert sha(__file__)==spec['reader_sha256']
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    code=ROOT/'英文小论文/code/native_selection_collection12_20261011_v2';sys.path.insert(0,str(code))
    import verify_local
    selection=module('selection_actual_frozen_core',ROOT/'英文小论文/code/native_selection_analysis12_20261011_v2/selection_core.py')
    tools=module('selection_actual_stage_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py');p,_,_,_,readers=tools.prepared();base=tools.cached_baseline(2,p,readers['full']);ann=base['annotation'];images={i['id']:i for i in ann['images']}
    records=[]
    for arm,kind in [('original_stem','full'),('native_stem_scratch','reference'),('highfreq_rgb','control')]:
        downloads=output/'downloads';name=f'seed2_{arm}_selection_v2';folder=output/f'seed2_{arm}';folder.mkdir(exist_ok=False)
        reader_report=verify_local.verify(downloads/(name+'.zip'),downloads/(name+'_metadata.json'),folder/'extracted',folder/'packet_verification.json')
        diagnostic=Path(reader_report['diagnostic_folder']);identity=reader_report['identity'];reader=readers[kind]
        formal_root=ROOT/f'outputs/stem_fixed12_seed2_formal_collection12_20261011_v1/seed2_{arm}';formal=read(formal_root/'verification.json');rp=reader.local_protocol()
        assert formal['status']==reader.PASS and formal['identity']==identity and formal['all548_stage_arrays_reloaded_exact'] and formal['first100_inputs_and_pre_forward_RNG_exact']
        assert formal['verification_script_sha256']==rp['reader_files_sha256']['verify_terminal_local.py'] and formal['local_reader_protocol_sha256']==sha(reader.PROTOCOL)
        parent=formal_root/'extracted';assert sha(parent/'terminal_manifest.json')==formal['terminal_manifest_sha256']==reader_report['native_terminal_manifest_sha256']
        v=read(diagnostic/'verification.json');summary,flags,checks=readers['control'].native_endpoint(diagnostic/'endpoint_epoch12',ann)
        assert summary['identity']==identity and summary['native_stats']==v['original_stats']==read(parent/'run/endpoint_epoch12/summary.json')['native_stats']
        assert sha(diagnostic/'endpoint_epoch12/precision_recall.npz')==sha(parent/'run/endpoint_epoch12/precision_recall.npz')
        assert sha(diagnostic/'endpoint_epoch12/GT_hits.npz')==sha(parent/'run/endpoint_epoch12/GT_hits.npz')
        readers['control'].stages(diagnostic,identity,ann)
        sm=read(diagnostic/'selection_epoch12/manifest.json');ss=read(diagnostic/'selection_epoch12/summary.json')
        assert sm['trace_code_sha256']==spec['selection_stage_trace_sha256'] and ss['identity']==identity and ss['images']==548 and ss['image_ids']==list(range(548))
        assert ss['dense_locations']==8400 and ss['selected_queries']==300 and ss['additional_head_calls']==ss['additional_topk_calls']==0
        assert ss['cross_run_query_identity_assumed'] is False and ss['epoch_zero_based']==11 and ss['evaluated_weights']=='EMA'
        assert set(sm['files'])=={'summary.json'}|{f'image_{i:06d}.npz' for i in range(548)}
        error=0.;grid_error=0.;invalid_selected=0
        for i in range(548):
            n=f'image_{i:06d}.npz';path=diagnostic/'selection_epoch12'/n;assert sha(path)==sm['files'][n]
            with np.load(path,allow_pickle=False) as z:r={n:z[n] for n in z.files}
            _,valid,ids,_,_,_=selection.validate_record(r);grid_error=max(grid_error,selection.verify_grid_centers(r))
            assert int(r['image_id'])==i and r['original_hw'].tolist()==[images[i]['height'],images[i]['width']]
            invalid_selected+=int((~valid[ids]).sum())
            stage=diagnostic/'stages_epoch12'/n;assert sha(stage)==sha(parent/'run/stages_epoch12'/n)
            with np.load(stage,allow_pickle=False) as z:initial=z['encoder_initial_boxes']
            added=(r['selected_anchor_logits']+r['selected_bbox_offsets']).astype(np.float64)
            reconstructed=1/(1+np.exp(-added));error=max(error,float(np.max(np.abs(reconstructed-initial))))
        assert error<2e-7,'Observed selected priors+offsets do not reproduce encoder references'
        result=dict(status='PASS_ACTUAL548_SELECTION_TENSORS_AND_FORMAL_NATIVE_ENDPOINT_STAGE_JOIN',arm=arm,seed=2,identity=identity,packet_verification_sha256=sha(folder/'packet_verification.json'),formal_verification_sha256=sha(formal_root/'verification.json'),native_endpoint_checks=checks,all548_stage_files_byte_exact=True,all548_dense_arrays_and_recorded_TopK_checked=True,image_ID_size_join_exact=True,selected_prior_plus_offset_sigmoid_max_abs_error=error,reconstruction_tolerance=2e-7,valid_grid_center_max_abs_error=grid_error,invalid_prior_selected_queries=invalid_selected,dense_priors_are_not8400_predicted_boxes=True,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,unique_causal_root_proven=False,two_seed_aggregate_proven=False,reader_sha256=sha(__file__),reader_protocol_sha256=sha(output/'tensor_reader_protocol.json'))
        with (folder/'tensor_verification.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
        records.append(dict(arm=arm,status=result['status'],selected_reference_error=error,report_sha256=sha(folder/'tensor_verification.json')));print(json.dumps(records[-1]),flush=True)
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    with (output/'seed2_all_three_tensor_verification.json').open('x',encoding='utf8') as f:json.dump(dict(status='THREE_SEED2_ACTUAL_SELECTION_PACKETS_VERIFIED_NO_TWO_SEED_AGGREGATE',records=records,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0),f,indent=2)
if __name__=='__main__':main()
