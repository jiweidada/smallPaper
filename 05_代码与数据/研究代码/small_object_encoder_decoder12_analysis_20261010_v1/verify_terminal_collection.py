"""Verify downloaded fixed12 evidence and compute paired results on CPU only."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,zipfile
import numpy as np
from results import read,write,load_endpoint,pair
from evaluator_binding import same_result_evaluator
import analyze_endpoint
import input_scale_breakdown

ROOT=Path(__file__).resolve().parents[3]
TERMINAL=('FIXED12_AND_RELOAD_COMPLETE','FIXED12_RELOAD_FAILED','FAILED_TRAINING_NO_FIXED12')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def extract_checked(archive,metadata,destination):
    archive=Path(archive);destination=Path(destination)
    if sha(archive)!=metadata['sha256']:raise ValueError('Archive SHA mismatch')
    if metadata.get('status') not in TERMINAL:raise ValueError('Not a terminal evidence packet')
    assert archive.stat().st_size==metadata['bytes']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len({n.casefold() for n in names})
        assert len(names)==metadata['files'] and z.testzip() is None
        for name in names:
            assert '\\' not in name and (destination/name).resolve().is_relative_to(destination.resolve())
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['status']==metadata['status'] and manifest['seed']==metadata['seed']
        assert set(names)==set(manifest['files'])|{'manifest.json'}
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            assert shutil.disk_usage(destination.parent).free>sum(i.file_size for i in z.infolist())+64*1024**2
            destination.mkdir();z.extractall(destination)
        else:
            assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}==set(names)
    for name,item in manifest['files'].items():
        path=destination/name
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
    return manifest


def reconstruct_matching(endpoint):
    summary,flags,_=load_endpoint(endpoint);ids=flags['annotation_id']
    index={int(g):i for i,g in enumerate(ids)};rebuilt=np.zeros_like(flags['hits']);fp=np.zeros(10,dtype=np.int64)
    with gzip.open(Path(endpoint)/'matching_all_area.json.gz','rt',encoding='utf8') as f:records=json.load(f)
    seen=set();cells=set()
    for r in records:
        assert r['aRng']==[0,1e10] and r['maxDet']==100
        cell=(r['image_id'],r['category_id']);assert cell not in cells;cells.add(cell)
        assert not any(r['gtIgnore'])
        for g in r['gtIds']:
            assert g in index and g not in seen;seen.add(g)
            assert flags['category_id'][index[g]]==r['category_id']
        matches=np.asarray(r['dtMatches']);ignored=np.asarray(r['dtIgnore'],dtype=bool)
        assert matches.shape==ignored.shape==(10,len(r['dtIds']))
        for threshold in range(10):
            fp[threshold]+=int(((matches[threshold]==0)&~ignored[threshold]).sum())
            selected=matches[threshold][(matches[threshold]!=0)&~ignored[threshold]]
            assert np.equal(selected,np.floor(selected)).all()
            for g in selected.astype(np.int64):
                assert g in r['gtIds'] and not rebuilt[index[int(g)],threshold]
                rebuilt[index[int(g)],threshold]=True
    assert seen==set(index) and np.array_equal(rebuilt,flags['hits'])
    assert rebuilt.sum(0).tolist()==summary['TP_by_IoU'] and fp.tolist()==summary['FP_by_IoU']
    return {'images':len(summary['protocol']['image_ids']),'GT':len(ids),'matching_cells':len(cells),
            'GT_hits_reconstructed_exact':True,'FP_reconstructed_exact':True,
            'native_zero_match_sentinel_retained':True,'new_evaluator_calls':0}


def verify(archive,metadata_path,destination,report_path,analysis_output):
    destination=Path(destination);report_path=Path(report_path)
    assert not report_path.exists() and not Path(analysis_output).exists()
    metadata=read(metadata_path);manifest=extract_checked(archive,metadata,destination)
    seed=manifest['seed'];assert seed in (1,2)
    version={1:4,2:3}[seed];assert manifest['runtime_version']==version
    receipt=read(destination/'prepared/receipt.json');identity=receipt['identity'];assert manifest['identity']==identity
    local_code=ROOT/f'英文小论文/code/small_object_encoder_decoder12_20261010_v{version}'
    for name,h in identity['implementation'].items():assert sha(destination/'code'/name)==h==sha(local_code/name)
    assert sha(destination/'prepared/plan.json')==identity['plan_sha256']
    assert sha(destination/'prepared/resolved.yml')==receipt['resolved_sha256']
    assert sha(destination/'prepared/image_manifest.json')==identity['image_manifest_sha256']
    images=read(destination/'prepared/image_manifest.json');assert (len(images['train']),len(images['val']))==(6471,548)
    source=read(destination/'code/source_binding.json')['native_files_sha256'];assert len(source)==468
    for name,h in source.items():assert sha(destination/'native'/name)==h
    for name,h in identity['evaluator_binding']['files'].items():assert sha(destination/'evaluator/pycocoeval'/name)==h
    for name,key in [('baseline_epoch12.pth','baseline_checkpoint_sha256'),('baseline_first100.jsonl','baseline_first100_sha256'),
                      ('train_annotations.json','train_ann_sha256'),('val_annotations.json','val_ann_sha256')]:
        assert sha(destination/'reference'/name)==identity[key]
    common=manifest['common_initialization_reused_not_duplicated'];initial=(ROOT/common['local_relative_path']).resolve()
    assert initial.is_relative_to(ROOT.resolve()) and sha(initial)==common['sha256']==identity['common_init_sha256']
    actual=[json.loads(s) for s in (destination/'run/first100_training_inputs.jsonl').read_text().splitlines()]
    reference=[json.loads(s) for s in (destination/'reference/baseline_first100.jsonl').read_text().splitlines()]
    assert len(actual)<=100 and actual==reference[:len(actual)]
    report={'status':'TERMINAL_ARCHIVE_IDENTITIES_VERIFIED','seed':seed,'archive_sha256':sha(archive),
        'files_verified':len(manifest['files'])+1,'source_files_verified':468,'terminal_state':manifest['status'],
        'input_rows_verified':len(actual),'local_common_initialization_reused':str(initial),
        'model_forwards':0,'evaluator_calls':0,'training_updates':0,'automatic_promotion':False}
    if manifest['status']!='FIXED12_AND_RELOAD_COMPLETE':
        report['status']='FAILED_RUN_EVIDENCE_VERIFIED_NO_EFFICACY_ACCEPTANCE'
        report['reason']='See preserved trainer/reload errors; this does not independently diagnose why the run failed.'
        write(report_path,report);return report
    assert len(actual)==100
    run=destination/'run';done=read(run/'training_completion.json')
    assert done['identity']==identity and done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
    assert sha(run/'last.pth')==done['last_checkpoint_sha256']
    assert read(run/'runtime_start.json')['identity']==identity
    assert read(run/'args.json')['yaml_cfg']==read(destination/'prepared/resolved.yml')
    from verify_gradient_movement import verify as verify_movement
    movement=verify_movement(destination)
    epoch_rows=[json.loads(s) for s in (run/'training_audit_epochs.jsonl').read_text().splitlines()]
    assert len(epoch_rows)==12 and [r['epoch_zero_based'] for r in epoch_rows]==list(range(12))
    for i,row in enumerate(epoch_rows):
        assert row['attempts_total']==row['EMA_updates']==1617*(i+1)
        assert row['actual_optimizer_steps_total']<=row['attempts_total']
        assert all(value['finite'] for value in row['new_parameters'].values())
    replay=destination/'candidate_reload_v1';proof=read(replay/'verification.json')
    assert read(replay/'completion.json')['exit_code']==0 and proof['status']=='PASS_SAVED_CANDIDATE_RELOAD'
    assert proof['checkpoint_sha256']==done['last_checkpoint_sha256']
    assert proof['original_training_completion_sha256']==sha(run/'training_completion.json')
    assert proof['verification_script_sha256']==sha(destination/'tools/verify_candidate_reload.py')
    bound_tools=read(ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1/closeout_tools_v1/metadata.json')['files']
    for name,h in bound_tools.items():assert sha(destination/'tools'/name)==h
    assert proof['optimizer_updates']==proof['epochs_added']==proof['baseline_model_forwards']==0
    assert proof['candidate_model_forwards']==137 and proof['images']==548
    for sub in ('endpoint_epoch12','stages_epoch12'):
        a=read(run/sub/'manifest.json');b=read(replay/sub/'manifest.json')
        assert set(a['files'])==set(b['files'])
        for name,h in b['files'].items():assert sha(replay/sub/name)==h
    base,bh,_=load_endpoint(destination/'baseline_eval/endpoint_epoch12')
    candidate,ch,_=load_endpoint(run/'endpoint_epoch12');reloaded,rh,_=load_endpoint(replay/'endpoint_epoch12')
    same_result_evaluator(base,candidate);same_result_evaluator(candidate,reloaded)
    assert candidate['identity']==reloaded['identity']==identity
    assert max(abs(a-b) for a,b in zip(candidate['native_stats'],reloaded['native_stats']))<=1e-8
    assert all(np.array_equal(ch[k],rh[k]) for k in ch)
    for iid in range(548):
        name=f'image_{iid:06d}.npz'
        with np.load(run/'stages_epoch12'/name,allow_pickle=False) as a,np.load(replay/'stages_epoch12'/name,allow_pickle=False) as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    matching={name:reconstruct_matching(folder/'endpoint_epoch12') for name,folder in
              [('baseline',destination/'baseline_eval'),('candidate',run),('reloaded_candidate',replay)]}
    paired=pair(base,candidate,bh,ch);saved=read(run/'fixed12_comparison.json')
    for name,value in paired.items():assert saved[name]==value,name
    delta=paired['delta_AP_metrics_pp'];passed=delta['APs']>0 and delta['AP']>=0
    assert passed==saved['native_gate_passed']==done['native_gate_passed']
    stage=analyze_endpoint.analyze(destination/'baseline_eval',destination/'reference/val_annotations.json',analysis_output,run)
    scale_output=Path(analysis_output)/'input_scale'
    scale=input_scale_breakdown.run(analysis_output,destination/'reference/val_annotations.json',scale_output)
    report.update(status='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD',
        baseline_overall_percent=base['overall_percent'],candidate_overall_percent=candidate['overall_percent'],
        delta_AP_metrics_pp=delta,native_gate_passed_this_seed=passed,paired_native_results=paired,
        matching_reconstruction=matching,all548_saved_weight_stage_arrays_exact=True,
        gradient_driven_movement=movement,
        stage_analysis=str(Path(analysis_output).resolve()),stage_geometry_TP_conflicts=stage['native_TP_geometry_disagreement_cells'],
        input_scale_analysis=str(scale_output.resolve()),input_scale_analysis_seed=scale['seed'],
        cost={'native_parameters':10231997,'candidate_parameters':13806625,'added_parameters':3574628,
              'parameter_growth_percent':100*3574628/10231997,
              'training_elapsed_seconds_with_observers':read(destination/'screen/candidate_completion.json')['elapsed_seconds'],
              'peak_CUDA_allocated_bytes':max(x['CUDA_peak_allocated_bytes'] for x in epoch_rows),
              'peak_CUDA_reserved_bytes':max(x['CUDA_peak_reserved_bytes'] for x in epoch_rows),
              'inference_latency_benchmark_measured':False},
        final_training_counts={k:epoch_rows[-1][k] for k in ('attempts_total','actual_optimizer_steps_total','EMA_updates','attempts_without_optimizer_step')},
        scope='One seed exploratory fixed12 result. Both seeds required before capacity-matched controls; no significance,stability,novelty or causal-root claim.')
    write(report_path,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','metadata','destination','report','analysis-output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=verify(a.archive,a.metadata,a.destination,a.report,a.analysis_output)
    print(json.dumps({k:r[k] for k in ('status','seed','files_verified')},ensure_ascii=False))
