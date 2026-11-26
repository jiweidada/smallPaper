"""Formal savedEMA terminal verification anchored to an already-verified primary packet."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,importlib.util,json,stat,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def module(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def assert_replay(proof,exit,identity,done,init,closeout):
    assert proof['status']=='PASS_SAVED_CANDIDATE_RELOAD' and proof['identity']==identity
    assert proof['checkpoint_sha256']==done['last_checkpoint_sha256'] and done['identity']==identity
    assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12)) and done['status']=='NATIVE_FIT_FINISHED_EXACT12'
    assert proof['strict_loaded_tensors']==init['state_tensors'] and proof['parameters']==init['parameters']
    assert proof['all_parameters_frozen_like_native_EMA'] and proof['native_GT_hits_exact'] and proof['all548_stage_arrays_exact']
    assert proof['stats_max_abs_fraction_difference']==proof['stats_tolerance']==closeout['stats_fraction_tolerance']==0
    assert proof['images']==548 and proof['candidate_model_forwards']==137 and proof['original_stats']==proof['reloaded_stats']
    assert all(v==0 for v in proof['stage_max_abs_difference'].values())
    assert proof['baseline_model_forwards']==proof['optimizer_updates']==proof['epochs_added']==0 and not proof['automatic_retry']
    assert exit['exit_code']==0 and not exit['timed_out'] and not exit['automatic_retry'] and exit['baseline_model_forwards']==exit['epochs_added']==0
def verify(seed,arm,archive,metadata,destination,report):
    assert seed in (1,2) and arm in ('p2_skip','backbone_silu','backbone_attention')
    spec=read(HERE/'protocol.json');assert sha(__file__)==spec['reader_sha256']
    for n,h in spec['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    user=ROOT/'英文小论文/code/user_backbone12_20261011_v1'
    assert {p.name:sha(p) for p in user.iterdir() if p.is_file()}==spec['original_user28_sources']
    primary=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed{seed}_{arm}_v1';P=read(primary/'primary_verification_v1.json')
    assert P['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS' and P['verification_script_sha256']==spec['dependencies_sha256']['英文小论文/code/user_backbone_primary_collection12_20261011_v2/verify_primary.py']
    assert P['all_collected_files_verified'] and P['all_ZIP_entries_CRC_verified'] and P['actual_stage_images']==548 and P['identity']['seed']==seed and P['identity']['arm']==arm
    packet=Path(P['downloaded_packet']);pm=read(packet/'primary_manifest.json');assert sha(packet/'primary_manifest.json')==P['primary_manifest_sha256']
    archive,metadata,destination,report=map(Path,(archive,metadata,destination,report));meta=read(metadata)
    assert meta['status']=='STEM_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED' and meta['seed']==seed and meta['baseline_forwards']==meta['extra_epochs']==0
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len({n.casefold() for n in names})==meta['archive_entries']
        for info in z.infolist():
            name=info.filename;q=PurePosixPath(name)
            assert not info.is_dir() and stat.S_ISREG(info.external_attr>>16) and not q.is_absolute() and '..' not in q.parts and '\\' not in name and ':' not in name and q.as_posix()==name
        raw=z.read('terminal_manifest.json');assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256'];manifest=json.loads(raw)
        identity=manifest['identity'];assert identity==P['identity'] and manifest['schema']=='user_backbone12_terminal_v1'
        assert manifest['training_epochs']==12 and manifest['baseline_model_forwards_added']==0 and not manifest['automatic_retry']
        assert set(names)==set(manifest['files'])|{'terminal_manifest.json'}
        assert {n[5:]:v['sha256'] for n,v in manifest['files'].items() if n.startswith('code/')}==spec['original_user28_sources']
        destination.mkdir(parents=True,exist_ok=False)
        for n in names:
            p=destination/n;assert p.resolve().is_relative_to(destination.resolve());p.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();length=0
            with z.open(n) as f,p.open('xb') as out:
                for b in iter(lambda:f.read(8*1024**2),b''):out.write(b);h.update(b);length+=len(b)
            if n=='terminal_manifest.json':assert h.hexdigest()==meta['manifest_sha256']
            else:assert length==manifest['files'][n]['bytes'] and h.hexdigest()==manifest['files'][n]['sha256'],n
    shared=[n for n in pm['files'] if n.startswith(('code/','bundle/','run/'))]
    assert {'run/last.pth','run/training_completion.json','run/endpoint_epoch12/manifest.json','run/stages_epoch12/manifest.json'}<=set(shared)
    for n in shared:
        assert manifest['files'][n]==pm['files'][n] and sha(destination/n)==sha(packet/n)==pm['files'][n]['sha256'],n
    run=destination/'run';done=read(run/'training_completion.json');replay=destination/'candidate_reload_v1';proof=read(replay/'verification.json');init=read(destination/'bundle/cpu_initialization.json')['initialization'];closeout=read(user/'closeout_protocol.json')
    assert_replay(proof,read(replay/'completion.json'),identity,done,init,closeout)
    assert proof['verification_script_sha256']==spec['original_user28_sources']['verify_candidate_reload.py'] and proof['closeout_protocol_sha256']==sha(user/'closeout_protocol.json')
    assert proof['original_training_completion_sha256']==sha(run/'training_completion.json') and sha(run/'last.pth')==proof['checkpoint_sha256']==P['last_checkpoint_sha256']
    tools=module('formal_user_saved_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py');p,_,_,_,readers=tools.prepared();base=tools.cached_baseline(seed,p,readers['full']);ann=base['annotation']
    for n in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):assert identity[n]==base['summary']['identity'][n],n
    B,bh,_=readers['control'].native_endpoint(destination/'baseline_eval/endpoint_epoch12',ann);C,ch,_=readers['control'].native_endpoint(run/'endpoint_epoch12',ann);R,rh,_=readers['control'].native_endpoint(replay/'endpoint_epoch12',ann)
    assert B==base['summary'] and C['identity']==R['identity']==identity and C['native_stats']==R['native_stats']==proof['original_stats']==P['native_stats']
    assert all(np.array_equal(ch[n],rh[n]) for n in ch)
    for name in ('precision','recall'):
        with np.load(run/'endpoint_epoch12/precision_recall.npz',allow_pickle=False) as c,np.load(replay/'endpoint_epoch12/precision_recall.npz',allow_pickle=False) as r:assert np.array_equal(c[name],r[name])
    assert sha(run/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256'] and sha(run/'stages_epoch12/manifest.json')==done['stage_manifest_sha256']
    for folder,ident in ((destination/'baseline_eval',B['identity']),(run,identity),(replay,identity)):readers['control'].stages(folder,ident,ann)
    for i in range(548):
        name=f'image_{i:06d}.npz'
        with np.load(run/'stages_epoch12'/name,allow_pickle=False) as c,np.load(replay/'stages_epoch12'/name,allow_pickle=False) as r:assert set(c.files)==set(r.files) and all(np.array_equal(c[n],r[n]) for n in c.files)
    results=module('formal_user_results',user/'results.py');pair=results.pair(B,C,bh,ch);saved=read(run/'fixed12_comparison.json')
    for n,val in pair.items():assert saved[n]==val,n
    delta=pair['delta_AP_metrics_pp'];gate=delta['APs']>0 and delta['AP']>=0;assert saved['native_gate_passed']==done['native_gate_passed']==gate
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    out=dict(status='PASS_USER_FORMAL_FIXED12_PRIMARY_AND_SAVED_EMA_EXACT',identity=identity,downloaded_packet=str(destination),archive_sha256=meta['sha256'],metadata_sha256=sha(metadata),terminal_manifest_sha256=meta['manifest_sha256'],
        primary_verification_sha256=sha(primary/'primary_verification_v1.json'),primary_static_files_exact=len(shared),files_verified=len(names),all_ZIP_entries_CRC_verified=True,all548_stage_arrays_reloaded_exact=True,native_precision_recall_exact=True,native_GT_hits_exact=True,
        independent_saved_EMA_replay_verified=True,formal_terminal_reader_passed=True,strict_loaded_tensors=init['state_tensors'],parameters=init['parameters'],candidate_overall_percent=C['overall_percent'],baseline_overall_percent=B['overall_percent'],delta_AP_metrics_pp=delta,native_gate_passed_this_seed=gate,
        native_source_files_reused_and_verified=468,evaluator_files_reused_and_verified=7,verification_script_sha256=sha(__file__),local_reader_protocol_sha256=sha(HERE/'protocol.json'),new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_training_epochs=0,new_optimizer_updates=0,
        two_seed_efficacy_proven=False,root_or_novelty_proven=False)
    write(report,out);print(json.dumps(dict(status=out['status'],seed=seed,arm=arm,delta=delta,gate=gate)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--arm',choices=('p2_skip','backbone_silu','backbone_attention'),required=True)
    for n in ('archive','metadata','destination','report'):p.add_argument('--'+n,required=True)
    a=p.parse_args();verify(a.seed,a.arm,a.archive,a.metadata,a.destination,a.report)
