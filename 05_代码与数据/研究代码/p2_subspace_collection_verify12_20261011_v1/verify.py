"""Offline artifact and native endpoint verification; no model or evaluator calls."""
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
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify(seed):
    protocol=read(HERE/'protocol.json');assert sha(__file__)==protocol['reader_sha256']
    for n,h in protocol['dependencies_sha256'].items():assert sha(ROOT/n)==h,n
    folder=ROOT/'outputs/p2_subspace_collection12_20261011_v1';archive=folder/f'downloads/runtime_seed{seed}_v1.zip';meta_path=folder/f'downloads/metadata_seed{seed}_v1.json';meta=read(meta_path)
    assert meta['status']=='COMPLETED_P2_RUNTIME_SUBSPACE_ARTIFACTS_COLLECTED_NO_NEW_FORWARD' and meta['seed']==seed
    assert meta['new_model_forwards']==meta['new_evaluator_calls']==meta['new_training_epochs']==0
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    out=folder/f'seed{seed}';out.mkdir(exist_ok=False);destination=out/'extracted'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len({n.casefold() for n in names})==meta['archive_entries']
        for i in z.infolist():
            n=i.filename;p=PurePosixPath(n)
            assert not i.is_dir() and stat.S_ISREG(i.external_attr>>16) and not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n and p.as_posix()==n
        assert z.testzip() is None
        raw=z.read('manifest.json');assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256'];manifest=json.loads(raw)
        assert manifest['schema']=='P2_runtime_subspace_complete_collection_v1' and manifest['seed']==seed
        assert set(names)==set(manifest['files'])|{'manifest.json'}
        assert manifest['collector_sha256']==meta['collector_sha256']==protocol['collector_sha256']
        assert manifest['new_model_forwards']==manifest['new_evaluator_calls']==manifest['new_training_epochs']==0
        assert manifest['diagnostic_source_sha256']==protocol['diagnostic_source_sha256']
        destination.mkdir(exist_ok=False)
        for n in names:
            p=destination/n;assert p.resolve().is_relative_to(destination.resolve());p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(n) as f,p.open('xb') as w:
                for b in iter(lambda:f.read(8*1024**2),b''):w.write(b)
            if n!='manifest.json':assert p.stat().st_size==manifest['files'][n]['bytes'] and sha(p)==manifest['files'][n]['sha256'],n
        assert {p.name:sha(p) for p in (destination/'code').iterdir() if p.is_file()}==protocol['diagnostic_source_sha256']
        assert sha(destination/'collection_source.py')==protocol['collector_sha256']
    parent_folder=ROOT/f'outputs/user_seed2_formal_collection12_20261011_v1/seed{seed}_p2_skip'
    formal=read(parent_folder/'verification.json');assert formal['status']=='PASS_USER_FORMAL_FIXED12_PRIMARY_AND_SAVED_EMA_EXACT' and formal['independent_saved_EMA_replay_verified'] and formal['formal_terminal_reader_passed']
    parent=Path(formal['downloaded_packet']);done=read(destination/'parent_training_completion.json');proof=read(destination/'parent_saved_EMA_verification.json')
    assert sha(destination/'parent_training_completion.json')==sha(parent/'run/training_completion.json')
    assert sha(destination/'parent_saved_EMA_verification.json')==sha(parent/'candidate_reload_v1/verification.json')
    assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12)) and proof['identity']==formal['identity']==done['identity']
    assert proof['checkpoint_sha256']==done['last_checkpoint_sha256']==manifest['parent_checkpoint_sha256']==meta['parent_checkpoint_sha256']==sha(parent/'run/last.pth')
    cpu=destination/f'seed{seed}_CPU';gpu=destination/f'seed{seed}_GPU';preparation=read(cpu/'report.json');cpu_done=read(cpu/'completion.json');report=read(gpu/'report.json');gpu_done=read(gpu/'completion.json');wait_done=read(destination/f'seed{seed}_wait_completion.json')
    assert preparation['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS' and cpu_done['exit_code']==gpu_done['exit_code']==wait_done['exit_code']==0
    assert cpu_done['report_sha256']==sha(cpu/'report.json') and cpu_done['weights_npz_sha256']==sha(cpu/'weights.npz')
    assert gpu_done['report_sha256']==sha(gpu/'report.json') and wait_done['completion_sha256']==sha(gpu/'completion.json')
    assert report['source_sha256']==protocol['diagnostic_source_sha256']['evaluate.py'] and report['protocol_sha256']==protocol['diagnostic_source_sha256']['protocol.json']
    assert report['status']=='TWO_FIXED12_P2_RUNTIME_SUBSPACE_INTERVENTIONS_NATIVE_EVALUATED' and report['seed']==seed
    assert report['parent_checkpoint_sha256']==done['last_checkpoint_sha256'] and report['parent_replay_verification_sha256']==sha(destination/'parent_saved_EMA_verification.json') and report['CPU_preparation_report_sha256']==sha(cpu/'report.json')
    assert report['candidate_model_forwards']==274 and report['new_native_evaluator_runs']==2
    assert report['new_BASE_forwards']==report['new_training_epochs']==report['new_optimizer_updates']==0
    assert report['checkpoint_unchanged'] and report['original28_training_sources_unchanged']
    assert report['root_proven'] is report['novelty_proven'] is report['training_efficacy_proven'] is False
    assert preparation['identity']==formal['identity'] and preparation['checkpoint_sha256']==done['last_checkpoint_sha256']
    assert preparation['source_sha256']==protocol['diagnostic_source_sha256']['prepare.py'] and preparation['protocol_sha256']==report['protocol_sha256']
    tools=module('p2_runtime_artifact_prepared_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py');p,_,_,_,readers=tools.prepared();base=tools.cached_baseline(seed,p,readers['full']);ann=base['annotation']
    parent_summary,_,_=readers['control'].native_endpoint(parent/'run/endpoint_epoch12',ann)
    assert parent_summary['identity']==formal['identity'] and parent_summary['native_stats']==proof['original_stats']
    spec=read(destination/'code/protocol.json');assert [r['mode'] for r in report['results']]==spec['interventions']
    with np.load(cpu/'weights.npz',allow_pickle=False) as z:weights={n:z[n] for n in z.files}
    assert set(weights)=={'original',*spec['interventions']}
    W=weights['original'];assert W.shape==(256,64,1,1) and W.dtype==np.float32 and hashlib.sha256(W.tobytes()).hexdigest()==preparation['parent_restore_weight_SHA256']
    assert np.linalg.matrix_rank(W[:,:,0,0].astype(np.float64))==64
    norm=np.linalg.norm(W.astype(np.float64));records=[]
    for result,case in zip(report['results'],preparation['cases']):
        mode=result['mode'];assert mode==case['mode'];w=weights[mode]
        assert w.shape==W.shape and w.dtype==W.dtype and hashlib.sha256(w.tobytes()).hexdigest()==case['weight_SHA256']
        rank=int(np.linalg.matrix_rank(w[:,:,0,0],tol=1e-6));err=float(abs(np.linalg.norm(w.astype(np.float64))/norm-1));assert rank==case['numerical_rank']==16 and err<1e-6
        if mode=='dc16_norm':expected=W.reshape(256,16,4).astype(np.float64).mean(-1,keepdims=True).repeat(4,-1).reshape(W.shape)
        else:
            u,s,vh=np.linalg.svd(W[:,:,0,0].astype(np.float64),full_matrices=False);expected=((u[:,:16]*s[:16])@vh[:16])[:,:,None,None]
        expected=expected*(norm/np.linalg.norm(expected));assert np.max(np.abs(expected-w))<1e-6
        identity=result['identity'];ri=identity['runtime_intervention'];training_identity={n:v for n,v in identity.items() if n!='runtime_intervention'};training_identity['schema']=formal['identity']['schema'];assert training_identity==formal['identity']
        assert ri['mode']==mode and ri['rank']==16 and ri['normalization']=='parent_restore_Frobenius_norm' and ri['parent_checkpoint_sha256']==done['last_checkpoint_sha256']
        assert ri['projected_restore_weight_sha256']==case['weight_SHA256'] and ri['CPU_preparation_report_sha256']==sha(cpu/'report.json')
        assert ri['parent_training_identity_sha256']==hashlib.sha256(json.dumps(formal['identity'],sort_keys=True).encode()).hexdigest()
        folder=gpu/mode;assert sha(folder/'endpoint_epoch12/manifest.json')==result['endpoint_manifest_sha256'] and sha(folder/'stages_epoch12/manifest.json')==result['stage_manifest_sha256']
        summary,flags,checks=readers['control'].native_endpoint(folder/'endpoint_epoch12',ann);readers['control'].stages(folder,identity,ann)
        assert summary['identity']==identity and summary['native_stats']==result['native_stats'] and result['parent_native_stats']==parent_summary['native_stats']
        assert result['candidate_model_forwards']==137 and result['changed_state_tensors']==['backbone.p2_skip.restore.weight'] and result['parameters']==10249437
        delta={n:summary['overall_percent'][n]-parent_summary['overall_percent'][n] for n in ('AP','AP50','AP75','APs','APm','APl')}
        assert abs(delta['AP']-result['delta_AP_points'])<1e-12 and abs(delta['APs']-result['delta_APs_points'])<1e-12
        class_delta=[dict(class_id=c['class_id'],class_name=c.get('class_name',c.get('name')),AP=c['AP']-b['AP'],APs=c['APs']-b['APs']) for c,b in zip(summary['categories_percent'],parent_summary['categories_percent'])]
        assert abs(np.mean([c['AP'] for c in class_delta])-delta['AP'])<1e-9 and abs(np.mean([c['APs'] for c in class_delta])-delta['APs'])<1e-9
        records.append(dict(mode=mode,numerical_rank=rank,relative_weight_norm_error=err,overall_percent=summary['overall_percent'],delta_vs_parent_points=delta,delta_vs_BASE_points={n:summary['overall_percent'][n]-base['summary']['overall_percent'][n] for n in delta},class_delta_vs_parent=class_delta,endpoint_reader_checks=checks,stage_images_verified=548))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    result=dict(status='PASS_COMPLETE_P2_RUNTIME_ARTIFACTS_AND_TWO_NATIVE_ENDPOINTS',seed=seed,archive_sha256=meta['sha256'],metadata_sha256=sha(meta_path),manifest_sha256=meta['manifest_sha256'],all_entries_and_CRC_verified=True,files_verified=len(names),parent_checkpoint_sha256=done['last_checkpoint_sha256'],parent_formal_verification_sha256=sha(parent_folder/'verification.json'),reader_sha256=sha(__file__),reader_protocol_sha256=sha(HERE/'protocol.json'),actual_remote_candidate_forwards=274,actual_remote_native_evaluator_runs=2,new_local_model_forwards=0,new_local_evaluator_calls=0,new_training_epochs=0,new_BASE_forwards=0,results=records,training_efficacy_proven=False,two_seed_efficacy_proven=False,unique_causal_root_proven=False,novelty_proven=False,scope=report['scope'])
    with (out/'verification.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],seed=seed,results=[dict(mode=r['mode'],AP=r['overall_percent']['AP'],APs=r['overall_percent']['APs'],delta=r['delta_vs_parent_points']) for r in records])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);verify(p.parse_args().seed)
