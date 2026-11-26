"""Verify actual component primary archives, explicitly without a saved-EMA replay claim."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,importlib.util,json,stat,sys,zipfile
from guards import ARMS,identity_plan,positive_terminal,preflight_launch
ROOT=Path(__file__).resolve().parents[3]
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,allow_nan=False)
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(seed, arm):
    assert arm in ARMS
    folder=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed{seed}_{arm}_v1'
    meta=read(folder/'remote_collection_metadata_v1.json')
    executed=read(folder/'remote_collection_execution_v1.json');transport=read(folder/'remote_collection_transport_v1.json')
    assert sha(folder/'remote_collection_execution_v1.json')==transport['sha256']
    assert (folder/'remote_collection_execution_v1.json').stat().st_size==transport['bytes']
    assert executed['exit_code']==0 and executed['metadata']==meta
    assert executed['status']=='STOPPED_USER_PRIMARY12_PACKAGE_COLLECTED'
    assert executed['receipt']['seed']==seed and executed['receipt']['arm']==arm and not executed['receipt']['automatic_retry']
    assert executed['receipt']['source_sha256']==sha(Path(__file__).parent/'collect.py')
    assert executed['receipt']['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    collector=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2'
    protocol=read(collector/'protocol.json')
    assert sha(__file__)==protocol['local_verifier_sha256']
    assert sha(collector/'guards.py')==protocol['remote_sources']['guards.py']
    assert meta['collector_sha256']==sha(collector/'collect.py') and meta['collector_protocol_sha256']==sha(collector/'protocol.json')
    assert meta['seed']==seed and meta['arm']==arm and not meta['saved_EMA_independent_replay_included']
    assert meta['new_model_forwards']==meta['new_evaluator_calls']==meta['new_training_epochs']==0
    archive=folder/'primary_artifacts_v1.zip'
    assert archive.stat().st_size==meta['bytes'] and sha(archive)==meta['sha256']
    expected=read(collector/'protocol.json')['user_files_sha256']
    destination=folder/'extracted'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len({n.casefold() for n in names})==meta['archive_entries']
        for info in z.infolist():
            n=info.filename;p=PurePosixPath(n)
            assert not info.is_dir() and stat.S_ISREG(info.external_attr>>16)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n and p.as_posix()==n
        raw=z.read('primary_manifest.json');assert hashlib.sha256(raw).hexdigest()==meta['manifest_sha256']
        manifest=json.loads(raw);identity=manifest['identity']
        assert manifest['schema']=='exact12_user_backbone_primary_artifact_collection_v1'
        assert set(names)==set(manifest['files'])|{'primary_manifest.json'}
        assert identity['seed']==seed and identity['arm']==arm and identity['implementation']==expected
        assert {n[5:]:v['sha256'] for n,v in manifest['files'].items() if n.startswith('code/')}==expected
        assert manifest['completed_epochs']==12 and manifest['evaluated_weights']=='EMA' and manifest['epoch_zero_based']==11
        assert not manifest['saved_EMA_independent_replay_included'] and manifest['stage_images']==548
        assert manifest['native_stats']==meta['native_stats'] and manifest['collector_sha256']==meta['collector_sha256']
        if destination.exists():
            assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}==set(names)
            assert z.testzip() is None
            assert sha(destination/'primary_manifest.json')==meta['manifest_sha256']
            for n,item in manifest['files'].items():
                assert (destination/n).stat().st_size==item['bytes'] and sha(destination/n)==item['sha256'],n
        else:
            destination.mkdir(exist_ok=False)
            for n in names:
                target=destination/n;assert target.resolve().is_relative_to(destination.resolve());target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256();length=0
                with z.open(n) as r,target.open('xb') as w:
                    for b in iter(lambda:r.read(8*1024**2),b''):w.write(b);h.update(b);length+=len(b)
                if n=='primary_manifest.json':assert h.hexdigest()==meta['manifest_sha256']
                else:assert length==manifest['files'][n]['bytes'] and h.hexdigest()==manifest['files'][n]['sha256'],n
    run=destination/'run';done=read(run/'training_completion.json')
    assert sha(run/'training_completion.json')==manifest['source_training_completion_sha256']
    assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
    assert done['identity']==identity and done['last_checkpoint_sha256']==sha(run/'last.pth')==meta['last_checkpoint_sha256']
    assert done['stage_manifest_sha256']==sha(run/'stages_epoch12/manifest.json')
    assert done['endpoint_manifest_sha256']==sha(run/'endpoint_epoch12/manifest.json')
    assert read(destination/'screen/candidate_completion.json')['exit_code']==0 and not read(destination/'screen/candidate_completion.json')['timed_out']
    assert read(destination/'screen/controller_completion.json')['status']=='COMPLETE_FIXED12'
    assert read(run/'first100_parity.json')['all100_rows_exact']
    receipt=read(destination/'bundle/receipt.json');plan=read(destination/'bundle/plan.json')
    assert receipt['identity']==identity and plan['schema']=='user_backbone12_training_v1' and plan['arm']==arm and plan['seed']==seed
    assert identity['protocol_sha256']==sha(destination/'code/training_protocol.json')
    assert identity['plan_sha256']==sha(destination/'bundle/plan.json') and receipt['resolved_sha256']==sha(destination/'bundle/resolved.yml')
    assert sha(destination/'bundle/cpu_initialization.json')==identity['cpu_initialization_sha256']
    initialization=read(destination/'bundle/cpu_initialization.json')['initialization']
    identity_plan(identity,plan,initialization,arm,seed)
    positive_terminal(identity,done,read(destination/'screen/controller_completion.json'),read(destination/'screen/candidate_completion.json'))
    assert identity['common_init_sha256']=='85497586487a507923404da886d52c1054999111ce6be1df92532b4998956f91'
    runtime=module('warm_primary_native_config',ROOT/'英文小论文/code/user_backbone12_20261011_v1/runtime.py')
    cfg=runtime.resolved_config(plan,{k:PurePosixPath(v) for k,v in receipt['paths'].items()})
    assert read(destination/'bundle/resolved.yml')==cfg and read(run/'args.json')['yaml_cfg']==cfg
    start=read(run/'runtime_start.json')
    assert start['identity']==identity and start['optimizer_coverage']['all_parameters_exactly_once']
    assert start['optimizer_coverage']['all_native_parameter_groups_unchanged']
    assert start['optimizer_coverage']['added_parameters']==identity['added_parameter_count']
    assert start['initialization']==initialization
    assert start['native_augmentation_and_supervision_weights_unchanged']
    assert start['evaluator_runtime']['binding']==identity['evaluator_binding']
    prefix={1:'/25085400120',2:'/24085403014'}[seed]
    preflight_launch(read(destination/'screen/detector_preflight_dispatch.json'),read(destination/'screen/detector_preflight_completion.json'),prefix,seed,PurePosixPath(receipt['paths']['screen_root'])/'bundle')
    full=read(destination/'screen/detector_preflight_v1/report.json')
    assert full['identity']==identity and full['status']=='PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
    assert full['optimizer_updates']==full['training_epochs']==0 and full['candidate_detector_forwards']==2
    assert full['initialization']==initialization and full['optimizer_coverage']['all_parameters_exactly_once']
    assert [case['side'] for case in full['cases']]==[640,832]
    assert all(case['finite_gradients'] for case in full['cases'])
    if arm=='p2_skip':assert all(all(case['branch'][k] for k in ('zero_restore_maintained','restore_gradient_active','upstream_bridge_gradients_initially_zero')) for case in full['cases'])
    for key in ('image_manifest','image_verification'):
        assert sha(destination/('provenance/'+key+'.json'))==identity[key+'_sha256']
    assert read(destination/'provenance/image_verification.json')['files_verified']==7019
    tools=module('component_primary_prepared_tools',ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py')
    p,geometry,onset,rank,readers=tools.prepared();base=tools.cached_baseline(seed,p,readers['full'])
    for n in ('common_init_sha256','baseline_checkpoint_sha256','baseline_first100_sha256','val_annotation_sha256','evaluator_binding'):
        assert identity[n]==base['summary']['identity'][n]
    summary,flags,_=readers['control'].native_endpoint(run/'endpoint_epoch12',base['annotation'])
    assert summary['identity']==identity and summary['native_stats']==meta['native_stats']
    readers['control'].stages(run,identity,base['annotation'])
    rows=[json.loads(x) for x in (run/'log.txt').read_text().splitlines() if x]
    assert [r['epoch'] for r in rows]==list(range(12)) and rows[-1]['test_coco_eval_bbox']==summary['native_stats']
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    result=dict(status='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS',identity=identity,
        downloaded_packet=str(destination),archive_sha256=meta['sha256'],primary_manifest_sha256=meta['manifest_sha256'],
        last_checkpoint_sha256=meta['last_checkpoint_sha256'],native_stats=meta['native_stats'],files_verified=len(manifest['files']),
        actual_stage_images=548,all_collected_files_verified=True,all_ZIP_entries_CRC_verified=True,
        saved_EMA_independent_replay_verified=False,formal_terminal_reader_passed=False,
        verification_script_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
    write(folder/'primary_verification_v1.json',result)
    print(json.dumps(dict(seed=seed,status=result['status'],files=result['files_verified'],AP=100*meta['native_stats'][0],APs=100*meta['native_stats'][3])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();verify(a.seed,a.arm)
