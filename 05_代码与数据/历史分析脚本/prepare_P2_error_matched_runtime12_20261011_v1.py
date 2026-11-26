"""Prepare one native altered-weight evaluation per seed, binding real CPU checks."""
from pathlib import Path
import ast,copy,hashlib,json,zipfile
R=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=R/'英文小论文/code/p2_error_matched_runtime12_20261011_v1';out=R/'outputs/p2_error_matched_runtime12_20261011_v1'
assert not root.exists() and not out.exists()
data={}
for k,v in read(R/'outputs/turn8_error_runtime_actual_bound_inputs.json').items():
    b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256'];data[k]=json.loads(b)
old=read(R/'英文小论文/code/p2_subspace_diagnostic12_20261011_v1/protocol.json')
bindings={};owners={}
for seed in (1,2):
    j=data[f'turn8_error_runtime_bound_inputs_seed{seed}'];f=j['files'];live=data[f'turn8_rank16_initial_actual_seed{seed}']
    assert f['CPU_report']['report']['status']=='PASS_ACTUAL_CPU_RANK16_NORM_DC_ERROR_AND64_BASIS_OPERATOR_CHECK'
    assert f['CPU_completion']['report']['exit_code']==0
    assert f['replay']['report']['status']=={1:'CANDIDATE_RELOAD_MISMATCH_NO_RETRY',2:'PASS_SAVED_CANDIDATE_RELOAD'}[seed]
    bindings[str(seed)]=dict(files={k:v['sha256'] for k,v in f.items()},CPU_source_files=j['CPU_source_files'],
        weights_sha256=f['CPU_report']['report']['weights_npz_sha256'],checkpoint_sha256=f['proposal']['report']['parent_checkpoint_sha256'],reference_replay_status=f['replay']['report']['status'])
    owners[str(seed)]=[live['queue_handle'],live['controller_handle'],live['candidate_handle']]
    if seed==1:owners[str(seed)]+=[live['remaining_formal_queue']['handle'],j['direction_queue_handle']]
contract='''"""Pure metadata contract; original replay status is preserved for each seed."""
def validate(seed,proposal,checked,completed,replay,original,protocol):
    assert type(seed) is int and seed in (1,2)
    binding=protocol['bindings'][str(seed)]
    assert checked['seed']==proposal['seed']==replay['seed']==original['seed']==seed
    assert checked['status']=='PASS_ACTUAL_CPU_RANK16_NORM_DC_ERROR_AND64_BASIS_OPERATOR_CHECK'
    assert completed['exit_code']==0 and completed['automatic_retry'] is False
    assert checked['rank']==16 and checked['basis64_exact'] is True and checked['CUDA_initialized'] is False
    assert checked['GPU_evaluation_completed'] is False and checked['root_or_efficacy_proven'] is False
    assert checked['norm_error']<1e-6 and abs(checked['error_difference'])<1e-6
    assert all(checked[k]==0 for k in ('full_detector_forwards','BASE_forwards','parent_forwards','evaluator_calls','optimizer_updates','training_epochs'))
    assert proposal['mode']=='svd16_dc_error_norm' and proposal['rank']==16
    assert proposal['model_replay_tolerance_changed'] is False
    assert replay['status']==binding['reference_replay_status']
    assert replay['strict_loaded_tensors']==801 and replay['parameters']==10249437
    assert replay['identity']==original['identity'] and replay['arm']=='p2_skip'
    assert replay['checkpoint_sha256']==original['checkpoint_sha256']==proposal['parent_checkpoint_sha256']==binding['checkpoint_sha256']
    assert replay['candidate_model_forwards']==137 and replay['images']==548
    assert replay['baseline_model_forwards']==replay['optimizer_updates']==replay['epochs_added']==0
    if seed==1:
        assert replay['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
        assert not replay['native_GT_hits_exact'] and not replay['all548_stage_arrays_exact'] and replay['stats_max_abs_fraction_difference']>0
    else:
        assert replay['status']=='PASS_SAVED_CANDIDATE_RELOAD'
        assert replay['native_GT_hits_exact'] and replay['all548_stage_arrays_exact'] and replay['stats_max_abs_fraction_difference']==0
    assert protocol['mode']=='svd16_dc_error_norm' and protocol['model_forwards_per_seed']==137
    assert protocol['parent_forwards_added']==protocol['BASE_forwards_added']==protocol['training_epochs_added']==0
'''
evaluate='''"""One frozen native P2 EMA with one altered restore matrix; no training."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,sys,subprocess,traceback,zipfile
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main(seed):
    assert seed in (1,2)
    spec=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in spec['remote_sources']}==spec['remote_sources']
    prefix=Path('/25085400120' if seed==1 else '/24085403014');base=prefix/'DEIM/experiments'
    user=base/'user_backbone12_20261011_v1';code=user/'code';screen=user/('seed%d_p2_skip'%seed);run=screen/'run'
    assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==spec['user28_sources']
    cpu=base/'p2_coefficient_error_CPU12_20261011_v2';original_cpu=base/'p2_subspace_diagnostic12_20261011_v1'/('seed%d_CPU'%seed)
    files=dict(CPU_report=cpu/('seed%d_CPU_checked/report.json'%seed),CPU_completion=cpu/('seed%d_CPU_checked/completion.json'%seed),proposal=cpu/('seed%d_input/report.json'%seed),replay=screen/'candidate_reload_v1/verification.json',reference_manifest=screen/'candidate_reload_v1/endpoint_epoch12/manifest.json',original_CPU=original_cpu/'report.json')
    binding=spec['bindings'][str(seed)]
    for n,p in files.items():assert sha(p)==binding['files'][n],n
    assert {p.name:sha(p) for p in (cpu/'code').iterdir() if p.is_file()}==binding['CPU_source_files']
    proposal,checked,completed,replay,original=[read(files[n]) for n in ('proposal','CPU_report','CPU_completion','replay','original_CPU')]
    from contract import validate
    validate(seed,proposal,checked,completed,replay,original,spec)
    weights=cpu/('seed%d_input/weights.npz'%seed);assert sha(weights)==binding['weights_sha256']
    assert sha(run/'last.pth')==binding['checkpoint_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==spec['GPU_uuid'][str(seed)]
    sys.path.insert(0,str(code))
    from runtime import verify_bundle,activate,register_model,resolved_config
    from results import load_endpoint,capture
    from stage_trace import StageTrace,TracedLoader,verify_trace
    from evaluator_binding import activate as bind_evaluator
    receipt,plan,paths=verify_bundle(screen/'bundle')
    assert receipt['identity']==original['identity'] and plan['arm']=='p2_skip' and plan['max_epochs']==12
    parent,_,_=load_endpoint(run/'endpoint_epoch12');reloaded,_,_=load_endpoint(screen/'candidate_reload_v1/endpoint_epoch12')
    assert parent['identity']==reloaded['identity']==receipt['identity']
    assert parent['native_stats']==replay['original_stats'] and reloaded['native_stats']==replay['reloaded_stats']
    activate(paths['native_root']);bind_evaluator(paths['evaluator_root'],receipt['identity']['evaluator_binding']);register_model()
    import numpy as np
    import torch
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver.det_engine import evaluate
    with np.load(weights,allow_pickle=False) as z:W=z['original'].copy();P=z[spec['mode']].copy()
    assert W.shape==P.shape==(256,64,1,1) and W.dtype==P.dtype==np.float32 and np.isfinite(P).all()
    assert hashlib.sha256(W.tobytes()).hexdigest()==proposal['original_weight_SHA256']
    assert hashlib.sha256(P.tobytes()).hexdigest()==proposal['projected_weight_SHA256']
    output=HERE.parent/('seed%d_GPU'%seed);output.mkdir(exist_ok=False)
    started=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],seed=seed,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0)
    write(output/'start.json',started)
    try:
        dist_utils.setup_distributed(print_rank=0,print_method='builtin',seed=seed)
        config=resolved_config(plan,dict(paths,output=output));write(output/'resolved.yml',config)
        cfg=YAMLConfig(str(output/'resolved.yml'));model=cfg.model.cuda()
        state=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
        assert state['last_epoch']==11 and state['user_backbone12_identity']==receipt['identity']
        expected=state['ema']['module'];model.load_state_dict(expected,strict=True)
        actual=model.state_dict();assert set(actual)==set(expected) and len(actual)==801
        assert all(torch.equal(v.detach().cpu(),expected[n].cpu()) for n,v in actual.items())
        assert sum(p.numel() for p in model.parameters())==10249437
        assert np.array_equal(model.backbone.p2_skip.restore.weight.detach().cpu().numpy(),W)
        with torch.no_grad():model.backbone.p2_skip.restore.weight.copy_(torch.from_numpy(P).to(model.backbone.p2_skip.restore.weight))
        changed=[n for n,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),expected[n].cpu())]
        assert changed==['backbone.p2_skip.restore.weight']
        identity=copy.deepcopy(receipt['identity']);identity.update(schema='P2_coefficient_error_matched_runtime_v1',runtime_intervention=dict(mode=spec['mode'],rank=16,parent_checkpoint_sha256=binding['checkpoint_sha256'],projected_restore_weight_sha256=proposal['projected_weight_SHA256'],CPU_preparation_report_sha256=sha(files['CPU_report']),normalization='parent_weight_norm_and_DC_relative_coefficient_error'))
        del state,expected,actual
        model.requires_grad_(False);model.eval();criterion=cfg.criterion.cuda();criterion.set_model(model);criterion.set_epoch(11)
        post,loader,evaluator=cfg.postprocessor.cuda(),cfg.val_dataloader,cfg.evaluator
        calls=0
        def observe(m,args,y):
            nonlocal calls
            assert not m.training and not torch.is_grad_enabled()
            assert y['pred_boxes'].shape[1:]==(300,4) and y['pred_logits'].shape[1:]==(300,10)
            assert torch.isfinite(y['pred_boxes']).all() and torch.isfinite(y['pred_logits']).all();calls+=1
        handle=model.register_forward_hook(observe);trace=StageTrace(model,identity,paths['val_ann'])
        try:
            with torch.no_grad():stats,obj=evaluate(model,criterion,post,TracedLoader(loader,trace),evaluator,torch.device('cuda:0'),test_only=False,output_dir=output,yolo_metrice=config['yolo_metrice'])
            assert obj is evaluator and calls==len(loader)==137
            trace.finish(output/'stages_epoch12');capture(evaluator.coco_eval['bbox'],stats,identity,paths['val_ann'],output/'endpoint_epoch12',epoch=11,environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0)))
            summary,_,_=load_endpoint(output/'endpoint_epoch12');verify_trace(output/'stages_epoch12',identity)
        finally:handle.remove();trace.close();dist_utils.cleanup()
        assert sha(run/'last.pth')==binding['checkpoint_sha256'] and {p.name:sha(p) for p in code.iterdir() if p.is_file()}==spec['user28_sources']
        saved={}
        refroot=base/('p2_direction_diagnostic12_20261011_v2/seed1_GPU' if seed==1 else 'p2_subspace_diagnostic12_20261011_v1/seed2_GPU')
        for mode in ('dc16_norm','svd16_norm'):
            path=refroot/mode/'endpoint_epoch12'
            if not (path/'manifest.json').is_file():saved[mode]=dict(status='SAVED_REFERENCE_NOT_AVAILABLE_NO_RETRY');continue
            try:
                old_summary,_,_=load_endpoint(path);intervention=old_summary['identity']['runtime_intervention']
                case=next(x for x in original['cases'] if x['mode']==mode)
                assert intervention['mode']==mode and intervention['parent_checkpoint_sha256']==binding['checkpoint_sha256']
                assert intervention['projected_restore_weight_sha256']==case['weight_SHA256']
                saved[mode]=dict(status='SAVED_NATIVE_REFERENCE_BOUND',summary_manifest_sha256=sha(path/'manifest.json'),overall_percent=old_summary['overall_percent'],delta_points={k:summary['overall_percent'][k]-old_summary['overall_percent'][k] for k in summary['overall_percent']})
            except Exception as e:saved[mode]=dict(status='INVALID_SAVED_REFERENCE_NOT_USED',error=repr(e))
        report=dict(status='ONE_DC_ERROR_MATCHED_RANK16_WEIGHT_NATIVE_EVALUATED',seed=seed,identity=identity,mode=spec['mode'],overall_percent=summary['overall_percent'],categories_percent=summary['categories_percent'],native_stats=summary['native_stats'],saved_control_comparisons=saved,
            descriptive_original_parent_delta_points={k:summary['overall_percent'][k]-parent['overall_percent'][k] for k in summary['overall_percent']},descriptive_reloaded_parent_delta_points={k:summary['overall_percent'][k]-reloaded['overall_percent'][k] for k in summary['overall_percent']},parent_replay_status=replay['status'],parent_replay_status_unchanged=True,
            changed_state_tensors=changed,unchanged_state_tensors=800,parameters=10249437,candidate_model_forwards=calls,new_native_evaluator_runs=1,new_BASE_forwards=0,new_parent_forwards=0,new_training_epochs=0,new_optimizer_updates=0,
            checkpoint_unchanged=True,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),CPU_report_sha256=sha(files['CPU_report']),root_proven=False,novelty_proven=False,training_efficacy_proven=False,
            scope='Rank16, parent weight norm and DC unweighted coefficient approximation error constrained. Actual feature covariance, output energy, spectral norm and task sensitivity remain unmatched. Compare saved DC/SVD controls conditionally. No parent replay, BASE inference, training or unique-root claim.')
        write(output/'report.json',report);write(output/'completion.json',dict(exit_code=0,report_sha256=sha(output/'report.json'),automatic_retry=False))
        pairs={}
        for folder,label in ((HERE,'code'),(output,'evaluation')):
            for p in folder.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:pairs[label+'/'+p.relative_to(folder).as_posix()]=p
        for n,p in files.items():pairs['provenance/'+n+'.json']=p
        pairs['provenance/weights.npz']=weights
        manifest=dict(identity=identity,files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in pairs.items()},new_BASE_forwards=0,new_parent_forwards=0,new_training_epochs=0)
        write(HERE.parent/('seed%d_manifest.json'%seed),manifest)
        archive=HERE.parent/('seed%d_artifacts_v1.zip'%seed)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for n,p in pairs.items():z.write(p,n)
            z.write(HERE.parent/('seed%d_manifest.json'%seed),'manifest.json')
        write(HERE.parent/('seed%d_archive_metadata.json'%seed),dict(bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(HERE.parent/('seed%d_manifest.json'%seed)),archive_entries=len(pairs)+1,seed=seed,model_forwards=calls,BASE_forwards=0,parent_forwards=0,training_epochs=0))
        print(json.dumps(dict(status=report['status'],seed=seed,overall=report['overall_percent'],comparisons=saved)),flush=True)
    except BaseException:
        write(output/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,new_BASE_forwards=0,new_parent_forwards=0,new_training_epochs=0));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=(1,2),required=True);main(a.parse_args().seed)
'''
wait='''"""Single altered-weight diagnostic after current exact owners and actual GPU idle."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time,subprocess,traceback,signal
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def live(h):
    pid=h.get('child_pid',h.get('pid'));ticks=h.get('child_start_ticks',h.get('start_ticks'))
    try:
        s=Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split();return s[19]==str(ticks) and s[0]!='Z'
    except FileNotFoundError:return False
def main(seed):
    spec=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in spec['remote_sources']}==spec['remote_sources']
    assert seed in (1,2) and not (ROOT/('seed%d_GPU'%seed)).exists()
    queue=ROOT/('queue_seed%d'%seed);queue.mkdir(exist_ok=False)
    start=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],seed=seed,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0)
    write(queue/'start.json',start);begin=time.monotonic()
    try:
        base=Path('/25085400120' if seed==1 else '/24085403014')/'DEIM/experiments'
        while True:
            owners=list(spec['predecessor_handles'][str(seed)])
            dynamic=[base/'p2_lowrank12_20261011_v3'/('seed%d_p2_lowrank16/finisher_dispatch.json'%seed)]
            if seed==1:dynamic += [base/'user_remaining_formal_closeout12_20261011_v1/backbone_silu_dispatch.json',base/'user_remaining_formal_closeout12_20261011_v1/backbone_attention_dispatch.json',base/'p2_direction_diagnostic12_20261011_v2/GPU_dispatch.json']
            for p in dynamic:
                if p.is_file():owners.append(read(p))
            observed=[dict(h,same_live=live(h)) for h in owners]
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not any(x['same_live'] for x in observed) and not apps:break
            temp=queue/'live.tmp';temp.write_text(json.dumps(dict(phase='waiting_exact_existing_owners_and_actual_GPU_idle',owners=observed,GPU_apps=apps,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())));temp.replace(queue/'live.json')
            assert time.monotonic()-begin<spec['max_wait_seconds'],'Only this CPU waiter exits; no existing-process signals or restart'
            time.sleep(30)
        cmd=[sys.executable,'-B',str(HERE/'evaluate.py'),'--seed',str(seed)]
        with (queue/'GPU.log').open('xb') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        handle=dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19],command=cmd,automatic_retry=False)
        write(queue/'GPU_dispatch.json',handle);launched=time.monotonic();timed_out=False
        while child.poll() is None:
            if time.monotonic()-launched>=spec['private_GPU_timeout_seconds']:
                timed_out=True
                if live(handle):os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if live(handle):os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                break
            time.sleep(5)
        write(queue/'GPU_completion.json',dict(exit_code=child.returncode,timed_out=timed_out,automatic_retry=False,existing_process_signals=0))
        assert child.returncode==0 and not timed_out,'Preserve all outputs; no retry'
        assert read(ROOT/('seed%d_GPU/completion.json'%seed))['exit_code']==0
        write(queue/'completion.json',dict(status='ONE_ERROR_MATCHED_NATIVE_DIAGNOSTIC_COMPLETED_ONCE',training_epochs=0,BASE_forwards=0,parent_forwards=0,automatic_retry=False,existing_process_signals=0))
    except BaseException:
        write(queue/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0,existing_process_signals=0));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=(1,2),required=True);main(a.parse_args().seed)
'''
root.mkdir();out.mkdir()
for n,s in [('contract.py',contract),('evaluate.py',evaluate),('wait.py',wait)]:ast.parse(s);(root/n).write_text(s,encoding='utf8')
spec=dict(schema='P2_error_matched_single_runtime_diagnostic_v1',mode='svd16_dc_error_norm',remote_sources={p.name:sha(p) for p in root.iterdir()},
    user28_sources=old['user28_sources'],bindings=bindings,predecessor_handles=owners,GPU_uuid=old['GPU_uuid'],
    model_forwards_per_seed=137,parent_forwards_added=0,BASE_forwards_added=0,training_epochs_added=0,
    max_wait_seconds=172800,private_GPU_timeout_seconds=1800,
    primary_question='Does a rank16 norm-matched control with DC-matched unweighted coefficient approximation error behave differently from saved DC16?',
    scope='One new altered-weight evaluation per seed on original trained P2. No parent replay, new BASE forward, training, best-epoch selection or efficacy/root claim. A failed original replay remains failed.')
(root/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
space={};exec(contract,space);validate=space['validate'];negatives=[]
for seed in (1,2):
    f=data[f'turn8_error_runtime_bound_inputs_seed{seed}']['files']
    args=[f[n]['report'] for n in ('proposal','CPU_report','CPU_completion','replay','original_CPU')]
    validate(seed,*args,spec)
    for name,index,key,value in [('wrong_mode',0,'mode','svd16_norm'),('CPU_failed',2,'exit_code',1),('norm_bad',1,'norm_error',.1),('error_bad',1,'error_difference',.1),('BASE_forward',1,'BASE_forwards',1),('wrong_checkpoint',3,'checkpoint_sha256','0'*64),('relabel_replay',3,'status','PASS_SAVED_CANDIDATE_RELOAD' if seed==1 else 'CANDIDATE_RELOAD_MISMATCH_NO_RETRY')]:
        a=copy.deepcopy(args);a[index][key]=value
        try:validate(seed,*a,spec)
        except (AssertionError,KeyError):negatives.append(dict(seed=seed,name=name))
        else:raise AssertionError(name)
assert "changed==['backbone.p2_skip.restore.weight']" in evaluate and 'calls==len(loader)==137' in evaluate
review=dict(status='REVIEWED_TWO_SEED_ONCE_ONLY_ERROR_MATCHED_NATIVE_RUNTIME_WITH_REAL_CPU_BINDINGS',files={p.name:sha(p) for p in root.iterdir()},metadata_negative_cases=negatives,
    changed_model_states=1,unchanged_model_states=800,native_forwards_per_seed=137,new_parent_forwards=0,new_BASE_forwards=0,new_training_epochs=0,
    A_original_replay_failure_preserved=True,existing_process_signals=0,private_GPU_timeout_seconds=1800,
    efficacy_or_unique_root_proven=False,producer_sha256=sha(__file__))
(out/'source_review.json').write_text(json.dumps(review,indent=2),encoding='utf8')
archive=out/'deployment_v1.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():z.write(p,'code/'+p.name)
print(json.dumps(dict(status=review['status'],negative_cases=len(negatives),files=review['files'],deployment_sha256=sha(archive))))
