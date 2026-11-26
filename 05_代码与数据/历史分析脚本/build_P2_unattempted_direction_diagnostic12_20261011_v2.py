"""Separate two unattempted direction controls from the failed parent replay."""
from pathlib import Path
import ast,copy,hashlib,json,zipfile
R=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'英文小论文/code/p2_subspace_diagnostic12_20261011_v1'
new=R/'英文小论文/code/p2_direction_diagnostic12_20261011_v2'
out=R/'outputs/p2_direction_diagnostic12_20261011_v2'
assert not new.exists() and not out.exists()
captured=read(R/'outputs/turn6_reviewed_remote_actual_inputs.json')
data={}
for k,v in captured.items():
    b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256'];data[k]=json.loads(b)
actual=data['turn6_seed1_unattempted_subspace_actual_inputs_seed1']
assert actual['original_GPU_diagnostics_exist'] is False
cpu=actual['CPU_report'];complete=actual['CPU_completion'];failure=actual['replay_verification']
assert complete['exit_code']==0 and cpu['checkpoint_sha256']==failure['checkpoint_sha256']
assert cpu['source_sha256']==sha(old/'prepare.py') and cpu['protocol_sha256']==sha(old/'protocol.json')
failurefile=R/'outputs/p2_seed1_replay_mismatch_collection12_20261011_v1/extracted/candidate_reload_v1/verification.json'
assert read(failurefile)==failure and failure['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
assert failure['strict_loaded_tensors']==801 and failure['parameters']==10249437
assert not failure['all548_stage_arrays_exact'] and not failure['native_GT_hits_exact']
before={p.name:sha(p) for p in old.iterdir()}
contract='''"""Pure metadata contract: this does not upgrade a failed replay to a pass."""
def validate(preparation, completed, replay, protocol):
    assert preparation['seed']==replay['seed']==1 and replay['arm']=='p2_skip'
    assert preparation['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS'
    assert completed['exit_code']==0 and not completed['automatic_retry']
    assert replay['status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
    assert replay['strict_loaded_tensors']==801 and replay['parameters']==10249437
    assert not replay['all548_stage_arrays_exact'] and not replay['native_GT_hits_exact']
    assert replay['candidate_model_forwards']==137 and replay['baseline_model_forwards']==0
    assert replay['optimizer_updates']==replay['epochs_added']==0 and not replay['automatic_retry']
    assert replay['images']==548 and replay['stats_max_abs_fraction_difference']>0
    assert preparation['identity']==replay['identity']
    assert preparation['checkpoint_sha256']==replay['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
    assert preparation['source_sha256']==protocol['cpu_prepare_source_sha256']
    assert preparation['protocol_sha256']==protocol['cpu_protocol_sha256']
    assert completed['report_sha256']==protocol['CPU_report_sha256']
    assert completed['weights_npz_sha256']==protocol['CPU_weights_sha256']
    assert [x['mode'] for x in preparation['cases']]==protocol['interventions']==['dc16_norm','svd16_norm']
    assert all(x['numerical_rank']==16 and x['basis64_equal'] and x['relative_Frobenius_norm_error']<1e-6 for x in preparation['cases'])
    assert protocol['primary_contrast']=='svd16_norm_minus_dc16_norm'
    assert protocol['reference_replay_status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
    assert protocol['parent_forwards_added']==protocol['BASE_forwards_added']==protocol['training_epochs_added']==0
    assert protocol['formal_training_efficacy_or_unique_root_proven'] is False
'''
source=(old/'evaluate.py').read_text(encoding='utf8')
def replace(target,replacement):
    global source
    assert source.count(target)==1,target
    source=source.replace(target,replacement)
replace('Two native runtime counterfactuals on a positively replay-verified fixed12 P2 EMA.',
    'Two previously unattempted direction controls; preserve failed replay and compare controls directly.')
replace("assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]", "assert seed==1 and prefix=='/25085400120'")
replace("cpu=HERE.parent/f'seed{seed}_CPU'", "cpu=Path(prefix)/'DEIM/experiments/p2_subspace_diagnostic12_20261011_v1/seed1_CPU'")
replace("preparation['source_sha256']==sha(HERE/'prepare.py') and preparation['protocol_sha256']==sha(HERE/'protocol.json')",
        "preparation['source_sha256']==protocol['cpu_prepare_source_sha256'] and preparation['protocol_sha256']==protocol['cpu_protocol_sha256']")
replace("assert replay['status']=='PASS_SAVED_CANDIDATE_RELOAD' and replay['all548_stage_arrays_exact'] and replay['native_GT_hits_exact'] and replay['stats_max_abs_fraction_difference']==0",
    "from reference_contract import validate\n    validate(preparation,completed,replay,protocol)\n    assert sha(screen/'candidate_reload_v1/verification.json')==protocol['failed_replay_verification_sha256']\n    assert sha(cpu/'report.json')==protocol['CPU_report_sha256'] and sha(cpu/'weights.npz')==protocol['CPU_weights_sha256']\n    assert not (Path(prefix)/'DEIM/experiments/p2_subspace_diagnostic12_20261011_v1/seed1_GPU').exists(), 'These controls must still be unattempted'")
replace("parent,parent_hits,_=load_endpoint(run/'endpoint_epoch12');assert parent['identity']==receipt['identity'] and parent['native_stats']==replay['original_stats']",
    "parent,parent_hits,_=load_endpoint(run/'endpoint_epoch12');assert parent['identity']==receipt['identity'] and parent['native_stats']==replay['original_stats']\n    reloaded,_,_=load_endpoint(screen/'candidate_reload_v1/endpoint_epoch12')\n    assert reloaded['identity']==receipt['identity'] and reloaded['native_stats']==replay['reloaded_stats']\n    assert sha(screen/'candidate_reload_v1/endpoint_epoch12/manifest.json')==protocol['reloaded_reference_endpoint_manifest_sha256']")
replace("schema='P2_saved12_runtime_subspace_intervention_v1'", "schema='P2_unattempted_direction_control_v2'")
replace("write(out/'report.json',record);results.append(record);total_calls+=calls",
    "record.update(overall_percent=summary['overall_percent'],categories_percent=summary['categories_percent'],descriptive_reloaded_reference_delta_points=dict(AP=100*(summary['native_stats'][0]-reloaded['native_stats'][0]),APs=100*(summary['native_stats'][3]-reloaded['native_stats'][3])),reference_replay_failed_preserved=True,parent_delta_is_not_formal_efficacy=True)\n                write(out/'report.json',record);results.append(record);total_calls+=calls")
replace("status='TWO_FIXED12_P2_RUNTIME_SUBSPACE_INTERVENTIONS_NATIVE_EVALUATED'", "status='TWO_PREVIOUSLY_UNATTEMPTED_P2_DIRECTION_CONTROLS_NATIVE_EVALUATED'")
replace("scope='Runtime fixed-weight interventions on the same trainedEMA, with rank and weight Frobenius norm matched. Feature/output energy and training capacity are not matched. Fresh native matching is performed for these altered predictions. This is conditional model reliance, not new12epoch training efficacy or a unique causal root.'",
    "scope='Direct SVD16-minus-DC16 direction contrast with all non-restore checkpoint tensors exact. Both controls were previously unattempted. Rank and restore-weight norm are matched; operator approximation error and feature energy are not matched. Failed original parent replay remains failed; original/reloaded references are descriptive only. No parent/BASE forwards, training efficacy or unique root claim.'")
replace("write(root/'report.json',report);write(root/'completion.json'",
    "report.update(primary_contrast=dict(name='svd16_norm_minus_dc16_norm',delta_points={k:results[1]['overall_percent'][k]-results[0]['overall_percent'][k] for k in results[0]['overall_percent']}),parent_reference_replay_status=replay['status'],reference_replay_failed_preserved=True,old_failed_replay_retried=False,parent_model_forwards_added=0)\n        write(root/'report.json',report);write(root/'completion.json'")
assert 'PASS_SAVED_CANDIDATE_RELOAD' not in source
new.mkdir();out.mkdir()
(new/'evaluate.py').write_text(source,encoding='utf8');(new/'reference_contract.py').write_text(contract,encoding='utf8')
# The queue owns only its first attempt. Existing exact owners and their future
# finishers remain untouched; it waits for both handles and GPU availability.
wait='''"""Wait for reviewed exact owners; attempt two new controls once without signals."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
def live(h):
    pid=h.get('child_pid',h.get('pid'));expected=h.get('child_start_ticks',h.get('start_ticks'))
    try:
        state=Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()
        return state[19]==str(expected) and state[0]!='Z'
    except FileNotFoundError:return False
def main():
    p=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in p['remote_sources']}==p['remote_sources']
    assert not (ROOT/'seed1_GPU').exists()
    start=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0)
    write(ROOT/'queue_start.json',start);begin=time.monotonic()
    try:
        base=Path('/25085400120/DEIM/experiments');low=base/'p2_lowrank12_20261011_v3';remaining=base/'user_remaining_formal_closeout12_20261011_v1'
        while True:
            owners=list(p['predecessor_handles'])
            for f in [low/'seed1_p2_lowrank16/finisher_dispatch.json',remaining/'backbone_silu_dispatch.json',remaining/'backbone_attention_dispatch.json']:
                if f.is_file():owners.append(read(f))
            observed=[dict(h,same_live=live(h)) for h in owners]
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not any(x['same_live'] for x in observed) and not apps:break
            temp=ROOT/'queue_live.tmp';temp.write_text(json.dumps(dict(phase='waiting_exact_rank16_and_first_SiLU_attention_closeouts_GPU_idle',owners=observed,GPU_apps=apps,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())));temp.replace(ROOT/'queue_live.json')
            assert time.monotonic()-begin<p['max_wait_seconds'],'Timeout stops only this CPU waiter; no signals or restart'
            time.sleep(30)
        cmd=[sys.executable,'-B',str(HERE/'evaluate.py'),'--prefix','/25085400120','--seed','1']
        with (ROOT/'seed1_GPU.log').open('xb') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        write(ROOT/'GPU_dispatch.json',dict(pid=child.pid,start_ticks=Path('/proc/%s/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19],command=cmd,automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0))
        rc=child.wait();assert rc==0,'Preserve partial outputs; no retry'
        done=read(ROOT/'seed1_GPU/completion.json');assert done['exit_code']==0
        write(ROOT/'queue_completion.json',dict(status='TWO_UNATTEMPTED_DIRECTION_CONTROLS_COMPLETED_ONCE',completion_sha256=sha(ROOT/'seed1_GPU/completion.json'),process_signals=0,automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0))
    except BaseException:
        write(ROOT/'queue_failure.json',dict(status='FAILED_NO_RETRY_NO_PROCESS_SIGNALS',traceback=traceback.format_exc(),process_signals=0,automatic_retry=False,training_epochs=0,BASE_forwards=0,parent_forwards=0));raise
if __name__=='__main__':main()
'''
(new/'wait.py').write_text(wait,encoding='utf8')
manifest=R/'outputs/p2_seed1_replay_mismatch_collection12_20261011_v1/extracted/candidate_reload_v1/endpoint_epoch12/manifest.json'
native=read(old/'protocol.json')
owners=data['turn6_actual_rank16_native_same_epoch_seed1']
protocol=dict(schema='P2_unattempted_direction_control_v2',remote_sources={p.name:sha(p) for p in new.iterdir()},
    user28_sources=native['user28_sources'],GPU_uuid={'1':native['GPU_uuid']['1']},interventions=['dc16_norm','svd16_norm'],
    cpu_prepare_source_sha256=sha(old/'prepare.py'),cpu_protocol_sha256=sha(old/'protocol.json'),
    CPU_report_sha256=complete['report_sha256'],CPU_weights_sha256=complete['weights_npz_sha256'],
    parent_checkpoint_sha256=failure['checkpoint_sha256'],failed_replay_verification_sha256=sha(failurefile),
    reloaded_reference_endpoint_manifest_sha256=sha(manifest),reference_replay_status=failure['status'],
    primary_contrast='svd16_norm_minus_dc16_norm',parent_forwards_added=0,BASE_forwards_added=0,training_epochs_added=0,
    formal_training_efficacy_or_unique_root_proven=False,max_wait_seconds=172800,
    predecessor_handles=[owners['queue_handle'],owners['controller_handle'],owners['candidate_handle'],owners['remaining_formal_queue']['handle']],
    scope='Two previously unattempted controls, direct direction contrast. Failed replay is pinned and preserved; both parent snapshots are descriptive only. No parent replay retry or new training. Unlike V1, this is not gated as positive exact-parent efficacy.')
(new/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
for p in new.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
space={};exec(contract,space);validate=space['validate'];validate(cpu,complete,failure,protocol)
negatives=[]
for name,obj,key,value in [('erase_failed_replay','replay','status','PASS_SAVED_CANDIDATE_RELOAD'),('checkpoint_changed','replay','checkpoint_sha256','0'*64),
    ('BASE_forward','replay','baseline_model_forwards',1),('wrong_identity','replay','identity',{}),('new_training','protocol','training_epochs_added',1),
    ('claim_root','protocol','formal_training_efficacy_or_unique_root_proven',True)]:
    c,d,f,p=map(copy.deepcopy,(cpu,complete,failure,protocol));{'cpu':c,'complete':d,'replay':f,'protocol':p}[obj][key]=value
    try:validate(c,d,f,p)
    except (AssertionError,KeyError):negatives.append(name)
    else:raise AssertionError(name)
assert len(negatives)==6 and {p.name:sha(p) for p in old.iterdir()}==before
review=dict(status='REVIEWED_TWO_UNATTEMPTED_DIRECTION_CONTROLS_WITH_FAILED_REFERENCE_PRESERVED',
    files={p.name:sha(p) for p in new.iterdir()},parent_replay_status=failure['status'],old_GPU_attempts=0,new_parent_forwards=0,new_BASE_forwards=0,new_training_epochs=0,
    actual_CPU_projection_reused=True,actual_CPU_weight_modes=cpu['cases'],metadata_negative_cases=negatives,old_v1_sources_unchanged=True,
    scientific_primary_contrast=protocol['primary_contrast'],formal_efficacy_or_unique_root_proven=False,builder_sha256=sha(__file__))
(out/'source_review.json').write_text(json.dumps(review,indent=2),encoding='utf8')
with zipfile.ZipFile(out/'seed1_deployment_v2.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in new.iterdir():z.write(p,'code/'+p.name)
print(json.dumps(dict(status=review['status'],negative_cases=negatives,files=review['files'],deployment_sha256=sha(out/'seed1_deployment_v2.zip'))))
