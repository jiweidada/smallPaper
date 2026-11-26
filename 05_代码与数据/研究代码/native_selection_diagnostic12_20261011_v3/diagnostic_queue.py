"""Serialize saved-candidate diagnostic replays after native-owned training/closeouts."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time,traceback
from replay import verify_tools,sha,read,write,same_live
HERE=Path(__file__).resolve().parent

def state(r):return dict(r,same_live=same_live(r,'pid','start_ticks'))
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def paths(prefix,seed):
    b=Path(prefix)/'DEIM/experiments'
    return [(b/'original_stem12_20261011_v1'/f'seed{seed}_original_stem/bundle','original_stem'),
      (b/'native_stem_scratch12_20261011_v3'/f'seed{seed}_native_stem_scratch/bundle','native_stem_scratch'),
      (b/'stem_component12_20261011_v1'/f'seed{seed}_highfreq_rgb/bundle','highfreq_rgb')]
def observe(root,phase,**data):
    p=root/'live.tmp';p.write_text(json.dumps(dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)));p.replace(root/'live.json')

def verify_candidates(prefix,seed,protocol):
    """Validate every binding without importing any native runtime or model."""
    jobs=[]
    for bundle,arm in paths(prefix,seed):
        code=bundle.parent.parent/'code';allowed=protocol['allowed_native_roots'][code.parent.name]
        assert arm in allowed['arms']
        assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==allowed['files']
        receipt,plan=read(bundle/'receipt.json'),read(bundle/'plan.json')
        identity=receipt['identity'];native=read(code/'training_protocol.json')
        assert plan['arm']==identity['arm']==arm and plan['seed']==identity['seed']==seed
        assert plan['schema']==native['schema'] and plan['max_epochs']==12
        assert sha(bundle/'plan.json')==identity['plan_sha256']
        assert sha(bundle/'resolved.yml')==receipt['resolved_sha256']
        assert identity['implementation']==allowed['files']
        assert identity['protocol_sha256']==sha(code/'training_protocol.json')
        screen=Path(receipt['paths']['screen_root']);run=Path(receipt['paths']['output'])
        assert screen.resolve()==bundle.parent.resolve() and run.resolve()==(screen/'run').resolve()
        assert not same_live(read(screen/'controller_start.json'),'pid','start_ticks')
        assert not same_live(read(screen/'candidate_start.json'),'child_pid','child_start_ticks')
        assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
        candidate=read(screen/'candidate_completion.json')
        assert candidate['exit_code']==0 and not candidate['timed_out']
        done=read(run/'training_completion.json')
        assert done['status']=='NATIVE_FIT_FINISHED_EXACT12' and done['completed_epochs']==12
        assert done['entered_epochs']==list(range(12)) and done['identity']==identity
        assert sha(run/'last.pth')==done['last_checkpoint_sha256']
        assert sha(run/'endpoint_epoch12/manifest.json')==done['endpoint_manifest_sha256']
        assert sha(run/'stages_epoch12/manifest.json')==done['stage_manifest_sha256']
        closeout=screen/'post_training/completion.json'
        assert read(closeout)['status'] in ('STEM_FIXED12_RELOAD_AND_COLLECTION_FINISHED','NATIVE_STEM_SCRATCH_FIXED12_RELOAD_AND_COLLECTION_FINISHED')
        assert read(screen/'candidate_reload_v1/verification.json')['status']=='PASS_SAVED_CANDIDATE_RELOAD'
        reload_done=read(screen/'candidate_reload_v1/completion.json')
        assert reload_done['exit_code']==0 and not reload_done['timed_out']
        jobs.append(dict(arm=arm,bundle=str(bundle),checkpoint_sha256=done['last_checkpoint_sha256'],
                         native_closeout_sha256=sha(closeout)))
    assert len(jobs)==3 and len({j['arm'] for j in jobs})==3
    return jobs

def predecessor_handles(base,seed):
    result=[read(base/'native_stem_scratch12_20261011_v3/reference_start.json'),
            read(base/'stem_component12_20261011_v1/parallel_start.json'),
            read(base/'warm_stem12_20261011_v1'/f'seed{seed}_warm_stem/dispatcher_dispatch.json'),
            read(base/'user_backbone12_20261011_v1'/f'queue_dispatch_seed{seed}.json')]
    # A failed queue may leave its already-authorized trainers running. Include
    # actual children even after the parent disappears; never infer completion
    # from a parent handle or a temporarily idle GPU alone.
    for arm in ('backbone_silu','p2_skip','backbone_attention'):
        screen=base/'user_backbone12_20261011_v1'/f'seed{seed}_{arm}'
        for name in ('detector_preflight_dispatch.json','supervisor_dispatch.json'):
            if (screen/name).exists():result.append(read(screen/name))
        if (screen/'candidate_start.json').exists():
            r=read(screen/'candidate_start.json')
            result.append(dict(pid=r['child_pid'],start_ticks=r['child_start_ticks']))
    # Preserve and wait for the old observer, including its actual child.
    old=base/'native_selection_diagnostic12_20261011_v2'/f'seed{seed}_queue'
    if (old/'start.json').exists():result.append(read(old/'start.json'))
    for p in old.glob('*_dispatch.json'):result.append(read(p))
    for bundle,arm in paths(base.parent.parent,seed):
        worker=bundle.parent/'native_selection_diagnostic12_v2/start.json'
        if worker.exists():
            r=read(worker);result.extend([dict(pid=r['pid'],start_ticks=r['start_ticks']),dict(pid=r['controller_pid'],start_ticks=r['controller_start_ticks'])])
    return result


def verify_cpu_and_preserved_failure(base,seed,protocol,root):
    cpu=HERE.parent/f'seed{seed}_CPU'
    smoke=read(cpu/'import_verification.json');fixture=read(cpu/'fixture/verification.json');exit=read(cpu/'completion.json')
    assert exit['exit_code']==0 and smoke['status']=='PASS_REAL_ENTRYPOINT_STDLIB_QUEUE_AND_TORCH_IMPORT'
    assert smoke['source_sha256']==protocol['files'] and smoke['protocol_sha256']==sha(HERE/'diagnostic_protocol.json')
    assert smoke['CUDA_initialized'] is False and smoke['model_forwards']==smoke['training_epochs']==0
    assert fixture['status']=='PASS_NATIVE_PRODUCER_AND_STAGE_JOIN_CPU_FIXTURES' and fixture['CUDA_initialized'] is False
    assert fixture['diagnostic_code_sha256']==protocol['files'] and len(fixture['cases'])==2
    assert all(c['exact_fixture_outputs'] and c['additional_head_calls']==0 and c['actual_topk_calls']==1 and c['RNG_unchanged'] for c in fixture['cases'])
    assert exit['import_sha256']==sha(cpu/'import_verification.json') and exit['fixture_sha256']==sha(cpu/'fixture/verification.json')
    old=base/'native_selection_diagnostic12_20261011_v2'
    assert {n:sha(old/'code'/n) for n in protocol['preserved_v2_sources']}==protocol['preserved_v2_sources']
    failed=read(old/f'seed{seed}_queue/failure.json');assert failed['status']=='FAILED_NO_AUTOMATIC_RETRY'
    bundle,arm=paths(base.parent.parent,seed)[0]
    out=bundle.parent/'native_selection_diagnostic12_v2';log=(out/'worker.log').read_text()
    assert "ImportError: cannot import name 'Queue' from 'queue'" in log and 'selection_stage_trace.py' in log
    assert 'queue.py' in log and read(out/'completion.json')['exit_code']==1
    assert not any((out/n).exists() for n in ('verification.json','endpoint_epoch12','stages_epoch12','selection_epoch12'))
    write(root/'repair_precondition.json',dict(status='CPU_IMPORT_AND_NATIVE_FIXTURES_PASSED_OLD_PREMODEL_IMPORT_FAILURE_PRESERVED',
        import_report_sha256=sha(cpu/'import_verification.json'),fixture_sha256=sha(cpu/'fixture/verification.json'),
        old_queue_failure_sha256=sha(old/f'seed{seed}_queue/failure.json'),old_worker_log_sha256=sha(out/'worker.log'),
        old_worker_exit_sha256=sha(out/'completion.json'),new_training_epochs=0,new_BASE_forwards=0,
        old_instrumentation_failure_before_model_creation=True,failed_training_retried=False))


def main(prefix,seed):
    assert prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    protocol=verify_tools();root=HERE.parent/f'seed{seed}_queue';root.mkdir(exist_ok=False)
    b=Path(prefix)/'DEIM/experiments'
    owned=predecessor_handles(b,seed)
    write(root/'start.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        source_sha256=sha(__file__),protocol_sha256=sha(HERE/'diagnostic_protocol.json'),owned_predecessor_handles=owned,
        training_commands=0,BASE_forwards=0,existing_process_signals=0,automatic_retry=False))
    begun=time.monotonic()
    try:
        while True:
            owned=predecessor_handles(b,seed)
            if not any(same_live(r,'pid','start_ticks') for r in owned) and not gpu():break
            observe(root,'waiting_actual_predecessor_handles_and_GPU_idle',handles=[state(r) for r in owned])
            if time.monotonic()-begun>protocol['queue_timeout_seconds']:
                raise TimeoutError('Diagnostic wait expired; training is neither stopped nor restarted')
            time.sleep(30)
        # Every job must have positive native closeout evidence before any GPU
        # diagnostic is launched. Model/runtime imports occur only in children.
        verify_cpu_and_preserved_failure(b,seed,protocol,root)
        jobs=verify_candidates(prefix,seed,protocol)
        for job in jobs:
            assert not gpu(),'Another GPU process appeared; no diagnostic retry'
            command=[sys.executable,'-B',str(HERE/'replay.py'),'--bundle',job['bundle']]
            with (root/(job['arm']+'.log')).open('xb') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
                h=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=command)
                write(root/(job['arm']+'_dispatch.json'),h)
                while child.poll() is None:
                    observe(root,'diagnostic_owned_child',arm=job['arm'],handle=state(h))
                    time.sleep(15)
            write(root/(job['arm']+'_exit.json'),dict(exit_code=child.returncode,automatic_retry=False,training_epochs=0,BASE_forwards=0))
            assert child.returncode==0,'Selection replay failed; preserve all records, no retry or following job'
        write(root/'completion.json',dict(status='ALL_THREE_NATIVE_SELECTION_DIAGNOSTICS_FINISHED',jobs=jobs,
            total_expected_candidate_forwards=3*137,BASE_forwards=0,training_epochs=0,existing_process_signals=0,automatic_retry=False))
    except BaseException:
        write(root/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc(),training_epochs=0,BASE_forwards=0,existing_process_signals=0))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.prefix,a.seed)
