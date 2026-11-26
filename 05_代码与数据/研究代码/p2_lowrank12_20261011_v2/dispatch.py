"""One candidate per host; CPU/GPU preflight may run before the cross-seed gate."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time,traceback
from runtime import HERE,read,write,sha,verify_bundle

def live(r):
    try:
        f=Path('/proc/'+str(r['pid'])+'/stat').read_text().split()
        return f[21]==str(r['start_ticks']) and f[2]!='Z'
    except FileNotFoundError:return False

def resource(screen,plan):
    p=read(HERE/'training_protocol.json')
    gpu=[s.strip() for s in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    assert gpu[0]==p['GPU_uuid'][str(plan['seed'])]
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),UUID=gpu[0],total_MiB=int(gpu[1]),used_MiB=int(gpu[2]),free_MiB=int(gpu[3]),apps=apps,
        eligible=not apps and int(gpu[3])>=6656,scope='This distinct one-candidate queue requires actual GPU idle. It does not stop, resume, overwrite or restart previous jobs.')

def reviewed_two_seed_gate():
    gate=read(HERE.parent/'training_decision.json')
    assert gate['schema']=='p2_lowrank12_joint_runtime_release_v1' and gate['training_protocol_sha256']==sha(HERE/'training_protocol.json')
    assert gate['reviewed_model_source_sha256']==sha(HERE/'user_model.py') and gate['reviewed_branch_source_sha256']==sha(HERE/'lowrank_branch.py')
    assert gate['status']=='RELEASE_AFTER_TWO_SEED_NATIVE_RUNTIME_EVIDENCE' and gate['seeds']==[1,2]
    assert gate['BASE_training_epochs_added']==gate['BASE_model_forwards_added']==0
    assert not gate['training_efficacy_proven'] and not gate['novelty_proven']
    for s in ('1','2'):
        r=gate['runtime'][s]
        assert r['native_images']==548 and r['native_batches_per_case']==137 and r['independent_parent_EMA_replay_verified']
        assert r['dc16_norm']['APs']<r['parent']['APs']
        assert r['svd16_norm']['APs']>=r['parent']['APs'] and r['svd16_norm']['AP']>=r['parent']['AP']
        assert len(r['runtime_report_sha256'])==len(r['parent_checkpoint_sha256'])==64
    return gate

def authorize_training(screen,plan,protocol,receipt):
    gate=reviewed_two_seed_gate()
    proof=read(screen/'detector_preflight_v1/report.json')
    assert proof['status']=='PASS_LOW_RANK_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES' and proof['identity']==receipt['identity']
    assert proof['optimizer_updates']==proof['training_epochs']==0
    assert proof['preflight_source_sha256']==sha(HERE/'preflight.py') and proof['protocol_sha256']==sha(HERE/'training_protocol.json')
    assert resource(screen,plan)['eligible']
    write(screen/'training_release_receipt.json',dict(gate_sha256=sha(HERE.parent/'training_decision.json'),source_sha256=sha(__file__),resource=resource(screen,plan)))

def wait_old_closeouts_and_idle(screen,plan,paths):
    assert resource(screen,plan)['eligible']
    write(screen/'old_closeout_handles_stopped.json',dict(scope='Own candidate stopped; all previous pipelines were terminal before joint release.',actual_GPU_idle=True,existing_process_signals=0))

def publish(root,phase,**data):
    p=root/'queue_live.tmp';p.write_text(json.dumps(dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data)));p.replace(root/'queue_live.json')

def child(root,name,command):
    with (root/(name+'.log')).open('xb') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
    r=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],command=command)
    write(root/(name+'_dispatch.json'),r)
    return p,r

def main(prefix,seed):
    assert prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    root=HERE.parent;bundle=root/f'seed{seed}_p2_lowrank16/bundle'
    receipt,plan,paths=verify_bundle(bundle);screen=paths['screen_root']
    queue=root/f'queue_seed{seed}';queue.mkdir(exist_ok=False)
    write(queue/'start.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],source_sha256=sha(__file__),seed=seed))
    began=time.monotonic()
    try:
        while not (root/'training_decision.json').exists() or not resource(screen,plan)['eligible']:
            publish(queue,'waiting_positive_joint_runtime_review_and_actual_GPU_idle',decision_exists=(root/'training_decision.json').exists(),resource=resource(screen,plan))
            assert time.monotonic()-began<172800,'Wait expired; no signals or retry'
            time.sleep(30)
        reviewed_two_seed_gate()
        # A preflight may have been completed in a separate private process.
        if not (screen/'detector_preflight_v1').exists():
            p,h=child(screen,'detector_preflight',[sys.executable,'-B',str(HERE/'preflight.py'),'--bundle',str(bundle)])
            publish(queue,'full_detector_preflight',handle=h);p.wait()
            write(screen/'detector_preflight_completion.json',dict(exit_code=p.returncode,training_epochs=0,optimizer_updates=0))
            assert p.returncode==0,'No training after failed preflight'
        else:
            assert read(screen/'detector_preflight_completion.json')['exit_code']==0
            assert not live(read(screen/'detector_preflight_dispatch.json'))
        assert resource(screen,plan)['eligible']
        p,h=child(screen,'supervisor',[sys.executable,'-B',str(HERE/'supervise.py'),'--bundle',str(bundle)])
        publish(queue,'one_authorized_fixed12_candidate',handle=h);p.wait()
        assert p.returncode==0,'Failed configuration never extended/restarted'
        assert read(screen/'run/training_completion.json')['completed_epochs']==12
        controller=read(screen/'controller_start.json');assert not live(controller)
        p,h=child(screen,'finisher',[sys.executable,'-B',str(HERE/'finish.py'),'--bundle',str(bundle),'--controller-pid',str(controller['pid']),'--controller-start-ticks',str(controller['start_ticks'])])
        publish(queue,'one_independent_saved_EMA_closeout',handle=h);p.wait();assert p.returncode==0
        write(queue/'completion.json',dict(status='ONE_LOW_RANK_CONFIG_FIXED12_TRAIN_RELOAD_COLLECTION_FINISHED',seed=seed,BASE_forwards=0,BASE_epochs=0,automatic_retry=False))
    except BaseException:
        write(queue/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc(),existing_process_signals=0,automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.prefix,a.seed)
