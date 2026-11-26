"""Wait for exact prior owners and positive P2 replay; dispatch the native diagnostic once."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time,traceback
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False)
def live(record):
    try:
        f=Path('/proc/'+str(record['pid'])+'/stat').read_text().split();return f[21]==str(record['start_ticks']) and f[2]!='Z'
    except FileNotFoundError:return False
def main(prefix,seed):
    assert seed in (1,2) and prefix=={1:'/25085400120',2:'/24085403014'}[seed]
    spec=read(HERE/'protocol.json');assert {n:sha(HERE/n) for n in spec['remote_sources']}==spec['remote_sources']
    root=HERE.parent;cpu=root/f'seed{seed}_CPU';report=read(cpu/'report.json');co=read(cpu/'completion.json')
    assert co['exit_code']==0 and co['report_sha256']==sha(cpu/'report.json') and co['weights_npz_sha256']==sha(cpu/'weights.npz')
    assert report['status']=='PASS_ACTUAL_FIXED12_CPU_RANK16_NORM_MATCHED_WEIGHT_INTERVENTIONS'
    assert report['seed']==seed and report['source_sha256']==sha(HERE/'prepare.py') and report['protocol_sha256']==sha(HERE/'protocol.json')
    predecessors=spec['predecessor_handles'][str(seed)]
    assert not any((root/f'seed{seed}_{n}.json').exists() for n in ('wait_start','wait_completion','wait_failure','GPU_dispatch'))
    receipt=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],seed=seed,source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json'),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor_handles=predecessors,automatic_retry=False,new_training_epochs=0,new_BASE_forwards=0)
    write(root/f'seed{seed}_wait_start.json',receipt);begin=time.monotonic()
    try:
        replay=Path(prefix)/f'DEIM/experiments/user_backbone12_20261011_v1/seed{seed}_p2_skip/candidate_reload_v1/verification.json'
        while True:
            still=[dict(r,same_live=live(r)) for r in predecessors]
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            status=read(replay)['status'] if replay.exists() else None
            if status and status!='PASS_SAVED_CANDIDATE_RELOAD':raise AssertionError('Parent replay failed; no diagnostic, signal or retry')
            if not any(r['same_live'] for r in still) and not apps and status=='PASS_SAVED_CANDIDATE_RELOAD':break
            temp=root/f'seed{seed}_wait_live.tmp';temp.write_text(json.dumps(dict(phase='waiting_exact_prior_queues_Warm_dispatcher_P2_replay_and_GPU_idle',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessors=still,GPU_compute_apps=apps,parent_replay_status=status)))
            temp.replace(root/f'seed{seed}_wait_live.json')
            if time.monotonic()-begin>spec['max_wait_seconds']:raise TimeoutError('No signal, process restart or automatic retry')
            time.sleep(30)
        log=(root/f'seed{seed}_GPU.log').open('xb')
        cmd=[spec['python'][str(seed)],'-B',str(HERE/'evaluate.py'),'--prefix',prefix,'--seed',str(seed)]
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        log.close();f=Path('/proc/'+str(process.pid)+'/stat').read_text().split()
        write(root/f'seed{seed}_GPU_dispatch.json',dict(pid=process.pid,start_ticks=f[21],command=cmd,automatic_retry=False,new_training_epochs=0))
        rc=process.wait();assert rc==0,'Native intervention failed; preserve outputs, no retry'
        done=read(root/f'seed{seed}_GPU/completion.json');assert done['exit_code']==0
        write(root/f'seed{seed}_wait_completion.json',dict(status='SINGLE_NATIVE_SUBSPACE_DIAGNOSTIC_COMPLETED',receipt=receipt,exit_code=0,completion_sha256=sha(root/f'seed{seed}_GPU/completion.json'),existing_process_signals=0,automatic_retry=False,new_training_epochs=0))
    except BaseException:
        write(root/f'seed{seed}_wait_failure.json',dict(receipt=receipt,traceback=traceback.format_exc(),existing_process_signals=0,automatic_retry=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();main(a.prefix,a.seed)
