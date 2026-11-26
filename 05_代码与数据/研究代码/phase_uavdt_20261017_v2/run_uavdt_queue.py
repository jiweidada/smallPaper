"""Finite three-job account queue; measured concurrency, no retries, distinct directories."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
PLAN=json.loads((ROOT/'plan.json').read_text())
ENV=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
STATE={'account':PLAN['account'],'dataset':'UAVDT_v1','pid':os.getpid(),'state':'starting','active_jobs':{},
       'training_completed_jobs':[],'completed_jobs':[],'jobs':PLAN['jobs'],'retries':0}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix('.next');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)
def state(name,**extra):
    STATE.update(extra,state=name,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(ROOT/'queue_state.json',STATE)
def name(arm):return f'uavdt_{arm}_full96_seed{PLAN["seed"]}'

def bind_checkpoint(job):
    import torch
    out=ROOT/'outputs'/job
    rows=[json.loads(line) for line in (out/'log.txt').read_text().splitlines() if line.strip()]
    rows=[r for r in rows if 'test_uavdt_eval_bbox' in r]
    assert [r['epoch'] for r in rows]==list(range(96))
    cp=out/'best_stg2.pth';checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    epoch=checkpoint['last_epoch'];assert 84<=epoch<=95
    metrics=rows[epoch]['test_uavdt_eval_bbox'];selected=checkpoint['uavdt_stage2_best_stat']
    assert selected['epoch']==epoch==checkpoint['best_stat']['epoch']
    assert abs(metrics[0]-selected['uavdt_eval_bbox'])<1e-12
    expected=max((r['test_uavdt_eval_bbox'][0],-r['epoch']) for r in rows[84:])
    assert epoch==-expected[1] and metrics[0]==expected[0]
    result={'job':job,'epochs':96,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'epoch_zero_based':epoch,
            'metric_names':['AP70','AR70'],'metrics_raw_0_to_1':dict(zip(['AP70','AR70'],metrics)),
            'log_sha256':sha(out/'log.txt'),'initialization_sha256':PLAN['initialization_sha256']}
    with (out/'checkpoint_log_binding.json').open('x') as f:json.dump(result,f,indent=2)
    return result

def verify_frozen():
    accepted=json.loads((ROOT/'final_acceptance.json').read_text());assert accepted['state']=='ready'
    for item in json.loads((ROOT/'frozen_manifest.json').read_text())['files']:
        p=ROOT/item['relative'];assert p.is_relative_to(ROOT) and sha(p)==item['sha256'],str(p)
    assert sha(Path(PLAN['initialization']))==PLAN['initialization_sha256']
    return accepted

def run():
    accepted=verify_frozen();release=json.loads((ROOT/'release_after_preflight.json').read_text())
    assert release['both_accounts_accepted'] is True and release['plan_sha256']==sha(ROOT/'plan.json')
    maximum=json.loads((ROOT/'capacity_receipt.json').read_text())['selected_concurrency'];assert 1<=maximum<=3
    STATE['maximum_concurrent_trainings']=maximum
    with (ROOT/'queue.lock').open('x') as f:json.dump({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]},f)
    pending=list(PLAN['jobs']);active={};failed=False
    state('training')
    while pending or active:
        for job,entry in list(active.items()):
            code=entry['process'].poll()
            if code is None:continue
            entry['handle'].close();active.pop(job);STATE['active_jobs'].pop(job)
            record=json.loads((ROOT/'jobs'/(job+'.json')).read_text());record.update(returncode=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if code==0:
                try:record.update(state='trained_pending_independent_eval',best=bind_checkpoint(job));STATE['training_completed_jobs'].append(job)
                except Exception as exc:record.update(state='failed',error=str(exc));failed=True
            else:record['state']='failed';failed=True
            write(ROOT/'jobs'/(job+'.json'),record)
        while pending and len(active)<maximum and not failed:
            arm=pending.pop(0);job=name(arm);out=ROOT/'outputs'/job
            assert not out.exists() and not (ROOT/'jobs'/(job+'.json')).exists()
            command=[sys.executable,'-u',str(ROOT/'code/train_uavdt.py'),'-c',PLAN['configs'][arm],'--seed',str(PLAN['seed']),
                     '--use-amp','-t',PLAN['initialization'],'--output-dir',str(out)]
            handle=(ROOT/'logs'/(job+'.log')).open('xb')
            process=subprocess.Popen(command,cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
            active[job]={'process':process,'handle':handle,'started':time.monotonic()}
            STATE['active_jobs'][job]={'pid':process.pid,'start_ticks':ticks,'output':str(out)}
            write(ROOT/'jobs'/(job+'.json'),{'job':job,'arm':arm,'seed':PLAN['seed'],'state':'running','pid':process.pid,'start_ticks':ticks,
                 'command':command,'expected_output_directory':str(out),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        for job,entry in active.items():
            if time.monotonic()-entry['started']>accepted['training_timeout_seconds']:
                # Keep other normal jobs running; owned process group is stopped only for this declared hard timeout.
                import signal
                os.killpg(entry['process'].pid,signal.SIGTERM)
                failed=True
        state('draining_after_failure' if failed else 'training',not_started=pending)
        if failed and not active:raise RuntimeError('Training failed; no retries or later tasks')
        time.sleep(10)
    state('independent_evaluation')
    for arm in PLAN['jobs']:
        job=name(arm)
        for split in ['val','test']:
            with (ROOT/'logs'/(job+'_'+split+'_eval.log')).open('xb') as log:
                p=subprocess.run([sys.executable,'-u',str(ROOT/'code/evaluate_uavdt.py'),'--config',PLAN['configs'][arm],
                    '--checkpoint',str(ROOT/'outputs'/job/'best_stg2.pth'),'--data',PLAN['data_root'],'--split',split,
                    '--out',str(ROOT/'results'/job/split)],cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            assert p.returncode==0,'Independent evaluation failed; no automatic retry'
        record=json.loads((ROOT/'jobs'/(job+'.json')).read_text());record.update(state='complete',independent_evaluation_complete=True)
        write(ROOT/'jobs'/(job+'.json'),record);STATE['completed_jobs'].append(job);state('independent_evaluation',completed_job=job)
    state('size_localization_analysis')
    with (ROOT/'logs/analysis.log').open('xb') as log:
        p=subprocess.run([sys.executable,'-u',str(ROOT/'code/analyze_uavdt.py'),str(ROOT)],cwd=ROOT/'code',env=ENV,
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,'Predeclared size/localization analysis failed; no automatic retry'
    state('complete')

if __name__=='__main__':
    try:run()
    except BaseException as exc:state('failed',error=type(exc).__name__+': '+str(exc));raise
