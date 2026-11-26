"""Second authorized recovery; preserve all prior recovery artifacts and the frozen recipe."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_uavdt_queue as original

ROOT=original.ROOT
PLAN=original.PLAN
ENV=original.ENV
STATE=original.STATE
write=original.write
state=original.state
PROOF=json.loads((ROOT/'recovery_20261019_v2/resume_preflight.json').read_text())
STAMP=PROOF['recovery_id']
ARCHIVE=ROOT/'recovery_20261019_v2'/STAMP

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def live_owned_processes():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            args=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            if any(s.startswith(str(ROOT/'code')+'/') for s in args):
                stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
                if stat[0]!='Z':found.append({'pid':int(proc.name),'command':args})
        except (FileNotFoundError,PermissionError):pass
    return found

def preflight():
    import torch
    assert not ARCHIVE.exists(),'Recovery is single use; inspect existing receipt rather than launch twice'
    assert not live_owned_processes(),'An existing owned training/controller is still running'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    assert not any(s.strip().isdigit() for s in gpu.splitlines()),'GPU is not idle'
    accepted=original.verify_frozen()
    assert sha(ROOT/'plan.json')==PROOF['plan_sha256']
    validation_path=ROOT/'recovery_20261019_v2/checkpoint_validation.json'
    assert sha(validation_path)==PROOF['checkpoint_validation_sha256']
    validation=json.loads(validation_path.read_text());assert validation['all_passed']
    checked={j['job']:j for j in validation['jobs']}
    assert sha(ROOT/'frozen_manifest.json')==accepted['frozen_manifest_sha256']
    release=json.loads((ROOT/'release_after_preflight.json').read_text())
    assert release['both_accounts_accepted'] and release['plan_sha256']==PROOF['plan_sha256']
    assert PLAN['epoches']==96 and PLAN['maximum_concurrent_trainings']==3
    assert PLAN['seed']==PROOF['seed'] and PLAN['account']==PROOF['account']
    assert json.loads((ROOT/'capacity_receipt.json').read_text())['selected_concurrency']==3
    assert sha(Path(PLAN['initialization']))==PLAN['initialization_sha256']
    data=Path(PLAN['data_root'])
    for split,expected in PLAN['annotations_sha256'].items():
        assert sha(data/'annotations'/(split+'.json'))==expected
    sources={j['job']:j for j in PROOF['jobs']}
    for arm in PLAN['jobs']:
        job=original.name(arm);out=ROOT/'outputs'/job;cp=out/'last.pth';proof=sources[job]
        assert cp.resolve().parent==out.resolve() and sha(cp)==proof['checkpoint_sha256']
        assert checked[job]['passed'] and checked[job]['checkpoint_sha256']==proof['checkpoint_sha256']
        loaded=torch.load(cp,map_location='cpu',weights_only=False)
        assert loaded['last_epoch']==proof['last_epoch']<95
        assert {'model','ema','optimizer','scaler','lr_warmup_scheduler'}.issubset(loaded)
        assert loaded['optimizer']['state'] and loaded['ema']['updates']>0
        rows=[json.loads(s) for s in (out/'log.txt').read_text().splitlines() if s.strip()]
        rows=[r for r in rows if 'test_uavdt_eval_bbox' in r]
        assert [r['epoch'] for r in rows]==list(range(loaded['last_epoch']+1)), 'Checkpoint/log boundary mismatch'
        del loaded
    return sources

def archive_and_lock(sources):
    ARCHIVE.mkdir(parents=True,exist_ok=False)
    for relative in ['queue_state.json','queue.lock','launch_receipt.json']:
        source=ROOT/relative
        if source.exists():shutil.copy2(source,ARCHIVE/source.name)
    for job,source in sources.items():
        dest=ARCHIVE/job;dest.mkdir()
        for path in [ROOT/'jobs'/(job+'.json'),ROOT/'outputs'/job/'args.json',ROOT/'outputs'/job/'log.txt']:
            if path.exists():shutil.copy2(path,dest/path.name)
        cp=ROOT/'outputs'/job/'last.pth'
        shutil.copy2(cp,dest/'last.pth')
        assert sha(dest/'last.pth')==source['checkpoint_sha256']
    if (ROOT/'queue.lock').exists():(ROOT/'queue.lock').unlink()
    with (ROOT/'queue.lock').open('x') as f:
        json.dump({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                   'recovery_id':STAMP},f)
    receipt={'account':PLAN['account'],'state':'resumed_once','pid':os.getpid(),
             'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
             'controller':str(Path(__file__).resolve()),'maximum_concurrent_trainings':3,
             'root':str(ROOT),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'recovery_id':STAMP,'source_preflight_sha256':sha(ROOT/'recovery_20261019_v2/resume_preflight.json')}
    write(ROOT/'launch_receipt.json',receipt)
    write(ARCHIVE/'resume_receipt.json',receipt)
    STATE.update(recovery_id=STAMP,maximum_concurrent_trainings=3)

def finish_evaluation():
    state('independent_evaluation')
    for arm in PLAN['jobs']:
        job=original.name(arm)
        for split in ['val','test']:
            with (ROOT/'logs'/(job+'_'+split+'_eval.log')).open('xb') as log:
                process=subprocess.run([sys.executable,'-u',str(ROOT/'code/evaluate_uavdt.py'),
                    '--config',PLAN['configs'][arm],'--checkpoint',str(ROOT/'outputs'/job/'best_stg2.pth'),
                    '--data',PLAN['data_root'],'--split',split,'--out',str(ROOT/'results'/job/split)],
                    cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            assert process.returncode==0,'Independent evaluation failed; no automatic retry'
        path=ROOT/'jobs'/(job+'.json');record=json.loads(path.read_text())
        record.update(state='complete',independent_evaluation_complete=True);write(path,record)
        STATE['completed_jobs'].append(job);state('independent_evaluation',completed_job=job)
    state('size_localization_analysis')
    with (ROOT/'logs/analysis.log').open('xb') as log:
        process=subprocess.run([sys.executable,'-u',str(ROOT/'code/analyze_uavdt.py'),str(ROOT)],
            cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert process.returncode==0,'Analysis failed; no automatic retry'
    state('complete')

def run():
    sources=preflight();archive_and_lock(sources)
    active={};failed=False
    state('resuming')
    for arm in PLAN['jobs']:
        job=original.name(arm);out=ROOT/'outputs'/job
        command=[sys.executable,'-u',str(ROOT/'code/train_uavdt.py'),'-c',PLAN['configs'][arm],
                 '--seed',str(PLAN['seed']),'--use-amp','-r',str(out/'last.pth'),'--output-dir',str(out)]
        handle=(ROOT/'logs'/(job+'.log')).open('ab')
        handle.write(('\nRESUME_FROM_OWN_CHECKPOINT '+json.dumps(sources[job])+'\n').encode());handle.flush()
        process=subprocess.Popen(command,cwd=ROOT/'code',env=ENV,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        active[job]={'process':process,'handle':handle,'started':time.monotonic()}
        STATE['active_jobs'][job]={'pid':process.pid,'start_ticks':ticks,'output':str(out)}
        write(ROOT/'jobs'/(job+'.json'),{'job':job,'arm':arm,'seed':PLAN['seed'],'state':'running',
            'pid':process.pid,'start_ticks':ticks,'command':command,'expected_output_directory':str(out),
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recovery_id':STAMP,
            'resumed_from':sources[job]})
    state('training')
    while active:
        for job,entry in list(active.items()):
            code=entry['process'].poll()
            if code is None:
                if time.monotonic()-entry['started']>PLAN['training_timeout_seconds']:
                    import signal
                    os.killpg(entry['process'].pid,signal.SIGTERM);failed=True
                continue
            entry['handle'].close();active.pop(job);STATE['active_jobs'].pop(job)
            path=ROOT/'jobs'/(job+'.json');record=json.loads(path.read_text())
            record.update(returncode=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if code==0:
                try:
                    record.update(state='trained_pending_independent_eval',best=original.bind_checkpoint(job))
                    STATE['training_completed_jobs'].append(job)
                except Exception as exc:record.update(state='failed',error=repr(exc));failed=True
            else:record['state']='failed';failed=True
            write(path,record)
        state('draining_after_failure' if failed else 'training')
        if active:time.sleep(10)
    if failed:raise RuntimeError('An existing job failed; no automatic retry or later tasks')
    finish_evaluation()

if __name__=='__main__':
    try:run()
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        lock=json.loads((ROOT/'queue.lock').read_text()) if (ROOT/'queue.lock').exists() else {}
        if lock.get('pid')==os.getpid():state('failed',error=error)
        else:write(ROOT/'recovery_20261019_v2'/('preflight_failure_'+STAMP+'.json'),{'error':error,'pid':os.getpid()})
        raise
