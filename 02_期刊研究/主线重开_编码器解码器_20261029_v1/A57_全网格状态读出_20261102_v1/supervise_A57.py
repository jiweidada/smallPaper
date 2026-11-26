"""Bounded sequential B/M/Q supervisor; every child starts from common init."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def utc():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text('utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def identity(pid):
    try:
        stat=Path('/proc/%d/stat'%pid).read_text();parts=stat.rsplit(') ',1)[1].split()
        return {'pid':pid,'start_ticks':int(parts[19]),'state':parts[0]}
    except FileNotFoundError:return None
def owned_alive(row):
    current=identity(row['pid'])
    return current is not None and current['start_ticks']==row['start_ticks'] and current['state']!='Z'
def validate():
    plan=load(ROOT/'protocol_FROZEN_v1.json')
    assert all(sha(ROOT/n)==h for n,h in plan['files_sha256'].items())
    assert sha(ROOT/'common_initialization.pth')==plan['initialization_sha256']
    for name,row in plan.get('external_files',{}).items():assert sha(name)==row['sha256']
    return plan


def judge(metrics,design):
    b=metrics['B'];gate=design['baseline_learning_gate_at_fixed12']
    baseline=b['AP']>=gate['B_AP_at_least'] and b['APs']>=gate['B_APs_at_least']
    if not baseline:return {'status':'BASELINE_INADEQUATE_SHORT_HORIZON_STOPPED',
        'baseline_learning_gate_passed':False,'candidate_trained':False,
        'interpretation':'Host/12-epoch recipe inadequate; not a trained Q performance negative.'}
    if set(metrics)!={'B','M','Q'}:return {'status':'BASELINE_PASSED_M_Q_PENDING','baseline_learning_gate_passed':True}
    m,q=metrics['M'],metrics['Q'];gates=design['primary_candidate_gates_pp_all_required']
    delta={'Q_minus_B_APs':q['APs']-b['APs'],'Q_minus_M_APs':q['APs']-m['APs'],
           'Q_minus_B_AP':q['AP']-b['AP']}
    passed={name:delta[name]>=gates[name+'_at_least'] for name in delta}
    return {'status':'CANDIDATE_PASSED_FIXED12_SCREEN' if all(passed.values()) else 'RETIRED_EXACT_RECIPE_FIXED12',
        'baseline_learning_gate_passed':True,'candidate_trained':True,'delta_percentage_points':delta,
        'gates_passed':passed,'publication_novelty_cleared':False,'mainline_validated':False,
        'interpretation':'Single-seed bounded resource screen; no statistical, full-convergence or publication claim.'}


def stop_owned_child(child,row):
    actions=[]
    if child.poll() is None and owned_alive(row):
        assert os.getpgid(child.pid)==child.pid
        os.killpg(child.pid,signal.SIGTERM);actions.append('SIGTERM_OWNED_PROCESS_GROUP')
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if owned_alive(row):os.killpg(child.pid,signal.SIGKILL);actions.append('SIGKILL_OWNED_PROCESS_GROUP')
            child.wait(timeout=10)
    return actions


def main():
    plan=validate();design=load(ROOT/'pilot_design_FROZEN_v1.json')
    with (ROOT/'supervisor_once.lock').open('x') as f:f.write(str(os.getpid()))
    start=time.monotonic();metrics={};children=[];cancelled=[];active_child=None;active_identity=None
    state={'status':'STARTING','supervisor':identity(os.getpid()),'children':children,
           'protocol_sha256':sha(ROOT/'protocol_FROZEN_v1.json'),'metrics_fixed12':metrics,
           'queue':['B','M','Q'],'cancelled':cancelled,'post_training_queue':None,'start_utc':utc()}
    try:
        for arm in ['B','M','Q']:
            validate()
            if arm=='M' and not judge(metrics,design)['baseline_learning_gate_passed']:
                cancelled.extend(['M','Q']);state['verdict']=judge(metrics,design);break
            assert not (ROOT/('arm_'+arm)).exists()
            assert time.monotonic()-start<72*3600
            log=(ROOT/('train_'+arm+'.log')).open('xb')
            active_child=subprocess.Popen([sys.executable,'-u',str(ROOT/'train_A57.py'),'--arm',arm],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();active_identity=identity(active_child.pid);assert active_identity is not None
            child_row={'arm':arm,**active_identity,'start_utc':utc(),'exit_code':None};children.append(child_row)
            arm_start=time.monotonic()
            state.update(status='RUNNING',active_arm=arm,queue=[a for a in ['B','M','Q'] if a>arm],utc=utc())
            save(ROOT/'supervisor_state.json',state)
            while active_child.poll() is None:
                current=identity(active_child.pid)
                if current is None or current['state']=='Z':
                    active_child.wait(timeout=10);break
                assert current['start_ticks']==active_identity['start_ticks'],('Worker PID identity changed',active_identity)
                elapsed=time.monotonic()-start;arm_elapsed=time.monotonic()-arm_start
                if elapsed>=72*3600 or arm_elapsed>=32*3600:
                    child_row['termination_actions']=stop_owned_child(active_child,active_identity)
                    child_row['exit_code']=active_child.returncode
                    child_row['time_cap_reason']='72h total' if elapsed>=72*3600 else '32h arm'
                    raise RuntimeError('Frozen wall-time cap reached; no restart or extension')
                state.update(utc=utc(),total_elapsed_seconds=elapsed,arm_elapsed_seconds=arm_elapsed)
                save(ROOT/'supervisor_state.json',state);time.sleep(20)
            child_row.update(exit_code=active_child.returncode,exit_utc=utc(),
                             elapsed_seconds=time.monotonic()-arm_start)
            assert active_child.returncode==0,('Arm failed; cancel unstarted arms',arm,active_child.returncode)
            result=load(ROOT/('arm_'+arm)/'exit.json')
            assert result['status']=='COMPLETED_FIXED12_RAW' and result['exit_code']==0
            assert result['model_counter']['entries']==result['model_counter']['completions']==19867
            assert result['counts']['optimizer_updates']==9708 and result['counts']['epochs_completed']==12
            assert result['protocol_sha256']==state['protocol_sha256']
            metrics[arm]=result['metrics_fixed12'];state['verdict']=judge(metrics,design)
            active_child=None;active_identity=None
            save(ROOT/'supervisor_state.json',state)
        validate()
        state.update(status='COMPLETED_BOUNDED_SCREEN',active_arm=None,queue=[],post_training_queue=None,
                     exit_code=0,utc=utc(),total_elapsed_seconds=time.monotonic()-start)
        save(ROOT/'supervisor_state.json',state);save(ROOT/'supervisor_exit.json',state)
    except BaseException:
        if active_child is not None and active_child.poll() is None:
            state['termination_actions']=stop_owned_child(active_child,active_identity)
        done={row['arm'] for row in children}
        cancelled.extend(a for a in ['B','M','Q'] if a not in done and a not in cancelled)
        state.update(status='FAILED_PRESERVED_NO_RETRY',queue=[],post_training_queue=None,exit_code=1,
                     traceback=traceback.format_exc(),utc=utc(),total_elapsed_seconds=time.monotonic()-start)
        save(ROOT/'supervisor_state.json',state);save(ROOT/'supervisor_exit.json',state)
        raise


if __name__=='__main__':main()
