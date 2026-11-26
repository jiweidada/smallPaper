"""One sequential B/M/Q pilot, frozen endpoints and hard stop gates."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent

def save(name,obj):
    p=ROOT/name
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    os.replace(tmp,p)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cancel(arms,reason):
    for arm in arms:
        with (ROOT/('arm_'+arm+'_CANCELLED_UNSTARTED.json')).open('x') as f:
            json.dump(dict(status='CANCELLED_UNSTARTED',arm=arm,reason=reason,
                           new_detector_forwards=0,optimizer_updates=0),f,indent=2)

with (ROOT/'formal_supervisor_once.lock').open('x') as f:
    f.write(str(os.getpid()))
start=time.monotonic()
results={}
child=None
try:
    plan=json.loads((ROOT/'protocol_FROZEN_v1.json').read_text())
    assert plan['arms']==['B','M','Q'] and plan['epochs_per_arm']==12
    for n,h in plan['files_sha256'].items():
        assert digest(ROOT/n)==h,n
    assert digest(ROOT/'capacity_protocol_FROZEN_v1.json')==plan['capacity_protocol_sha256']
    capacity=json.loads((ROOT/'capacity_result.json').read_text())
    assert capacity['status']=='PASS' and capacity['real_detector_forwards']==6
    save('supervisor_progress.json',dict(status='READY',remaining=['B','M','Q'],completed=[],supervisor_pid=os.getpid()))
    b_failed=False
    for index,arm in enumerate(['B','M','Q']):
        remaining=plan['maximum_seconds_total']-(time.monotonic()-start)
        assert remaining>0
        with (ROOT/(arm+'_stdout.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'train_A58.py'),'--arm',arm],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
            save('supervisor_progress.json',dict(status='FORMAL_RUNNING',arm=arm,child=identity,
                completed=list(results),remaining=['B','M','Q'][index+1:],supervisor_pid=os.getpid(),
                elapsed_seconds=time.monotonic()-start,protocol_sha256=digest(ROOT/'protocol_FROZEN_v1.json')))
            try:
                rc=child.wait(timeout=min(plan['maximum_seconds_per_arm'],remaining))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError(f'{arm} fixed wall-time limit reached; no retry')
        child=None
        assert rc==0,(arm,rc)
        result=json.loads((ROOT/('arm_'+arm)/'result.json').read_text())
        assert result['status']=='COMPLETED' and result['epochs']==12
        assert result['optimizer_updates']==9708 and result['full_model_forwards']==10159
        assert result['protocol_sha256']==digest(ROOT/'protocol_FROZEN_v1.json')
        results[arm]=result
        if arm=='B':
            m=result['endpoint_metrics']
            passed=m['AP']>=plan['baseline_learning_gate']['AP'] and m['APs']>=plan['baseline_learning_gate']['APs']
            save('B12_learning_gate.json',dict(status='PASS' if passed else 'FAIL',metrics=m,
                rule=plan['baseline_learning_gate'],candidate_trained=False))
            if not passed:
                b_failed=True
                cancel(['M','Q'],'Fresh B12 did not meet frozen learning gate; routing hypothesis untested')
                break
    if b_failed:
        judgment=dict(status='BASELINE_GATE_FAILED_CANDIDATE_UNTESTED',mainline_validated=False,
                      endpoint='B12 EMA',completed_arms=list(results),unstarted_cancelled=['M','Q'])
    else:
        b,m,q=(results[x] for x in ['B','M','Q'])
        delta=dict(Q_minus_B_APs=q['endpoint_metrics']['APs']-b['endpoint_metrics']['APs'],
                   Q_minus_M_APs=q['endpoint_metrics']['APs']-m['endpoint_metrics']['APs'],
                   Q_minus_B_AP=q['endpoint_metrics']['AP']-b['endpoint_metrics']['AP'],
                   Q_over_M_median_latency=q['latency']['median_ms']/m['latency']['median_ms'])
        gates=dict(APs_vs_B=delta['Q_minus_B_APs']>=.5,APs_vs_M=delta['Q_minus_M_APs']>=.3,
                   AP_preservation=delta['Q_minus_B_AP']>=-.2,latency=delta['Q_over_M_median_latency']<=1.2)
        judgment=dict(status='PASSES_MINIMAL_PILOT_REQUIRES_MECHANISM_AND_NOVELTY_EVIDENCE' if all(gates.values()) else 'RETIRED_THIS_FIXED_RECIPE_NO_RESCUE',
            mainline_validated=False,candidate_has_pilot_support=all(gates.values()),delta=delta,gates=gates,
            endpoint='fixed12 EMA, no best-epoch selection',completed_arms=list(results),
            limits='One seed and short training; no statistical significance, convergence or novelty proof')
    save('fixed12_judgment.json',judgment)
    save('supervisor_progress.json',dict(status='COMPLETED_BOUNDED_SCREEN',completed=list(results),
        remaining=[],child=None,supervisor_pid=os.getpid(),elapsed_seconds=time.monotonic()-start,
        judgment=judgment))
    save('supervisor_exit.json',dict(exit=0))
except BaseException:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
    completed=list(results)
    for arm in ['B','M','Q']:
        if not (ROOT/('arm_'+arm)).exists() and not (ROOT/('arm_'+arm+'_CANCELLED_UNSTARTED.json')).exists():
            cancel([arm],'Fixed pilot failed or reached wall-time cap; no automatic retry')
    save('supervisor_progress.json',dict(status='FAILED_STOPPED_NO_RETRY',completed=completed,
        remaining=[],child=None,supervisor_pid=os.getpid(),elapsed_seconds=time.monotonic()-start,
        traceback=traceback.format_exc()))
    save('supervisor_exit.json',dict(exit=1))
    raise
