"""One bounded sequential B->M->Q queue; immutable code, no retry or rescue."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/root/uav_a60_20261102_v1/pilot_v1')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def proc_ticks(pid):return Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1].split()[19]

def stop_own_child(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def validate_arm(arm,scope):
    out=ROOT/arm;r=json.loads((out/'result.json').read_text());exit_record=json.loads((out/'exit.json').read_text())
    assert exit_record['exit']==0 and r['status']=='COMPLETED_FIXED12_RAW'
    assert r['counts']==dict(started=78514,completed=78514,updates=9708,backwards=77652,epochs=12,
        train_forwards=77652,eval_forwards=822,latency_forwards=40)
    assert r['protocol_sha256']==sha(ROOT/'protocol_FROZEN_v1.json')
    assert r['initialization_sha256']==scope['initialization_sha256']
    assert r['source_sha256']==scope['source_sha256']
    events=[json.loads(x) for x in (out/'actual_events.jsonl').read_text().splitlines()]
    assert sum(x['event']=='forward_start' for x in events)==78514
    assert sum(x['event']=='forward_complete' for x in events)==78514
    assert sum(x['event']=='optimizer_returned' for x in events)==9708
    assert sha(out/'epoch12_raw.pth')==r['fixed12_checkpoint_sha256']
    assert sha(out/'fixed12_optimizer.pth')==r['optimizer_sha256']
    for epoch in range(1,13):
        row=json.loads((out/('epoch%02d_flow.json'%epoch)).read_text())
        if arm!='B':
            b=json.loads((ROOT/'B'/('epoch%02d_flow.json'%epoch)).read_text())
            assert row['raw_stream_sha256']==b['raw_stream_sha256']
        if arm=='Q':
            m=json.loads((ROOT/'M'/('epoch%02d_flow.json'%epoch)).read_text())
            assert row['noisy_stream_sha256']==m['noisy_stream_sha256']
            assert row['condition_stream_sha256']!=m['condition_stream_sha256']
    return r

def main():
    assert not (ROOT/'supervisor_start_v1.json').exists()
    scope=json.loads((ROOT/'protocol_FROZEN_v1.json').read_text())
    for name,h in scope['source_sha256'].items():assert sha(ROOT/'code'/name)==h,name
    assert sha(ROOT/'dataset_manifest_v1.json')==scope['dataset_manifest_sha256']
    assert sha(ROOT/'common_init_A60_v1.pth')==scope['initialization_sha256']
    assert all(not (ROOT/a).exists() for a in ['B','M','Q'])
    save('supervisor_start_v1.json',dict(supervisor_pid=os.getpid(),supervisor_start_ticks=proc_ticks(os.getpid()),
        protocol_sha256=sha(ROOT/'protocol_FROZEN_v1.json'),queue=['B','M','Q'],no_retry=True,
        per_arm_seconds=scope['maximum_seconds_per_arm'],total_seconds=scope['maximum_seconds_total']))
    results={};child=None;started=time.monotonic();current=None
    try:
        for arm in ['B','M','Q']:
            current=arm
            for name,h in scope['source_sha256'].items():assert sha(ROOT/'code'/name)==h,name
            remaining=scope['maximum_seconds_total']-(time.monotonic()-started)
            assert remaining>0
            with (ROOT/(arm+'_stdout.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/train_A60_v1.py'),'--arm',arm],
                    cwd=ROOT/'code',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(arm+'_launch_v1.json',dict(pid=child.pid,start_ticks=proc_ticks(child.pid),
                    supervisor_pid=os.getpid(),supervisor_start_ticks=proc_ticks(os.getpid()),
                    protocol_sha256=sha(ROOT/'protocol_FROZEN_v1.json'),arm=arm))
                try:code=child.wait(timeout=min(scope['maximum_seconds_per_arm'],remaining))
                except subprocess.TimeoutExpired:
                    stop_own_child(child);raise RuntimeError('Frozen wall-clock limit exceeded; no retry')
            save(arm+'_worker_reaped_v1.json',dict(exit=code,original_pid=child.pid,wait_completed=True))
            assert code==0,arm+' failed; cancel unstarted queue without retry'
            results[arm]=validate_arm(arm,scope);child=None
            if arm=='B':
                m=results['B']['metrics'];gate=m['AP']>=8. and m['APs']>=3.
                save('B_learning_gate_v1.json',dict(passed=gate,AP=m['AP'],APs=m['APs'],minimum_AP=8.,minimum_APs=3.))
                if not gate:
                    save('supervisor_exit_v1.json',dict(exit=0,status='BASELINE_NOT_LEARNED_CANCEL_M_Q',
                        completed_arms=['B'],cancelled_unstarted_arms=['M','Q'],remaining_queue=[],
                        mainline_validated=False,candidate_hypothesis_tested=False,
                        elapsed_seconds=time.monotonic()-started,no_retry=True))
                    return
        b=results['B'];m=results['M'];q=results['Q']
        delta=dict(Q_minus_B_APs=q['metrics']['APs']-b['metrics']['APs'],
            Q_minus_M_APs=q['metrics']['APs']-m['metrics']['APs'],Q_minus_B_AP=q['metrics']['AP']-b['metrics']['AP'],
            Q_over_M_median_latency=q['latency']['median_ms']/m['latency']['median_ms'])
        gates=dict(APs_vs_B=delta['Q_minus_B_APs']>=.5,APs_vs_M=delta['Q_minus_M_APs']>=.3,
            AP_preservation=delta['Q_minus_B_AP']>=-.2,latency=delta['Q_over_M_median_latency']<=1.10)
        passed=all(gates.values())
        save('fixed12_judgment_v1.json',dict(status='PILOT_SUPPORT_REQUIRES_INDEPENDENT_CONFIRMATION' if passed else 'RETIRED_THIS_FIXED_RECIPE_NO_RESCUE',
            candidate_has_pilot_support=passed,mainline_validated=False,delta=delta,gates=gates,
            metrics={a:r['metrics'] for a,r in results.items()},latencies={a:r['latency'] for a,r in results.items()},
            full_input_stream_audit_passed=True,endpoint='fixed12 raw model; no best-epoch choice',
            limitations='One short seed, exploratory candidate selection; no significance, convergence, global novelty or submission guarantee.'))
        save('supervisor_exit_v1.json',dict(exit=0,status='ALL_THREE_FIXED12_COMPLETED',completed_arms=['B','M','Q'],
            remaining_queue=[],model_forwards=235542,optimizer_updates=29124,formal_epochs=36,
            candidate_has_pilot_support=passed,mainline_validated=False,elapsed_seconds=time.monotonic()-started,no_retry=True))
    except BaseException:
        stop_own_child(child)
        save('supervisor_exit_v1.json',dict(exit=1,status='FAILED_NO_RETRY_CANCEL_UNSTARTED',current_arm=current,
            completed_arms=list(results),remaining_queue=[],mainline_validated=False,no_retry=True,
            elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
