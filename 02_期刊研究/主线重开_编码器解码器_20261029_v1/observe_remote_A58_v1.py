"""Read-only: sources, exact process identities and actual-event prefixes."""
from pathlib import Path
import collections
import hashlib
import json
import subprocess
import time

ROOT=Path('/root/uav_a58_20261102_v1/pilot_v1')
def read(p):return json.loads(p.read_text())
def identity(pid,expected_ticks=None):
    p=Path('/proc')/str(pid)
    if not p.exists():return dict(pid=pid,exists=False)
    stat=(p/'stat').read_text().split()
    ticks=stat[21]
    cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    return dict(pid=pid,exists=True,state=stat[2],start_ticks=ticks,
                expected_ticks_match=expected_ticks is None or ticks==str(expected_ticks),command=cmd)

plan=read(ROOT/'protocol_FROZEN_v1.json')
source_errors=[n for n,h in plan['files_sha256'].items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
assert not source_errors,source_errors
deployment=read(ROOT/'formal_deployment_v1.json')
supervisor=read(ROOT/'supervisor_progress.json') if (ROOT/'supervisor_progress.json').exists() else {}
result=dict(server_epoch_time=time.time(),source_binding_verified=True,source_files=len(plan['files_sha256']),
    protocol_sha256=hashlib.sha256((ROOT/'protocol_FROZEN_v1.json').read_bytes()).hexdigest(),
    supervisor=supervisor,supervisor_identity=identity(deployment['pid'],deployment['start_ticks']),
    arms={},new_audit_detector_forwards=0)
if supervisor.get('child'):
    c=supervisor['child'];result['child_identity']=identity(c['pid'],c['start_ticks'])
for a in ['B','M','Q']:
    d=ROOT/('arm_'+a)
    arm=dict(directory_exists=d.exists())
    journal=d/'actual_events.jsonl'
    if journal.exists():
        raw=journal.read_bytes();parts=raw.splitlines();events=[];partial=False
        for i,line in enumerate(parts):
            try:events.append(json.loads(line))
            except json.JSONDecodeError:
                assert i==len(parts)-1;partial=True
        counts=collections.Counter(e['event'] for e in events)
        phases=collections.Counter(e.get('phase') for e in events if e['event']=='forward_start')
        assert counts['forward_start']<=10159 and counts['optimizer_returned']<=9708
        assert counts['forward_start']-counts['forward_complete'] in [0,1]
        arm.update(journal_counts=dict(counts),forward_phases=dict(phases),
                   journal_bytes=len(raw),journal_prefix_sha256=hashlib.sha256(raw).hexdigest(),partial_last_line=partial)
    for n in ['progress.json','result.json','initialization.json']:
        if (d/n).exists():arm[n]=read(d/n)
    if (d/'epochs.jsonl').exists():
        rows=[json.loads(s) for s in (d/'epochs.jsonl').read_text().splitlines() if s]
        arm['completed_epochs']=[dict(epoch=r['epoch'],mean_loss=r['mean_loss'],
            optimizer_updates=r['optimizer_updates'],residual_gate=r['residual_gate'],
            EMA_validation=r.get('EMA_validation')) for r in rows]
    cancel=ROOT/('arm_'+a+'_CANCELLED_UNSTARTED.json')
    if cancel.exists():arm['cancellation']=read(cancel)
    result['arms'][a]=arm
for n in ['supervisor_exit.json','fixed12_judgment.json','B12_learning_gate.json']:
    if (ROOT/n).exists():result[n]=read(ROOT/n)
result['GPU']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(result))
