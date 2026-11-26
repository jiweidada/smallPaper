"""One scheduling-only handover, guarded before signaling never-started queues."""
from pathlib import Path
import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import zipfile


def main(prefix, seed, archive, digest):
    root = Path(prefix) / 'DEIM/experiments/user_backbone12_20261011_v1'
    old_handle = root / f'queue_dispatch_seed{seed}.json'
    old = json.loads(old_handle.read_bytes())
    assert old['source_sha256'] == 'deaec7e84fc08e0e7feda3b743a83ac59786156d1740c53db04b3507d5ddbd8e'
    assert hashlib.sha256(archive).hexdigest() == digest
    target = root / 'recovery_v2';target.mkdir(exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert set(z.namelist()) == {'dispatch.py','lock_support.py','recovery_manifest.json'}
        assert z.testzip() is None
        for name in z.namelist():
            with (target / name).open('xb') as f:f.write(z.read(name))
    sys.path.insert(0,str(target))
    from lock_support import snapshot, assert_recoverable, same_live, LocalLaunchLock, read, sha, mount_type
    manifest=read(target/'recovery_manifest.json')
    assert all(sha(target/n)==h for n,h in manifest['recovery_files'].items())
    state=snapshot(root,seed,old)
    assert same_live(old)
    proc=Path('/proc')/str(old['pid'])
    old_stat=(proc/'stat').read_text().split()
    old_cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    assert str(root/'code/dispatch.py') in old_cmd and old_stat[21]==str(old['start_ticks'])
    wchan=(proc/'wchan').read_text().strip()
    if seed==2:assert wchan=='nlmclnt_block', 'NFS-lock blockage no longer present; refuse handover'
    phase=read(root/f'queue_seed{seed}/queue_live.json')['phase']
    assert phase=='waiting_capacity'
    assert state['old_dispatcher_stopped'] is False
    proposed=dict(state,old_dispatcher_stopped=True)
    assert_recoverable(proposed)
    # Prove the local lock exclusion/release operations before touching the queue.
    lock=LocalLaunchLock(root,seed);lock.acquire()
    other=LocalLaunchLock(root,seed)
    try:
        other.acquire()
    except FileExistsError:
        excluded=True
    else:
        raise AssertionError('Local lock exclusion failed')
    lock.release();assert not lock.path.exists()
    original_bytes=old_handle.read_bytes()
    assert json.loads(original_bytes)==old
    with (root/f'queue_dispatch_seed{seed}.prelock_v1.json').open('xb') as f:f.write(original_bytes)
    snapshot_value=dict(seed=seed,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_dispatcher=old,original_cmdline=old_cmd,original_wchan=wchan,
        shared_directory_filesystem=mount_type(root),pre_signal_state=state,
        queue_live=read(root/f'queue_seed{seed}/queue_live.json'),
        local_atomic_lock_exclusion_passed=excluded,local_lock_released=True,
        configuration_preflight_or_training_started=False,frozen28_source_changed=False,
        archive_sha256=digest,old_training_process_signals=0)
    with (root/f'queue_handover_seed{seed}_v2_before.json').open('x') as f:json.dump(snapshot_value,f,indent=2)
    # Only the exact zero-child scheduling process is signaled. No trainer is stopped.
    assert same_live(old) and not (proc/'task'/str(old['pid'])/'children').read_text().split()
    assert snapshot(root,seed,old)['actual_configurations_started']==0
    os.kill(old['pid'],signal.SIGTERM)
    deadline=time.monotonic()+15
    while same_live(old) and time.monotonic()<deadline:time.sleep(.1)
    assert not same_live(old),'Old dispatcher still live; replacement MUST NOT start'
    after=snapshot(root,seed,old);assert_recoverable(after)
    command=[sys.executable,'-B',str(target/'dispatch.py'),'--prefix',prefix,'--seed',str(seed)]
    with (root/f'queue_recovery_seed{seed}_v2.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
    handle=dict(pid=child.pid,start_ticks=(Path('/proc')/str(child.pid)/'stat').read_text().split()[21],
        command=command,source_sha256=sha(target/'dispatch.py'),automatic_retry=False)
    with (root/f'queue_recovery_seed{seed}_v2_launch.json').open('x') as f:json.dump(handle,f,indent=2)
    deadline=time.monotonic()+8
    new_start=root/f'queue_seed{seed}/recovery_start_v2.json'
    while not new_start.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.1)
    assert child.poll() is None,'Replacement dispatcher failed; no retry'
    assert new_start.exists() and read(new_start)['pid']==child.pid
    assert read(old_handle)['pid']==child.pid and read(old_handle)['start_ticks']==handle['start_ticks']
    assert read(old_handle)['original_dispatcher']==old
    result=dict(status='PASS_ZERO_START_QUEUE_HANDOVER_TO_LOCAL_ATOMIC_LOCK',seed=seed,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_dispatcher=old,
        old_dispatcher_stopped=True,old_dispatcher_signal='SIGTERM',queued_dispatcher_signals=1,
        old_training_process_signals=0,replacement_handle=handle,replacement_same_live=same_live(handle),
        public_handle_updated_for_existing_dynamic_diagnostic_waiters=True,
        original_queue_start_preserved=True,pre_signal_state=state,post_signal_state=after,
        frozen28_source_verified=True,model_or_training_sources_changed=False,
        local_atomic_lock_exclusion_passed=excluded,shared_directory_filesystem=mount_type(root),
        original_wchan=wchan,archive_sha256=digest,training_retry=False,BASE_epochs_added=0,
        BASE_forwards_added=0,configuration_epochs_added_by_handover=0,max_epochs_each=12)
    with (root/f'queue_handover_seed{seed}_v2_after.json').open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--archive-b64',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    value=main(a.prefix,a.seed,base64.b64decode(a.archive_b64),a.sha256)
    raw=json.dumps(value).encode();h=hashlib.sha256(raw).hexdigest();b=base64.b64encode(raw).decode()
    for i in range(4):print('\rH'+str(i)+' '+h[16*i:16*i+16])
    print('\rBYTES '+str(len(raw)));print('\rB64_BEGIN')
    for i in range(0,len(b),64):print('\r'+b[i:i+64])
    print('\rB64_END')
