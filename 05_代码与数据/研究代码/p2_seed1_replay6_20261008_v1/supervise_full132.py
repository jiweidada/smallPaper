"""Launch exactly one full132 candidate after the authorized completed24 gate."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from screen_runtime import file_sha, source_manifest
import hashlib


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    root=Path(sys.argv[1]);work=root/'code';arm=root/'p2_seed2';arm.mkdir(exist_ok=True)
    if (arm/'launch.json').exists() or (arm/'run').exists():raise RuntimeError('Full132 arm was already launched; automatic retry prohibited')
    gate=json.loads((root/'gate_24_to_132.json').read_text())
    assert gate['allowed'] and gate['completed_epochs']==24
    assert gate['delta_APs_pp']>=.8 and gate['delta_AP_pp']>=-.2
    assert json.loads((root/'full132_amp_preflight.json').read_text())['passed']
    assert json.loads((root/'full132_cpu_checks.json').read_text())['passed']
    alignment=json.loads((root/'full132_configuration_alignment.json').read_text())
    assert file_sha(root/'p2_full132.yml')==alignment['full_config_sha256']
    assert hashlib.sha256(json.dumps(source_manifest(work),sort_keys=True).encode()).hexdigest()==alignment['source_digest']
    init='/24085403014/DEIM/experiments/enc_nms_screen24_20261006_v1/init/common_init_seed0.pth'
    command=[sys.executable,'-u','train_screen.py','-c',str(root/'p2_full132.yml'),'-t',init,
             '--seed','2','--use-amp','-d','cuda:0','--output-dir',str(arm/'run')]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-43038ae9-d619-20e2-0216-822fd1c50fed',
             OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONDONTWRITEBYTECODE='1',SCREEN_RECEIPT_DIR=str(arm/'audit'))
    start=time.monotonic();log=arm/'train.log';hard_timeout=False
    with log.open('wb') as output:
        child=subprocess.Popen(command,cwd=work,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
        receipt={'start_UTC':utc(),'supervisor_PID':os.getpid(),'trainer_PID':child.pid,'trainer_start_ticks':ticks,
                 'command':command,'cwd':str(work),'gate_sha256':file_sha(root/'gate_24_to_132.json'),
                 'human_authorization':'如果24轮效果可以  你就执行132 完整的  现在给你授权了',
                 'common_init_fresh_start':True,'timeout_hours':72}
        (arm/'launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        previous=-1;stalled=0
        while child.poll() is None:
            size=log.stat().st_size;stalled=stalled+1 if size==previous else 0;previous=size;elapsed=time.monotonic()-start
            state={'time_UTC':utc(),'alive':True,'elapsed_seconds':elapsed,'log_bytes':size,
                   'advisory':'output_stall_suspected' if stalled>=3 else None}
            with (arm/'monitor.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
            if elapsed>72*3600:
                (arm/'hard_timeout_notice.json').write_text(json.dumps(state,indent=2));os.killpg(child.pid,signal.SIGTERM);hard_timeout=True
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(60)
        code=child.wait()
    (arm/'exit.json').write_text(json.dumps({'end_UTC':utc(),'exit_code':code,'hard_timeout':hard_timeout,'elapsed_seconds':time.monotonic()-start},indent=2))
    sys.exit(code if code>=0 else 1)


if __name__=='__main__':main()
