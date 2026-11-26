"""One authorized arm, no automatic retry or follow-on queue."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--arm', required=True)
    p.add_argument('--init', required=True); p.add_argument('--gpu', required=True)
    p.add_argument('--timeout-hours', type=float, default=72)
    args = p.parse_args(); root = Path(args.root).resolve(); work = root/'code'
    arm = root/args.arm; arm.mkdir(exist_ok=True)
    if (arm/'launch.json').exists() or (arm/'run').exists():
        raise RuntimeError('Arm already launched; no automatic retry')
    preflight = json.loads((root/f'preflight_{args.arm}.json').read_text())
    assert preflight['passed'] and preflight['checks']['real_batch_amp_optimizer_and_dn']
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
               OPENBLAS_NUM_THREADS='6', PYTHONDONTWRITEBYTECODE='1',SCREEN_RECEIPT_DIR=str(arm/'audit'))
    command = [sys.executable,'-u','train_screen.py','-c',str(root/f'{args.arm}.yml'),
               '-t',args.init,'--seed','2','--use-amp','-d','cuda:0','--output-dir',str(arm/'run')]
    start = time.monotonic(); log = arm/'train.log'
    with log.open('wb') as stdout:
        child = subprocess.Popen(command,cwd=work,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
        start_ticks = int(Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
        receipt = {'start_UTC':utc(),'supervisor_PID':os.getpid(),'trainer_PID':child.pid,
                   'trainer_start_ticks':start_ticks,'command':command,'cwd':str(work),
                   'gpu_UUID':args.gpu,'timeout_hours':args.timeout_hours,'human_request':'删除吧   跑你的实验计划；配置文件记得对齐'}
        (arm/'launch.json').write_text(json.dumps(receipt,indent=2))
        previous = -1; stalled = 0; hard_timeout = False
        while child.poll() is None:
            size = log.stat().st_size; stalled = stalled+1 if size == previous else 0; previous = size
            elapsed = time.monotonic()-start
            status = {'time_UTC':utc(),'alive':True,'elapsed_seconds':elapsed,'log_bytes':size,
                      'unchanged_checks':stalled,'advisory': 'output_stall_suspected' if stalled >= 3 else None}
            with (arm/'monitor.jsonl').open('a') as f: f.write(json.dumps(status)+'\n')
            if elapsed > args.timeout_hours*3600:
                (arm/'hard_timeout_notice.json').write_text(json.dumps(status,indent=2))
                os.killpg(child.pid,signal.SIGTERM); hard_timeout = True
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(60)
        code = child.wait()
    (arm/'exit.json').write_text(json.dumps({'end_UTC':utc(),'exit_code':code,'hard_timeout':hard_timeout,
                                            'elapsed_seconds':time.monotonic()-start},indent=2))
    sys.exit(code if code >= 0 else 1)


if __name__ == '__main__': main()
