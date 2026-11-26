"""Own one candidate diagnostic process, with exit/timeout and archive evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().split()[21]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--seed',type=int,choices=(1,2),required=True)
    args=ap.parse_args();root=Path(args.root).resolve();p=json.loads((root/'protocol.json').read_text())
    for name,h in p['code_sha256'].items():assert sha(root/name)==h,name
    spec=p['seeds'][str(args.seed)]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert gpu==spec['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    run=root/f'seed{args.seed}';run.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(root/'runner.py'),'--seed',str(args.seed),'--protocol',str(root/'protocol.json'),
             '--inputs',str(root/'inputs.zip'),'--output',str(run/'outputs')]
    with (run/'console.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        write(run/'start.json',dict(utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),child_pid=child.pid,
            child_start_ticks=ticks(child.pid),command=command,timeout_seconds=p['timeout_seconds'],
            gpu_uuid=gpu,protocol_sha256=sha(root/'protocol.json')))
        timeout=False
        try:exit_code=child.wait(timeout=p['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timeout=True;child.kill();exit_code=child.wait()
    write(run/'completion.json',dict(utc=utc(),child_pid=child.pid,child_exit_code=exit_code,timed_out=timeout,child_live=False,no_retry=True))
    archive=root/f'terminal_seed{args.seed}_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in sorted(run.rglob('*')):
            if f.is_file():z.write(f,'run/'+f.relative_to(run).as_posix())
        for name in ('runner.py','controller.py','protocol.json'):z.write(root/name,'code/'+name)
    write(root/f'terminal_seed{args.seed}_v1_metadata.json',dict(utc=utc(),seed=args.seed,archive_bytes=archive.stat().st_size,
          archive_sha256=sha(archive),child_exit_code=exit_code,timed_out=timeout,protocol_sha256=sha(root/'protocol.json')))
    print(json.dumps(dict(status='COLLECTED_SUFFIX_DIAGNOSTIC',seed=args.seed,child_exit_code=exit_code,archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
