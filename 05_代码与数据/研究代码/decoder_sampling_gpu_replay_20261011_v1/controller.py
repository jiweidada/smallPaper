"""One-shot owned process with a fixed timeout and actual child exit evidence."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
    root=Path(a.root).resolve();p=json.loads((root/'protocol.json').read_text())
    assert sha(__file__)==p['controller_script_sha256']
    assert sha(root/'replay.py')==p['replay_script_sha256']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert gpu==p['gpu_uuid'],gpu
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not apps,apps
    run=root/'run';run.mkdir(exist_ok=False)
    command=[sys.executable,'-B',str(root/'replay.py'),'--inputs',str(root/'inputs.zip'),'--protocol',str(root/'protocol.json'),'--output',str(run/'outputs')]
    with (run/'console.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        write(run/'start.json',dict(utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),
                                   child_pid=child.pid,child_start_ticks=ticks(child.pid),command=command,
                                   timeout_seconds=p['timeout_seconds'],gpu_uuid=gpu,protocol_sha256=sha(root/'protocol.json')))
        timed_out=False
        try:code=child.wait(timeout=p['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();code=child.wait()
    write(run/'completion.json',dict(utc=utc(),child_exit_code=code,timed_out=timed_out,
                                     child_pid=child.pid,child_live=False,no_retry=True))
    archive=root/'replay_result_v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for f in sorted(run.rglob('*')):
            if f.is_file():z.write(f,'run/'+f.relative_to(run).as_posix())
        for name in ('replay.py','controller.py','protocol.json'):z.write(root/name,'code/'+name)
    write(root/'replay_result_v1_metadata.json',dict(utc=utc(),archive_bytes=archive.stat().st_size,
          archive_sha256=sha(archive),child_exit_code=code,timed_out=timed_out,protocol_sha256=sha(root/'protocol.json')))
    print(json.dumps(dict(status='COLLECTED',child_exit_code=code,archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
