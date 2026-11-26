"""One-shot detached launch; no automatic resume or retry."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from supervise_A57 import ROOT,validate,identity,save,utc


def main():
    assert ROOT==Path('/root/uav_a57_20261102_v1')
    validate()
    assert not (ROOT/'supervisor_exit.json').exists()
    assert all(not (ROOT/('arm_'+a)).exists() for a in ['B','M','Q'])
    assert shutil.disk_usage(ROOT).free>20*(1<<30)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not gpu,('GPU compute process present; inspect before launch',gpu)
    watched=['capacity_native.py','capacity_batch4.py','eval_integration_A57.py','supervise_A57.py','train_A57.py']
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\x00')
        except (FileNotFoundError,PermissionError):continue
        if any(str(ROOT/n).encode() in args for n in watched):
            row=identity(int(proc.name));assert row is None or row['state']=='Z',('Owned prior process still live',row)
    with (ROOT/'formal_launch_once.lock').open('x') as f:f.write(json.dumps({'launcher_pid':os.getpid(),'utc':utc()}))
    env=os.environ.copy();env['PYTHONHASHSEED']='20261101';env['OMP_NUM_THREADS']='4'
    with (ROOT/'supervisor.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'supervise_A57.py')],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=identity(child.pid);assert row is not None
    receipt={'status':'LAUNCHED_SUPERVISOR_NOT_YET_TRAINING_PROOF','supervisor':row,'utc':utc(),
             'sequence':['B','M','Q'],'fresh_common_initialization':True,'post_training_queue':None}
    save(ROOT/'launch_receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
