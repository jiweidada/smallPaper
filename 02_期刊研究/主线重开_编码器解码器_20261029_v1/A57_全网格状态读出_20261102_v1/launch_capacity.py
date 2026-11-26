from pathlib import Path
import json
import os
import subprocess
import sys

P=Path(__file__).resolve().parent


def main():
    assert P==Path('/root/uav_a57_20261102_v1')
    with (P/'capacity_launch_once.lock').open('x') as f:f.write(str(os.getpid()))
    assert not (P/'capacity_process_exit.json').exists()
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (P/'capacity_supervisor.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(P/'supervise_capacity.py')],cwd=P,env=env,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    fields=(Path('/proc')/str(proc.pid)/'stat').read_text().split(') ',1)[1].split()
    receipt={'status':'CAPACITY_LAUNCHED','supervisor_PID':proc.pid,'supervisor_start_ticks':int(fields[19]),
             'script':str(P/'supervise_capacity.py'),'post_capacity_queue':None,
             'whole_model_forward_cap':6,'capacity_optimizer_update_cap':6,'formal_training_launched':False}
    with (P/'capacity_launch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
