from pathlib import Path
import json
import os
import subprocess
import sys

P=Path(__file__).resolve().parent


def save(name,x):
    with (P/name).open('x') as f:json.dump(x,f,indent=2)


def ticks(pid):
    return int((Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19])


def main():
    assert P==Path('/root/uav_a57_20261102_v1')
    with (P/'capacity_supervisor_once.lock').open('x') as f:f.write(str(os.getpid()))
    with (P/'capacity_worker.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,str(P/'capacity_native.py')],cwd=P,stdout=log,stderr=subprocess.STDOUT)
        identity={'supervisor_PID':os.getpid(),'supervisor_start_ticks':ticks(os.getpid()),
                  'worker_PID':proc.pid,'worker_start_ticks':ticks(proc.pid),
                  'worker_script':str(P/'capacity_native.py'),'post_capacity_queue':None}
        save('capacity_processes.json',identity)
        timed_out=False
        try:code=proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out=True
            assert ticks(proc.pid)==identity['worker_start_ticks']
            proc.terminate()
            try:code=proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                assert ticks(proc.pid)==identity['worker_start_ticks']
                proc.kill();code=proc.wait(timeout=15)
        save('capacity_process_exit.json',{'status':'CAPACITY_PROCESS_EXITED','child_exit_code':code,
                                           'timed_out':timed_out,**identity})
        return code


if __name__=='__main__':sys.exit(main())
