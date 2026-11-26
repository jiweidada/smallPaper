"""One bounded integration child; does not launch training."""
import os
import signal
import subprocess
import sys
import traceback
from supervise_A57 import ROOT,identity,owned_alive,save,utc


def main():
    with (ROOT/'eval_integration_supervisor_once.lock').open('x') as f:f.write(str(os.getpid()))
    with (ROOT/'eval_integration.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'eval_integration_A57.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=identity(child.pid);assert row is not None
    save(ROOT/'eval_integration_launch.json',{'supervisor':identity(os.getpid()),'worker':row,'utc':utc()})
    try:
        code=child.wait(timeout=600)
        save(ROOT/'eval_integration_exit.json',{'exit_code':code,'worker':row,'timed_out':False,
                                              'post_integration_queue':None,'utc':utc()})
        return code
    except BaseException:
        actions=[]
        if child.poll() is None and owned_alive(row):
            os.killpg(child.pid,signal.SIGTERM);actions.append('SIGTERM_OWNED_GROUP')
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if owned_alive(row):os.killpg(child.pid,signal.SIGKILL);actions.append('SIGKILL_OWNED_GROUP')
                child.wait(timeout=10)
        save(ROOT/'eval_integration_exit.json',{'exit_code':child.returncode,'worker':row,'timed_out':True,
            'traceback':traceback.format_exc(),'actions':actions,'post_integration_queue':None,'utc':utc()})
        raise


if __name__=='__main__':sys.exit(main())
