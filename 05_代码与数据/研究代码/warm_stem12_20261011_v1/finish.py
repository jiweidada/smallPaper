"""Observe the owned training controller, then saved-EMA verification once."""
import argparse
import datetime
import os
from pathlib import Path
import time
import traceback
from runtime import verify_bundle,read,write,sha
from verify_candidate_reload import same_live,supervise as reload_saved
from collect_terminal import collect

def main(a):
    receipt,plan,paths=verify_bundle(a.bundle)
    screen=paths['screen_root']; out=screen/'post_training'
    out.mkdir(exist_ok=False)
    owned=dict(pid=a.controller_pid,start_ticks=a.controller_start_ticks)
    assert same_live(owned,'pid','start_ticks'), 'Owned controller must be observed live at dispatch'
    write(out/'start.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        training_controller=owned,identity=receipt['identity'],source_sha256=sha(__file__),
        automatic_retry=False,training_commands=0,baseline_forwards=0,epochs_added=0))
    started=time.monotonic()
    try:
        while same_live(owned,'pid','start_ticks'):
            (out/'live.json').write_text(__import__('json').dumps(dict(phase='waiting_owned_training',
                elapsed_seconds=time.monotonic()-started,observed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
            if time.monotonic()-started>=64800:
                raise TimeoutError('Observation budget expired; trainer is neither stopped nor restarted')
            time.sleep(15)
        assert read(screen/'controller_completion.json')['status']=='COMPLETE_FIXED12'
        assert read(screen/'candidate_completion.json')['exit_code']==0
        done=read(paths['output']/'training_completion.json')
        assert done['completed_epochs']==12 and done['entered_epochs']==list(range(12))
        assert done['identity']==receipt['identity']
        from dispatch import wait_old_closeouts_and_idle
        wait_old_closeouts_and_idle(screen,plan,paths)
        assert not (screen/'candidate_reload_v1').exists()
        reload_saved(Path(a.bundle).resolve())
        collect(Path(a.bundle).resolve())
        write(out/'completion.json',dict(status='STEM_FIXED12_RELOAD_AND_COLLECTION_FINISHED',
            identity=receipt['identity'],training_commands=0,baseline_forwards=0,epochs_added=0,
            automatic_retry=False,completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except BaseException:
        write(out/'failure.json',dict(status='STEM_CLOSEOUT_FAILED_NO_RETRY',traceback=traceback.format_exc(),
            training_commands=0,baseline_forwards=0,epochs_added=0,automatic_retry=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True)
    p.add_argument('--controller-pid',type=int,required=True);p.add_argument('--controller-start-ticks',required=True)
    main(p.parse_args())
