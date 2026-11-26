"""Dispatch one isolated fixed12 trainer and its separate owned finisher."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def proc(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().split()
    return dict(pid=pid,state=fields[2],start_ticks=fields[21])


def main(a):
    root=Path(a.prefix)/'DEIM/experiments/fdr_tail12_20261011_v2'
    code=root/'code';bundle=root/f'seed{a.seed}_endpoint2/bundle';screen=bundle.parent
    sys.path.insert(0,str(code))
    from runtime import verify_bundle,read,write,sha
    receipt,plan,paths=verify_bundle(bundle)
    assert plan['seed']==a.seed and plan['endpoint_factor']==2.0
    assert not (screen/'dispatch.lock').exists() and not (screen/'one_shot_launch.lock').exists()
    assert not paths['output'].exists()
    original=Path(a.prefix)/'DEIM/experiments/scale_protection12_20261010_v1'
    from verify_candidate_reload import same_live
    for arm in ('target_protect','random_protect'):
        prior=original/f'seed{a.seed}_{arm}'
        for file,pid,tick in (('controller_start.json','pid','start_ticks'),
                              ('candidate_start.json','child_pid','child_start_ticks')):
            assert not same_live(read(prior/file),pid,tick)
        complete=read(original/f'post_training_seed{a.seed}_{arm}_v1/completion.json')
        assert complete['status']=='POST_TRAINING_RELOAD_AND_COLLECTION_FINISHED'
        assert complete['reload_exit_code']==complete['collection_exit_code']==0
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert uuid==read(code/'training_protocol.json')['GPU_uuid'][str(a.seed)] and not apps
    with (screen/'dispatch.lock').open('x') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    source=Path(__file__).read_bytes()
    with (screen/'dispatch_source.py').open('xb') as f:f.write(source)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-B',str(code/'supervise.py'),'--bundle',str(bundle)]
    with (screen/'controller.log').open('xb') as log:
        controller=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                    start_new_session=True,cwd=root,env=env)
    record=proc(controller.pid)
    finish_command=[sys.executable,'-B',str(code/'finish.py'),'--bundle',str(bundle),
        '--controller-pid',str(controller.pid),'--controller-start-ticks',record['start_ticks']]
    with (screen/'finisher.log').open('xb') as log:
        finisher=subprocess.Popen(finish_command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True,cwd=root,env=env)
    result=dict(status='ISOLATED_FDR_FIXED12_TRAIN_AND_FINISH_DISPATCHED',seed=a.seed,
        controller=record,finisher=proc(finisher.pid),training_command=command,finish_command=finish_command,
        GPU_uuid=uuid,GPU_idle_at_dispatch=True,identity=receipt['identity'],
        dispatch_source_sha256=hashlib.sha256(source).hexdigest(),baseline_training_commands=0,
        max_epochs=12,endpoint='same-seed native epoch12 EMA',automatic_retry=False,
        previous_experiments_not_restarted=True,dispatched_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(screen/'dispatch_receipt.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    main(p.parse_args())
