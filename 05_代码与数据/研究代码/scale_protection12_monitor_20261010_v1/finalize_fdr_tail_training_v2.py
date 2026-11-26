"""Separate training and closeout controllers before any new training runs."""
from pathlib import Path
import ast
import json
import hashlib
import tarfile

ROOT=Path(__file__).resolve().parents[3]
CODE=ROOT/'英文小论文/code'
old=CODE/'fdr_tail12_20261011_v1'
new=CODE/'fdr_tail12_20261011_v2'
assert not new.exists();new.mkdir()
for p in old.iterdir():
    if p.is_file():
        (new/p.name).write_bytes(p.read_bytes())
s=(new/'supervise.py').read_text(encoding='utf8')
block="        from verify_candidate_reload import supervise as verify_saved_candidate\n        verify_saved_candidate(Path(a.bundle).resolve())\n        from collect_terminal import collect\n        collect(Path(a.bundle).resolve())"
assert s.count(block)==1
(new/'supervise.py').write_text(s.replace(block,''),encoding='utf8')
for name in ('runtime.py','prepare_training.py','training_protocol.json','verify_candidate_reload.py','closeout_protocol.json'):
    p=new/name;s=p.read_text(encoding='utf8')
    s=s.replace('fdr_tail12_20261011_v1','fdr_tail12_20261011_v2')
    s=s.replace('fdr_tail12_training_v1','fdr_tail12_training_v2')
    s=s.replace('fdr_tail12_closeout_v1','fdr_tail12_closeout_v2')
    if name=='runtime.py':
        s=s.replace("'verify_candidate_reload.py','collect_terminal.py','closeout_protocol.json')",
            "'verify_candidate_reload.py','collect_terminal.py','finish.py','closeout_protocol.json')")
    p.write_text(s,encoding='utf8')
finish='''"""Observe the owned training controller, then saved-EMA verification once."""
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
        assert not (screen/'candidate_reload_v1').exists()
        reload_saved(Path(a.bundle).resolve())
        collect(Path(a.bundle).resolve())
        write(out/'completion.json',dict(status='FDR_FIXED12_RELOAD_AND_COLLECTION_FINISHED',
            identity=receipt['identity'],training_commands=0,baseline_forwards=0,epochs_added=0,
            automatic_retry=False,completed_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except BaseException:
        write(out/'failure.json',dict(status='FDR_CLOSEOUT_FAILED_NO_RETRY',traceback=traceback.format_exc(),
            training_commands=0,baseline_forwards=0,epochs_added=0,automatic_retry=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True)
    p.add_argument('--controller-pid',type=int,required=True);p.add_argument('--controller-start-ticks',required=True)
    main(p.parse_args())
'''
(new/'finish.py').write_text(finish,encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cp=new/'closeout_protocol.json';v=json.loads(cp.read_text(encoding='utf8'))
v['files']={n:sha(new/n) for n in ('verify_candidate_reload.py','collect_terminal.py','tail_grid.py','finish.py')}
cp.write_text(json.dumps(v,indent=2),encoding='utf8')
for p in new.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
out=ROOT/'outputs/fdr_tail12_20261011_v2';out.mkdir(exist_ok=False)
receipt=dict(status='PRE_DISPATCH_CONTROLLER_REVIEW_REPAIRED_WITHOUT_TRAINING',
    original_package_v1_not_launched=True,original_training_epochs=0,
    issue='Inline saved-EMA verification requires training controller to stop; training and finishing now run in separate owned processes.',
    factor_or_architecture_or_training_budget_changed=False,source_packages_preserved=True)
with (out/'pre_dispatch_review.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
files={p.name:sha(p) for p in new.iterdir() if p.is_file()}
with (out/'package_freeze.json').open('x',encoding='utf8') as f:
    json.dump(dict(schema='fdr_tail12_package_freeze_v2',files=files,max_epochs=12,factor=2.,training_launches=0),f,indent=2)
t=out/'code_package_v2.tar.gz'
with tarfile.open(t,'x:gz') as a:
    for p in sorted(new.iterdir()):
        if p.is_file():a.add(p,arcname=p.name)
print(json.dumps(dict(status='ISOLATED_FIXED12_PACKAGE_FROZEN',files=len(files),bytes=t.stat().st_size,sha256=sha(t))))
