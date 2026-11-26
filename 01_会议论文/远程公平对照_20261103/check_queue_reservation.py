"""Synthetic test of the frozen supervisor's exclusive per-arm log guard.

No model, dataset, training code or real run directory is executed or changed.
"""
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile
ROOT = Path('/25085400120/mpsma_fair_20261103_v1')
original = ROOT/'supervise.py'
assert hashlib.sha256(original.read_bytes()).hexdigest() == next(
    f['sha256'] for f in json.loads((ROOT/'execution_protocol.json').read_text())['frozen_files'] if f['path']=='supervise.py')
folder = Path(tempfile.mkdtemp(prefix='synthetic_queue_reservation_',dir=ROOT))
(folder/'supervise.py').write_bytes(original.read_bytes())
protocol={'authorized':True,'arms':['BASE','FULL'],'epochs':320,
          'runtime':{'packages':{n:'synthetic' for n in ['torch','torchvision','numpy','Pillow','PyYAML','opencv-python','scipy','faster-coco-eval','timm']}},
          'budget':{'hard_timeout_seconds_per_arm':604800}}
(folder/'execution_protocol.json').write_text(json.dumps(protocol))
(folder/'FULL.train.log').write_text('SYNTHETIC RESERVATION: never execute this arm.\n')
(folder/'fair_runner.py').write_text('''from pathlib import Path
import json,sys,time
r=Path(__file__).resolve().parent
arm=sys.argv[-1]
(r/(arm+'.entered')).write_text('SYNTHETIC ONLY; zero optimizer updates')
time.sleep(1)
p=r/'runs'/arm;p.mkdir(parents=True)
(p/'receipt.json').write_text(json.dumps({'status':'COMPLETED','completed_epochs':320,'synthetic':True,'optimizer_updates':0}))
''')
command=[sys.executable,'-c',"import importlib.metadata as m,runpy; m.version=lambda name:'synthetic'; runpy.run_path("+repr(str(folder/'supervise.py'))+")"]
result=subprocess.run(command,capture_output=True,text=True,timeout=40)
queue=json.loads((folder/'queue.json').read_text())
assert result.returncode==1
assert len(queue['runs'])==1 and queue['runs'][0]['arm']=='BASE'
assert queue['runs'][0]['status']=='COMPLETED' and queue['runs'][0]['exit_code']==0
assert (folder/'BASE.entered').exists() and not (folder/'FULL.entered').exists()
assert not (folder/'runs/FULL').exists()
assert 'FileExistsError' in result.stderr and 'FULL.train.log' in result.stderr
report={'status':'SYNTHETIC_RESERVATION_TEST_PASSED','model_executions':0,'optimizer_updates':0,
        'supervisor_source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'base_exit_retained':True,'full_process_not_created':True,'queue_after_base':queue['status'],
        'expected_scheduler_exit':1,'expected_scheduler_exception':'FileExistsError on reserved FULL.train.log',
        'test_root':str(folder),'limitation':'The original queue remains RUNNING; its per-BASE result stays authoritative. A separate split-host ledger is required.'}
(ROOT/'queue_reservation_test.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
