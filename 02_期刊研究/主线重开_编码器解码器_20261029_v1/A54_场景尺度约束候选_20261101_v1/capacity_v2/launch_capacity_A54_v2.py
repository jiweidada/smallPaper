from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path(__file__).resolve().parent
assert P==Path('/root/uav_a54_20261101_v1/capacity_v2')
with (P/'capacity_launch_once.lock').open('x') as f:f.write(str(os.getpid()))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Existing compute process; inspect instead of launching'
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(P.parent)
with (P/'capacity.log').open('xb') as log:
    child=subprocess.Popen(['/opt/conda/envs/mamba/bin/python','-u',str(P/'capacity_A54_v2.py')],cwd=str(P),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.1)
stat=Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ',1)[1].split()
result={'status':'STARTED','scope':'CAPACITY_ONLY','pid':child.pid,'start_ticks':int(stat[19]),'state':stat[0],
    'protocol_sha256':hashlib.sha256((P/'capacity_protocol_FROZEN_v2.json').read_bytes()).hexdigest(),'started_unix':time.time()}
(P/'capacity_launch_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
