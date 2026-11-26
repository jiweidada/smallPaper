from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/uav_a52_20261101_v1')
assert sys.executable=='/opt/conda/envs/mamba/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
planpath=ROOT/'protocol_FROZEN_v1.json'
assert sha(planpath)=='c13b61bdbc09d81a051108f7a28516f14ae3a3521a81d8247574129ca413d027'
plan=json.loads(planpath.read_text())
assert all(sha(ROOT/n)==h for n,h in plan['files_sha256'].items())
capacity=json.loads((ROOT/'capacity_v2/capacity_result.json').read_text())
assert capacity['status']=='PASS' and capacity['whole_model_forwards']==9 and capacity['capacity_only_optimizer_updates']==6
for folder in [ROOT,ROOT/'capacity_v2']:
    identity=json.loads((folder/'capacity_launch_identity.json').read_text())
    p=Path('/proc',str(identity['pid']),'stat')
    if p.exists():
        fields=p.read_text().split(') ',1)[1].split()
        assert int(fields[19])!=int(identity['start_ticks']) or fields[0] in ('Z','X')
assert not (ROOT/'formal_launch_identity.json').exists()
with (ROOT/'formal_launch_once.lock').open('x') as f:f.write(str(os.getpid()))
with (ROOT/'formal_stdout.log').open('xb') as out,(ROOT/'formal_stderr.log').open('xb') as err:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'supervise_A52.py')],cwd=str(ROOT),stdout=out,stderr=err,start_new_session=True)
ticks=int(Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19])
identity={'pid':child.pid,'start_ticks':ticks,'stage':'FORMAL_M12_THEN_Q12','script':'supervise_A52.py',
 'launch_server_unix':time.time(),'protocol_sha256':sha(planpath),'no_extra_queue_after_Q':True,'no_auto_retry':True}
(ROOT/'formal_launch_identity.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity),flush=True)

