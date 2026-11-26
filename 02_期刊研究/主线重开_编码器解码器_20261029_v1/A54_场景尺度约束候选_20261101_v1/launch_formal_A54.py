from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
assert ROOT==Path('/root/uav_a54_20261101_v1') and sys.executable=='/opt/conda/envs/mamba/bin/python'
p=argparse.ArgumentParser();p.add_argument('--protocol-sha256',required=True);args=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
planpath=ROOT/'protocol_FROZEN_v1.json';assert sha(planpath)==args.protocol_sha256
plan=json.loads(planpath.read_text());assert all(sha(ROOT/n)==h for n,h in plan['files_sha256'].items())
assert all(sha(Path(n))==h for n,h in plan['external_sha256'].items())
capacity=json.loads((ROOT/'capacity_v2/capacity_result.json').read_text());assert capacity['status']=='PASS' and capacity['cumulative_A54_capacity_forwards']==9 and capacity['capacity_only_optimizer_updates']==6
for folder in [ROOT,ROOT/'capacity_v2']:
    identity=json.loads((folder/'capacity_launch_receipt.json').read_text());proc=Path('/proc',str(identity['pid']),'stat')
    if proc.exists():
        st=proc.read_text().rsplit(') ',1)[1].split();assert int(st[19])!=identity['start_ticks'] or st[0] in ['Z','X']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Compute process exists; inspect first'
with (ROOT/'formal_launch_once.lock').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(ROOT)
with (ROOT/'formal_stdout.log').open('xb') as out,(ROOT/'formal_stderr.log').open('xb') as err:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'supervise_A54.py')],cwd=str(ROOT),env=env,stdout=out,stderr=err,start_new_session=True)
ticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()[19])
identity={'pid':child.pid,'start_ticks':ticks,'stage':'FORMAL_M12_THEN_Q12','script':'supervise_A54.py','launch_server_unix':time.time(),
    'protocol_sha256':sha(planpath),'no_extra_queue_after_Q':True,'no_auto_retry':True}
(ROOT/'formal_launch_identity.json').write_text(json.dumps(identity,indent=2));print(json.dumps(identity))
