from pathlib import Path
import argparse,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['capacity','formal'],required=True);stage=p.parse_args().stage
if stage=='formal':
    plan=json.loads((ROOT/'protocol_FROZEN_v1.json').read_text())
    cap=json.loads((ROOT/'capacity_result.json').read_text())
    assert cap['status']=='PASS' and cap['protocol_sha256']==plan['capacity_protocol_sha256']
    for name,h in plan['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
with (ROOT/(stage+'_launch_once.lock')).open('x') as f:f.write(str(os.getpid()))
script='capacity_A51.py' if stage=='capacity' else 'supervise_A51.py'
with (ROOT/(stage+'_stdout.log')).open('xb') as out,(ROOT/(stage+'_stderr.log')).open('xb') as err:
    child=subprocess.Popen(['/opt/conda/envs/mamba/bin/python','-u',str(ROOT/script)],cwd=ROOT,
       stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
stat=Path('/proc',str(child.pid),'stat')
info={'stage':stage,'pid':child.pid,'start_ticks':stat.read_text().split(') ',1)[1].split()[19],
      'script':script,'no_auto_retry':True,'no_extra_queue':True}
(ROOT/(stage+'_launch_identity.json')).write_text(json.dumps(info,indent=2))
print(json.dumps(info))
