from pathlib import Path
import argparse,hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
def proc(identity):
    if not identity:return None
    p=Path('/proc',str(identity['pid']))
    if not p.exists():return {'live':False,'pid':identity['pid']}
    raw=(p/'stat').read_text().split(') ',1)[1].split()
    return {'live':raw[0] not in ('Z','X'),'state':raw[0],'pid':identity['pid'],
       'identity_matches':int(raw[19])==int(identity['start_ticks']),
       'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
p=argparse.ArgumentParser();p.add_argument('--verify-files',action='store_true');args=p.parse_args()
result={'observed_unix':time.time(),'capacity':read(ROOT/'capacity_result.json'),
 'capacity_process':proc(read(ROOT/'capacity_launch_identity.json')),
 'supervisor':read(ROOT/'supervisor_progress.json'),
 'supervisor_process':proc(read(ROOT/'supervisor_identity.json')),
 'formal_launch':read(ROOT/'formal_launch_identity.json'),'arms':{}}
for arm in ('M','Q'):
    folder=ROOT/('arm_'+arm)
    result['arms'][arm]={'progress':read(folder/'progress.json'),'result':read(folder/'result.json'),
      'initialization':read(folder/'initialization.json'),'exit':read(folder/'exit.json'),
      'process':proc(read(folder/'child_identity.json')),
      'metrics':{str(e):read(folder/('epoch%02d_EMA_metrics.json'%e)) for e in (4,8,12)},
      'stderr_tail':(folder/'stderr.log').read_text()[-1800:] if (folder/'stderr.log').exists() else None}
if args.verify_files:
    plan=read(ROOT/'protocol_FROZEN_v1.json')
    result['frozen_files_unchanged']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in plan['files_sha256'].items()}
result['capacity_stderr_tail']=(ROOT/'capacity_stderr.log').read_text()[-1800:] if (ROOT/'capacity_stderr.log').exists() else None
result['formal_stderr_tail']=(ROOT/'formal_stderr.log').read_text()[-1800:] if (ROOT/'formal_stderr.log').exists() else None
print(json.dumps(result))
