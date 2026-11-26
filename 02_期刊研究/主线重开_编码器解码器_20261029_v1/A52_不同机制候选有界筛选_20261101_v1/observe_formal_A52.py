from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parent
CAP=ROOT/'capacity_v2'
def load(p):return json.loads(p.read_text()) if p.exists() else None
def process(identity):
    if not identity:return None
    pid=identity['pid'];folder=Path('/proc',str(pid))
    if not folder.exists():return {'pid':pid,'live':False}
    fields=(folder/'stat').read_text().split(') ',1)[1].split()
    same=int(fields[19])==int(identity['start_ticks'])
    return {'pid':pid,'live':same and fields[0] not in ('Z','X'),'identity_matches':same,'state':fields[0]}
result={'observed_server_unix':time.time(),
 'capacity_identity':load(CAP/'capacity_launch_identity.json'),
 'capacity_progress':load(CAP/'capacity_progress.json'),
 'capacity_result':load(CAP/'capacity_result.json'),
 'capacity_stderr_tail':(CAP/'capacity_stderr.log').read_text(errors='replace')[-4000:] if (CAP/'capacity_stderr.log').exists() else ''}
result['capacity_process']=process(result['capacity_identity'])
result['prior_v1_guard_failure']=load(ROOT/'capacity_result.json')
plan=load(ROOT/'protocol_FROZEN_v1.json') or load(ROOT/'capacity_protocol_FROZEN_v1.json')
result['local_frozen_files_unchanged']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in plan['files_sha256'].items()}
result['supervisor']=load(ROOT/'supervisor_progress.json')
result['supervisor_process']=process(load(ROOT/'supervisor_identity.json'))
result['three_arm_result']=load(ROOT/'three_arm_result.json')
result['arms']={}
for a in ['M','Q']:
    folder=ROOT/('arm_'+a)
    result['arms'][a]={'progress':load(folder/'progress.json'),'result':load(folder/'result.json'),
     'initialization':load(folder/'initialization.json'),'exit':load(folder/'exit.json'),
     'process':process(load(folder/'child_identity.json')),
     'metrics':{str(e):load(folder/('epoch%02d_EMA_metrics.json'%e)) for e in [4,8,12]}}
print(json.dumps(result))

