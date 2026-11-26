from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text()) if p.exists() else None
def process(i):
    if not i:return None
    p=Path('/proc',str(i['pid']),'stat')
    try:s=p.read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:return {'pid':i['pid'],'live':False,'exists':False}
    same=int(s[19])==i['start_ticks'];return {'pid':i['pid'],'live':same and s[0] not in ['Z','X'],'identity_matches':same,'state':s[0],'start_ticks':int(s[19])}
plan=load(P/'protocol_FROZEN_v1.json')
result={'observed_server_unix':time.time(),'observation_only':True,'supervisor':load(P/'supervisor_progress.json'),
    'supervisor_process':process(load(P/'supervisor_identity.json')),'three_arm_result':load(P/'three_arm_result.json'),
    'frozen_files_unchanged':{n:hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in plan['files_sha256'].items()},'arms':{}}
for arm in ['M','Q']:
    a=P/('arm_'+arm);result['arms'][arm]={'progress':load(a/'progress.json'),'result':load(a/'result.json'),'initialization':load(a/'initialization.json'),
        'process':process(load(a/'child_identity.json')),'exit':load(a/'exit.json'),
        'metrics':{str(e):load(a/('epoch%02d_EMA_metrics.json'%e)) for e in [4,8,12]},
        'stderr_tail':(a/'stderr.log').read_text(errors='replace')[-2200:] if (a/'stderr.log').exists() else ''}
if result['supervisor'] is None or result['supervisor'].get('status','').startswith('FAILED'):
    result['formal_stderr_tail']=(P/'formal_stderr.log').read_text(errors='replace')[-3000:] if (P/'formal_stderr.log').exists() else ''
print(json.dumps(result))
