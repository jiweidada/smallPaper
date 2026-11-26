from pathlib import Path
import json,time
P=Path(__file__).resolve().parent
launch=json.loads((P/'capacity_launch_receipt.json').read_text());pid=launch['pid'];proc=Path(f'/proc/{pid}')
identity={'pid':pid,'expected_start_ticks':launch['start_ticks'],'exists':proc.exists()}
if proc.exists():
    try:
        s=(proc/'stat').read_text().rsplit(') ',1)[1].split();identity.update(state=s[0],actual_start_ticks=int(s[19]),identity_matches=int(s[19])==launch['start_ticks'])
        identity['live']=identity['identity_matches'] and s[0] not in ['Z','X']
    except FileNotFoundError:identity.update(exists=False,live=False)
else:identity['live']=False
result=json.loads((P/'capacity_result.json').read_text()) if (P/'capacity_result.json').exists() else None
progress=json.loads((P/'capacity_progress.json').read_text()) if (P/'capacity_progress.json').exists() else None
data={'observed_server_unix':time.time(),'observation_only':True,'identity':identity,'result':result,'progress':progress,
    'log_tail':(P/'capacity.log').read_text(errors='replace')[-5000:] if (P/'capacity.log').exists() else ''}
print(json.dumps(data))
