import pathlib,json,os
root=pathlib.Path(CONFIG['root'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(entry):
 p=pathlib.Path('/proc')/str(entry['pid'])
 if not p.is_dir():return {'alive':False}
 st=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'pid':entry['pid'],'state':st[0],'ticks':st[19],'identity_matches':str(entry['start_ticks'])==st[19],'wchan':(p/'wchan').read_text(),'status':[(s) for s in (p/'status').read_text().splitlines() if s.startswith(('Threads:','VmRSS:','SigPnd:','ShdPnd:'))]}
p=root/'pause_user_20261019_v1';receipt=read(p/'receipt.json')
result={'account':CONFIG['account'],'pause_receipt':receipt,'pause_failure':read(p/'failure.json'),'watcher_log_tail':(root/'pause_control_20261019_v1/watcher.log').read_text()[-6000:],'jobs':[],'controller':proc(receipt['controller'])}
for j in receipt['jobs']:
 log=root/'outputs'/j['job']/'log.txt';rows=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
 result['jobs'].append({'job':j['job'],'process':proc(j['identity']),'last_complete_epoch':rows[-1]['epoch'],'console_tail':(root/'logs'/(j['job']+'.log')).read_text()[-1000:]})
print(json.dumps(result),flush=True)
