"""Read-only process loading diagnostics, no signals or training changes."""
import datetime,json,os,subprocess
from pathlib import Path
root=Path(CONFIG['root'])
def tail(path,n=3500):
    if not path.is_file():return ''
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-n));return f.read().decode(errors='replace')
result={'checked_remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[],
        'controller_tail':tail(root/'logs/controller_recovery_20261018.log'),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
for path in (root/'jobs').glob('*.json'):
    record=json.loads(path.read_text());proc=Path('/proc')/str(record['pid'])
    item={'job':record['job'],'pid':record['pid'],'console_tail':tail(root/'logs'/(record['job']+'.log'))}
    if proc.is_dir():
        for name in ['wchan','io','stat','status']:
            try:
                text=(proc/name).read_text()
                if name=='status':text='\n'.join(s for s in text.splitlines() if s.startswith(('State:','VmRSS:','VmSize:','Threads:','voluntary_ctxt_switches:','nonvoluntary_ctxt_switches:')))
                item[name]=text
            except PermissionError:item[name]='unavailable'
        item['open_files']=[]
        for fd in (proc/'fd').iterdir():
            try:item['open_files'].append(os.readlink(fd))
            except FileNotFoundError:pass
    result['jobs'].append(item)
print(json.dumps(result),flush=True)
