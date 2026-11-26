"""Read-only inventory after a validation SSH disconnect."""
import datetime,json,os
from pathlib import Path
root=Path(CONFIG['root']);found=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        args=[a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
        if any(a==CONFIG['python'] or a.startswith(str(root)+'/') for a in args):
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            found.append({'pid':int(proc.name),'ppid':fields[1],'state':fields[0],'start_ticks':fields[19],
                          'utime':fields[11],'stime':fields[12],'command':args})
    except (FileNotFoundError,PermissionError):pass
print(json.dumps({'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':found}),flush=True)
