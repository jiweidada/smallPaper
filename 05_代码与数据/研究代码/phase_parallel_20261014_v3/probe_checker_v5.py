"""Observe only Python stdin preflight processes, without exposing startup argv."""
import json
import os
import pathlib
import sys

rows=[]
for path in pathlib.Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name)==os.getpid():
        continue
    try:
        args=(path/'cmdline').read_bytes().split(b'\0')
        if not args or args[0].decode(errors='replace')!=sys.executable or b'-u' not in args or b'-' not in args:
            continue
        status={k:v.strip() for line in (path/'status').read_text().splitlines() for k,_,v in [line.partition(':')]
                if k in ['Name','State','VmRSS','Threads']}
        descriptors=[]
        for descriptor in (path/'fd').iterdir():
            try:
                target=os.readlink(descriptor)
                if 'phase_mechanism_20261012' in target:
                    descriptors.append(target)
            except OSError:
                pass
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        rows.append({'pid':int(path.name),'start_ticks':int(stat[19]),'ppid':int(stat[1]),
                     'status':status,'wchan':(path/'wchan').read_text(),
                     'io':(path/'io').read_text(),'phase_file_descriptors':descriptors})
    except OSError:
        continue
print(json.dumps(rows))
