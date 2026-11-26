"""Read only recognized phase processes; never dump unrelated startup commands."""
import json
import os
import pathlib
import time

rows=[]
for path in pathlib.Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name)==os.getpid():
        continue
    try:
        args=[x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]
        names={pathlib.PurePosixPath(x).name for x in args}
        if not any('phase_mechanism_20261012' in x for x in args) or not names & {
            'resume_parallel_queue_v5.py','train.py','train_round2.py'}:
            continue
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        info={k:v.strip() for line in (path/'status').read_text().splitlines() for k,_,v in [line.partition(':')]
              if k in ['State','VmRSS','Threads']}
        descriptors=[]
        for fd in (path/'fd').iterdir():
            try:
                value=os.readlink(fd)
                if value.startswith('/25085400120') or value.startswith('/24085403014'):
                    descriptors.append(value)
            except OSError:
                pass
        rows.append({'pid':int(path.name),'ppid':int(stat[1]),'start_ticks':int(stat[19]),'status':info,
                     'wchan':(path/'wchan').read_text(),'io':(path/'io').read_text(),
                     'file_descriptors':descriptors,'command':args})
    except OSError:
        continue
print(json.dumps(rows))
