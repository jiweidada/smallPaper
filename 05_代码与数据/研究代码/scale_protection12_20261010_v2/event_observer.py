"""Write each worker's own Mosaic decisions without changing its RNG."""
from pathlib import Path
import json
import os
import torch


def append_event(folder, row):
    info = torch.utils.data.get_worker_info()
    value = dict(row, worker_id=-1 if info is None else info.id, process_pid=os.getpid(),
                 scope='Mosaic boundary target counts before native sanitation/MixUp; not loss supervision counts')
    p = Path(folder)/f'worker_pid{os.getpid()}.jsonl'
    # Each PID has its own file; no shared buffered stream is inherited by workers.
    with p.open('a', encoding='utf8') as f:
        f.write(json.dumps(value)+'\n')
