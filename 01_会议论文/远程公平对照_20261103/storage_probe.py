from pathlib import Path
import json,subprocess,os
out={'filesystems':subprocess.check_output(['df','-hT','/tmp','/25085400120','/dev/shm'],text=True),'memory':Path('/proc/meminfo').read_text().splitlines()[:4]}
p=Path('/proc/555')
if p.exists():
 out['input_process_stat']=p.joinpath('stat').read_text();out['input_process_io']=p.joinpath('io').read_text()
 out['input_process_fds']={f.name:os.readlink(f) for f in p.joinpath('fd').iterdir()}
root=Path('/25085400120/mpsma_fair_20261103_v1')
out['input_receipt_exists']=(root/'input_verification.json').exists()
print(json.dumps(out))
