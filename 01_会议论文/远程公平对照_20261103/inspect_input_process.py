from pathlib import Path
import json,os
p=Path('/proc/555'); out={'pid':555,'exists':p.exists()}
if p.exists():
    out['cmdline']=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    out['stat']=p.joinpath('stat').read_text()
    out['io']=p.joinpath('io').read_text()
    out['wchan']=p.joinpath('wchan').read_text()
    out['fds']={f.name:os.readlink(f) for f in p.joinpath('fd').iterdir()}
    out['fdinfo']={f.name:f.read_text() for f in p.joinpath('fdinfo').iterdir() if f.name not in ['0','1','2']}
root=Path('/25085400120/mpsma_fair_20261103_v1')
out['receipt_exists']=(root/'input_verification.json').exists()
if out['receipt_exists']: out['receipt']=json.loads((root/'input_verification.json').read_text())
print(json.dumps(out))
