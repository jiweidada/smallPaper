from pathlib import Path
import hashlib
import json
import subprocess

def run(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=35)
        return {"exit_code":p.returncode,"stdout":p.stdout[-15000:],"stderr":p.stderr[-2000:]}
    except Exception as e:return {"error":type(e).__name__,"message":str(e)}

root=Path('/25085400120') if Path('/25085400120').is_mount() else Path('/24085403014')
folders=[root/'deim_migration_runtime/envs',root/'mpsma_fair_20261103_v1/secondary_transfer',root/'mpsma_fair_20261103_v1/preflight',Path('/tmp/mpsma_module_ablation_20261107_data'),Path('/tmp/mpsma_fair_20261103_data')]
listing={str(p):[{"name":c.name,"directory":c.is_dir(),"bytes":c.stat().st_size if c.is_file() else None} for c in sorted(p.iterdir())][:30] for p in folders if p.is_dir()}
candidates=[root/'envs/deim/bin/python',root/'deim_migration_runtime/envs/deim/bin/python']
envs=[]
for p in candidates:
    if not p.is_file():continue
    code="import torch,torchvision,importlib.metadata as m,json,sys;print(json.dumps({'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'packages':{d.metadata['Name']:d.version for d in m.distributions()},'cuda_available':torch.cuda.is_available()}))"
    envs.append({"path":str(p),"result":run([str(p),'-c',code])})
print(json.dumps({"root":str(root),"listings":listing,"environments":envs,"quota":run(['quota','-s'])}))
