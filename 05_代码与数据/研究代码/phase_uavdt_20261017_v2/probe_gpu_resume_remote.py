"""Read-only checkpoint, recipe and process preflight after GPU reallocation."""
import datetime, hashlib, json, os, subprocess
from pathlib import Path
import torch

root = Path(CONFIG['root'])
def read(p): return json.loads(p.read_text()) if p.is_file() else None
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
plan=read(root/'plan.json')
result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'account':CONFIG['account'],
        'plan':plan,'queue':read(root/'queue_state.json'),'launch':read(root/'launch_receipt.json'),
        'lock':read(root/'queue.lock'),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'jobs':[],'processes':[],'frozen_mismatches':[]}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        command=[a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
        if any(str(root/'code') in a for a in command):
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            result['processes'].append({'pid':int(proc.name),'state':stat[0],'start_ticks':stat[19],'command':command})
    except (FileNotFoundError,PermissionError): pass
for item in read(root/'frozen_manifest.json')['files']:
    p=root/item['relative']
    if not p.is_file() or sha(p)!=item['sha256']: result['frozen_mismatches'].append(item['relative'])
result['initialization_sha256']=sha(Path(plan['initialization']))
result['data_exists']=Path(plan['data_root']).is_dir()
result['capacity']=read(root/'capacity_receipt.json')
for arm in plan['jobs']:
    job=f'uavdt_{arm}_full96_seed{plan["seed"]}'; out=root/'outputs'/job
    rows=[json.loads(s) for s in (out/'log.txt').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if 'test_uavdt_eval_bbox' in r]
    cp=out/'last.pth'; s=torch.load(cp,map_location='cpu',weights_only=False)
    optim=s.get('optimizer',{}); scaler=s.get('scaler',{})
    result['jobs'].append({'job':job,'arm':arm,'status':read(root/'jobs'/(job+'.json')),
        'logged_epochs':[r['epoch'] for r in rows], 'latest_metrics':rows[-1].get('test_uavdt_eval_bbox') if rows else None,
        'checkpoint':str(cp),'checkpoint_bytes':cp.stat().st_size,'checkpoint_sha256':sha(cp),
        'last_epoch':s.get('last_epoch'),'state_keys':sorted(s),
        'optimizer_state_entries':len(optim.get('state',{})),
        'optimizer_groups':len(optim.get('param_groups',[])),
        'optimizer_lr':[g.get('lr') for g in optim.get('param_groups',[])],
        'scaler':scaler,'ema_updates':s.get('ema',{}).get('updates'),
        'scheduler':s.get('lr_scheduler'),'warmup':s.get('lr_warmup_scheduler'),
        'available_checkpoints':[{ 'name':p.name,'bytes':p.stat().st_size} for p in sorted(out.glob('*.pth'))]})
    del s
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(result,default=str),flush=True)
