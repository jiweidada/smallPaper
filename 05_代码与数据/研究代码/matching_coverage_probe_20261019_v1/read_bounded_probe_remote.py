import json,pathlib,subprocess
root=pathlib.Path(CONFIG['storage'])/'DEIM/experiments/matching_coverage_probe_20261019_v1'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def tail(p,n=2000):return p.read_text(errors='replace')[-n:] if p.is_file() else None
tasks={}
for name in ['CPU_preflight_v2','CPU_extension_preflight_v2','inputs','probe_seed1','probe_seed2']:
    v=read(root/'task_records'/(name+'.json'))
    if v:
        v['console_tail']=tail(root/'task_records'/(name+'.log'),3000 if v['state']=='failed' else 400)
        tasks[name]=v
seed=1 if CONFIG['account']=='A' else 2
out=root/f'results_seed{seed}'
old=pathlib.Path(CONFIG['root'])/'pause_user_20261019_v1'
result={'account':CONFIG['account'],'tasks':tasks,'preflight':read(root/f'preflight_seed{seed}.json'),
 'extension_preflight':read(root/f'extension_preflight_seed{seed}.json'),
 'input_progress':read(root/'inputs/progress.json'),'input_complete':read(root/'inputs/complete.json'),
 'progress':read(out/'progress.json'),'complete':read(out/'complete.json'),'supplement_complete':read(out/'supplement_complete.json'),'failure':read(out/'failure.json'),
 'old_release':read(old/'release_receipt.json'),
 'GPU_compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)}
print(json.dumps(result),flush=True)
