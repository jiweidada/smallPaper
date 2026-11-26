import pathlib,json
root=pathlib.Path(CONFIG['storage'])/'DEIM/experiments/matching_coverage_probe_20261019_v1'
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def tail(p):return p.read_text(errors='replace')[-4500:] if p.is_file() else None
seed=1 if CONFIG['account']=='A' else 2
print(json.dumps({'account':CONFIG['account'],'task':read(root/'task_records/CPU_preflight_v2.json'),'preflight':read(root/f'preflight_seed{seed}.json'),'log_tail':tail(root/'task_records/CPU_preflight_v2.log'),'extension_task':read(root/'task_records/CPU_extension_preflight_v2.json'),'extension_preflight':read(root/f'extension_preflight_seed{seed}.json'),'extension_log_tail':tail(root/'task_records/CPU_extension_preflight_v2.log'),'inputs':read(root/'inputs/complete.json')}),flush=True)
