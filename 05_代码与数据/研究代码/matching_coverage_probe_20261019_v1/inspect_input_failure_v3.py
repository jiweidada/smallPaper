import pathlib,json
root=pathlib.Path(CONFIG['storage'])/'DEIM/experiments/matching_coverage_probe_20261019_v1'
result={'account':CONFIG['account'],'logs':{},'input_files':[]}
for rel in ['task_records/inputs.log','inputs_resume_v3_supervisor.log']:
 p=root/rel
 if p.is_file():result['logs'][rel]=p.read_text(errors='replace')[-6000:]
if (root/'inputs').is_dir():result['input_files']=[p.name for p in (root/'inputs').iterdir()]
print(json.dumps(result),flush=True)
