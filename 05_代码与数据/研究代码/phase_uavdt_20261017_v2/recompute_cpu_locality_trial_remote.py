"""Retrospective read-only parsing of preserved trial windows, including ANSI-colored logs."""
import datetime,json,re
from pathlib import Path
root=Path(CONFIG['root']);art=root/'artifacts/performance_CPU_locality_v1'
original=json.loads((art/'original_affinities.json').read_text());result={'account':CONFIG['account'],'stages':{},'restore_receipt':json.loads((art/'restore_receipt.json').read_text()),'comparison':{}}
for label in ['baseline','GPU_local_affinity','restored_baseline']:
 stage=json.loads((art/(label+'.json')).read_text());start=stage['start_remote_unix'];end=stage['end_remote_unix'];jobs={}
 for owner in original['owners']:
  job=owner['job'];text=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',(root/'logs'/(job+'.log')).read_text(errors='replace'));selected=[]
  for line in text.splitlines():
   m=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?Epoch:\s*(\d+)/96\s+\[\s*(\d+)/(\d+)\].*?time:\s*([\d.]+)\s+data:\s*([\d.]+)',line)
   if m:
    timestamp=datetime.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()
    if start+1<=timestamp<=end-1:selected.append({'time':timestamp,'epoch':int(m[2]),'batch':int(m[3]),'seconds_per_step':float(m[5]),'data_seconds':float(m[6])})
  steps=0;seconds=0.
  for b,a in zip(selected,selected[1:]):
   if a['epoch']==b['epoch'] and a['batch']>b['batch'] and a['time']>b['time']:steps+=a['batch']-b['batch'];seconds+=a['time']-b['time']
  jobs[job]={'records':selected,'valid_steps':steps,'valid_seconds':seconds,'images_per_second':4*steps/seconds if seconds else None,
   'epochs_in_window':sorted(set(r['epoch'] for r in selected))}
 result['stages'][label]={'start':start,'end':end,'jobs':jobs}
for owner in original['owners']:
 job=owner['job'];b,p,a=[result['stages'][s]['jobs'][job]['images_per_second'] for s in ['baseline','GPU_local_affinity','restored_baseline']]
 result['comparison'][job]={'before_images_s':b,'pinned_images_s':p,'restored_images_s':a,
 'gain_vs_combined_baseline_percent':100*(p/((a+b)/2)-1) if a and b and p else None}
result['parser_correction']='Native logs have ANSI prefixes. Retrospectively strip colors and use the preserved original stage wall-clock windows. No additional scheduling trial or training was run.'
result['limitations']='45/90/45 second sequential windows; epoch/evaluation transitions change co-runner load. Not a randomized benchmark or proof of a causal gain.'
print(json.dumps(result),flush=True)
