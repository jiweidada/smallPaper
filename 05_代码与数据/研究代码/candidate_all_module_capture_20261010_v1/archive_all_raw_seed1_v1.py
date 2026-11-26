import json,pathlib,zipfile,hashlib,datetime
r=pathlib.Path('/25085400120/DEIM/experiments/scale_protection12_20261010_v1')
d=r/'module_candidate_panel_seed1_v1'
c=json.loads((r/'module_candidate_completion_seed1_v1.json').read_text())
assert c['exit_code']==0 and c['status']=='CANDIDATE_MODULE_PANEL_COMPLETE' and not c['timed_out']
m=json.loads((d/'manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
z=r/'module_candidate_panel_seed1_raw_v1.zip'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
 for n,v in sorted(m['files'].items()):
  p=d/n
  assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
  out.write(p,'capture/'+n)
 out.write(d/'manifest.json','capture/manifest.json')
 for n in ['module_candidate_completion_seed1_v1.json','module_candidate_launch_seed1_v1.json','module_candidate_dispatch_seed1_v1.json','module_candidate_panel_seed1_v1.log','run_candidate_module_panel_once_v1.py','module_candidate_controller_seed1_v1.log']:
  out.write(r/n,'control/'+n)
 for p in (r/'module_capture_tools_v1').glob('*'):
  if p.is_file():out.write(p,'source/'+p.name)
metadata={'status':'ARCHIVED_ALL_CALLED_MODULE_RAW_OUTPUTS','bytes':z.stat().st_size,'sha256':sha(z),'raw_capture_bytes':sum(v['bytes'] for v in m['files'].values()),'manifest_sha256':sha(d/'manifest.json'),'actual_capture_exit_code':0,'completed_remote_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline_forwards':0,'epochs_added':0}
with (r/'module_candidate_panel_seed1_raw_v1_metadata.json').open('x') as f:json.dump(metadata,f,indent=2)
print(json.dumps(metadata),flush=True)

