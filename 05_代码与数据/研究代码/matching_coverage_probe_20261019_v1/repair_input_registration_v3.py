import datetime,hashlib,json
from control_probe import ctl,ROOTS,LOCAL,HERE
pw=json.loads(ctl.scheduled_monitor_v4.credentials_bytes());c=None
try:
 cfg=ctl.ACCOUNTS['A'];c=ctl.connect(cfg['port'],pw);sf=c.open_sftp();root=ROOTS['A']
 raw=(HERE/'make_inputs_registered_v3.py').read_bytes()
 with sf.open(root+'/make_inputs_registered_v3.py','wx') as f:f.write(raw)
 task={'root':root,'name':'inputs_registered_v3','timeout_seconds':900,'cpu_only':True,'command':[cfg['python'],'-u','-B',root+'/make_inputs_registered_v3.py','--data','/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone/VisDrone2019-DET-train']}
 with sf.open(root+'/inputs_registered_v3_task.json','wx') as f:f.write(json.dumps(task).encode())
 program='''import pathlib,json,subprocess,os,hashlib
root=pathlib.Path(PROBE_ROOT);inputs=root/'inputs'
assert inputs.is_dir() and not list(inputs.iterdir()),'Preserve and inspect partial inputs; do not regenerate'
assert not list(root.glob('isolated_results_seed*'))
assert hashlib.sha256((root/'make_inputs_registered_v3.py').read_bytes()).hexdigest()==EXPECTED
inputs.rename(root/'inputs_empty_failed_Mosaic_registration_v3')
with (root/'inputs_registered_v3_supervisor.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(root/'bounded_task.py'),str(root/'inputs_registered_v3_task.json')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps({'state':'CPU_data_preparation_with_native_Mosaic_import','supervisor_pid':p.pid,'previous_valid_inputs':0,'model_forwards':0}),flush=True)
'''
 r=ctl.execute(c,'A','PROBE_ROOT='+repr(root)+'\nEXPECTED='+repr(hashlib.sha256(raw).hexdigest())+'\n'+program,timeout=30)
 (LOCAL/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_Mosaic_import_repair.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(r,ensure_ascii=False));assert r['returncode']==0
finally:
 pw.clear()
 if c:c.close()
