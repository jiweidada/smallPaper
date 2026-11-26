"""Append diagnostics before first forward; launch only bounded CPU unit checks."""
import concurrent.futures,datetime,hashlib,json
from control_probe import ctl,ROOTS,LOCAL,HERE

def one(a,passwords):
    cfg=ctl.ACCOUNTS[a];root=ROOTS[a];client=ctl.connect(cfg['port'],passwords)
    try:
        manifest={'before_first_real_forward':True,'original_package_unchanged':True,'files':[]}
        sources=[(HERE/'diagnostic_extension_v2.py','extension_v2/diagnostic_extension_v2.py'),
                 (HERE/'probe_driver_v2.py','probe_driver_v2.py'),
                 (HERE/'preflight_extension_v2.py','preflight_extension_v2.py'),
                 (LOCAL/'02_正式前向前补充限定_20261020_v2.txt','extension_v2/protocol_v2.txt')]
        sf=client.open_sftp()
        sf.mkdir(root+'/extension_v2')
        for source,rel in sources:
            raw=source.read_bytes()
            with sf.open(root+'/'+rel,'wx') as f:f.write(raw)
            manifest['files'].append({'relative':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        with sf.open(root+'/extension_v2/manifest.json','wx') as f:f.write(json.dumps(manifest,indent=2).encode())
        task={'root':root,'name':'CPU_extension_preflight_v2','timeout_seconds':900,'cpu_only':True,
              'command':[cfg['python'],'-u','-B',root+'/preflight_extension_v2.py','--seed',str(1 if a=='A' else 2)]}
        with sf.open(root+'/CPU_extension_preflight_v2_task.json','wx') as f:f.write(json.dumps(task).encode())
        program='''import pathlib,json,hashlib,subprocess,os
root=pathlib.Path(PROBE_ROOT)
assert not list(root.glob('results_seed*')),'A real forward has already started; freeze changes'
manifest=json.loads((root/'extension_v2/manifest.json').read_text())
for f in manifest['files']:assert hashlib.sha256((root/f['relative']).read_bytes()).hexdigest()==f['sha256']
for f in json.loads((root/'package_manifest.json').read_text())['files']:assert hashlib.sha256((root/f['relative']).read_bytes()).hexdigest()==f['sha256']
assert not (root/'task_records/CPU_extension_preflight_v2.lock').exists()
with (root/'CPU_extension_preflight_v2_supervisor.log').open('xb') as log:
 p=subprocess.Popen([CONFIG['python'],'-u','-B',str(root/'bounded_task.py'),str(root/'CPU_extension_preflight_v2_task.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print(json.dumps({'account':CONFIG['account'],'supervisor_pid':p.pid,'phase':'CPU extension tests only','model_forward':False,'manifest_sha256':hashlib.sha256((root/'extension_v2/manifest.json').read_bytes()).hexdigest()}),flush=True)
'''
        return ctl.execute(client,a,'PROBE_ROOT='+repr(root)+'\n'+program,timeout=45)
    finally:client.close()
if __name__=='__main__':
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs={a:pool.submit(one,a,passwords) for a in ['A','B']};results={}
            for a,f in fs.items():
                try:results[a]=f.result()
                except Exception as e:results[a]={'returncode':None,'error':type(e).__name__+': '+str(e)}
    finally:passwords.clear()
    p=LOCAL/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_extension_v2_deploy.json')
    p.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'saved':str(p),'results':results},ensure_ascii=False,indent=2))
