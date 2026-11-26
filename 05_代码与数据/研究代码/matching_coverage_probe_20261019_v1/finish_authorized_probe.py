"""Single authorized bounded probe pipeline; no retries, training, or expansion."""
import concurrent.futures,datetime,json,pathlib,subprocess,sys,time,traceback
from control_probe import ctl,ROOTS,LOCAL,HERE

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def record(event):
    event={'client_utc':stamp(),**event}
    with (LOCAL/'执行流水_64输入.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    print(json.dumps(event,ensure_ascii=False),flush=True)

def status(a,passwords):
    client=ctl.connect(ctl.ACCOUNTS[a]['port'],passwords)
    try:
        sf=client.open_sftp();root=ROOTS[a];seed=1 if a=='A' else 2
        def read(rel):
            try:
                with sf.open(root+'/'+rel,'rb') as f:return json.loads(f.read())
            except FileNotFoundError:return None
        return {'account':a,'inputs':read('inputs/complete.json'),'input_progress':read('inputs/progress.json'),
                'input_task':read('task_records/inputs.json'),'probe_task':read(f'task_records/probe_seed{seed}.json'),
                'progress':read(f'results_seed{seed}/progress.json'),'complete':read(f'results_seed{seed}/complete.json'),
                'supplement':read(f'results_seed{seed}/supplement_complete.json'),'failure':read(f'results_seed{seed}/failure.json')}
    finally:client.close()

def run(name,args,timeout=900):
    log=LOCAL/('流水_'+name+'.log');record({'stage':name,'state':'starting','log':str(log)})
    with log.open('xb') as f:
        proc=subprocess.Popen([sys.executable,'-W','ignore','-X','utf8','-B']+args,stdout=f,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate();proc.wait(timeout=20);raise
    if rc:raise RuntimeError(name+': '+log.read_text(encoding='utf8',errors='replace')[-6000:])
    record({'stage':name,'state':'complete','returncode':rc})

def main():
    with (LOCAL/'authorized_finish_once.lock').open('x') as f:f.write(stamp())
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
    try:
        start=time.monotonic()
        while True:
            s=status('A',passwords);record({'stage':'wait_inputs','state':s})
            assert not (s['input_task'] or {}).get('state')=='failed',s
            if s['inputs']:
                assert s['inputs']['state']=='complete' and s['inputs']['inputs']==64;break
            assert time.monotonic()-start<1900;time.sleep(30)
        run('forward_A_dispatch',[str(HERE/'run_probe_stage.py'),'forward','--account','A'],120)
        run('same_inputs_to_B',[str(HERE/'transfer_probe.py'),'inputs'],900)
        run('forward_B_dispatch',[str(HERE/'run_probe_stage.py'),'forward','--account','B'],120)
        start=time.monotonic()
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures={a:pool.submit(status,a,passwords) for a in ['A','B']}
                states={a:f.result() for a,f in futures.items()}
            record({'stage':'wait_probes','states':states})
            for a,s in states.items():
                assert not s['failure'],s['failure']
                assert (s['probe_task'] or {}).get('state')!='failed',s
            if all(s['complete'] and s['supplement'] and (s['probe_task'] or {}).get('state')=='complete' for s in states.values()):break
            assert time.monotonic()-start<1900;time.sleep(30)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,'collect_'+a,[str(HERE/'transfer_probe.py'),'results','--account',a],900) for a in ['A','B']]
            for f in futures:f.result()
        run('analysis',['-s',str(HERE/'analyze_collected.py')],120)
        run('supplement_analysis',['-s',str(HERE/'summarize_diagnostics_v2.py')],120)
        record({'stage':'authorized_probe','state':'complete','short_training_started':False,'further_action':'report_results_to_user_only'})
    finally:passwords.clear()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        error={'client_utc':stamp(),'state':'failed','error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'automatic_retry':False,'no_training_launched':True}
        (LOCAL/'执行流水失败.json').write_text(json.dumps(error,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(error,ensure_ascii=False),flush=True);raise
