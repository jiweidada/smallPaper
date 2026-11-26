from pathlib import Path
import datetime,json,os,signal,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parent

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def ticks(pid):
    try:return int(Path('/proc/%s/stat'%pid).read_text().split(') ',1)[1].split()[19])
    except FileNotFoundError:return None

def stage(name,args,folder,limit):
    folder.mkdir(exist_ok=True)
    with (folder/'stdout.log').open('x') as stdout,(folder/'stderr.log').open('x') as stderr:
        child=subprocess.Popen([sys.executable,'-u']+args,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
        start=time.monotonic()
        identity={'stage':name,'pid':child.pid,'start_ticks':ticks(child.pid),'unix':time.time()}
        save(folder/'child_identity.json',identity)
        while child.poll() is None:
            save(ROOT/'supervisor_progress.json',{'status':'RUNNING','stage':name,'child':identity,
                        'elapsed_stage_seconds':time.monotonic()-start,'supervisor_pid':os.getpid()})
            if time.monotonic()-start>limit:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError('Own stage timeout: '+name)
            time.sleep(10)
        save(folder/'exit.json',{'exit_code':child.returncode,'identity':identity,'unix':time.time()})
        if child.returncode:raise RuntimeError(name+' failed with exit '+str(child.returncode))

def summarize():
    rows={a:json.loads((ROOT/('arm_'+a)/'result.json').read_text()) for a in ['B','M','Q']}
    assert all(v['status']=='COMPLETED' and v['epochs']==12 and v['optimizer_updates']==9708 for v in rows.values())
    for epoch in range(1,13):
        e={a:json.loads((ROOT/('arm_'+a)/('epoch%02d_input_flow.json'%epoch)).read_text()) for a in rows}
        for key in ['order_sha256','first_eight_actual_input_sha256','first_eight_pre_forward_rng_sha256','all_batches_targets_sha256']:
            assert e['B'][key]==e['M'][key]==e['Q'][key],(epoch,key)
    metrics={a:v['endpoint_metrics'] for a,v in rows.items()}
    qb=metrics['Q']['APs']-metrics['B']['APs'];qm=metrics['Q']['APs']-metrics['M']['APs']
    qa=metrics['Q']['AP']-metrics['B']['AP']
    gates={'Q_minus_B_APs_ge_0_5pp':qb>=.5,'Q_minus_M_APs_ge_0_3pp':qm>=.3,'Q_minus_B_AP_ge_minus_0_2pp':qa>=-.2}
    return {'status':'COMPLETED','metrics':metrics,'Q_minus_B_APs_pp':qb,'Q_minus_M_APs_pp':qm,
            'Q_minus_B_AP_pp':qa,'gates':gates,'pilot_pass':all(gates.values()),
            'decision':'WORTH_INDEPENDENT_REPLICATION_NOT_NOVELTY_PROOF' if all(gates.values()) else 'STOP_THIS_WEIGHTING_RECIPE_NO_RESCUE_SWEEP',
            'formal_training_updates':sum(v['optimizer_updates'] for v in rows.values()),
            'formal_model_forwards':sum(v['full_model_forwards'] for v in rows.values()),
            'common_initialization_and_input_rng_checks':'PASS','new_direction_automatically_queued':False,
            'scope':'single-seed12-epoch controlled pilot; not significance or completed paper',
            'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}

if __name__=='__main__':
    with (ROOT/'supervisor_once.lock').open('x') as f:f.write(str(os.getpid()))
    save(ROOT/'supervisor_identity.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'unix':time.time(),
              'stages':['preparation','capacity','B12','M12','Q12'],'auto_retry':False,'additional_training_queue':False})
    try:
        stage('preparation',[str(ROOT/'prepare_server_A48.py')],ROOT/'stage_preparation',20*60)
        stage('capacity',[str(ROOT/'capacity_A48.py')],ROOT/'stage_capacity',20*60)
        for arm in ['B','M','Q']:
            stage(arm+'12',[str(ROOT/'train_A48.py'),'--arm',arm],ROOT/('arm_'+arm),20*3600)
        result=summarize();save(ROOT/'three_arm_result.json',result)
        save(ROOT/'supervisor_progress.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        result={'status':'FAILED_STOPPED_NO_RETRY','error':repr(exc),'traceback':traceback.format_exc(),'unix':time.time()}
        save(ROOT/'supervisor_progress.json',result)
        print(json.dumps(result),flush=True)
        raise
