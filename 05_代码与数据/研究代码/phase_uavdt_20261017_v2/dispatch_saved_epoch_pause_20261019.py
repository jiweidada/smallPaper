"""Single-use deployment for the user-authorized epoch boundary pause."""
import concurrent.futures,datetime,hashlib,json,pathlib,shlex,sys
import control_uavdt as ctl

LOCAL=ctl.HERE.parents[1]/'data/07_第二数据集_20261017/用户暂停与本地备份_20261019_v1'
LOCAL.mkdir(exist_ok=True)
SOURCE=ctl.HERE/'pause_at_saved_epoch_20261019.py'

def one(account,passwords):
    cfg=ctl.ACCOUNTS[account];client=ctl.connect(cfg['port'],passwords)
    try:
        sftp=client.open_sftp();base=cfg['root']+'/pause_control_20261019_v1'
        sftp.mkdir(base)
        target=base+'/'+SOURCE.name;raw=SOURCE.read_bytes()
        with sftp.open(target,'wx') as f:f.write(raw)
        program='''import json,subprocess,os,pathlib,hashlib,datetime
p=pathlib.Path(CONFIG['pause_script']);assert hashlib.sha256(p.read_bytes()).hexdigest()==CONFIG['pause_script_sha256']
assert not (pathlib.Path(CONFIG['root'])/'pause_user_20261019_v1').exists()
log=p.parent/'watcher.log'
with log.open('xb') as f:
 process=subprocess.Popen([CONFIG['python'],'-u','-B',str(p),'--root',CONFIG['root']],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 ticks=pathlib.Path('/proc')/str(process.pid)/'stat'
 result={'account':CONFIG['account'],'pid':process.pid,'start_ticks':ticks.read_text().rsplit(')',1)[1].split()[19],'command':[CONFIG['python'],str(p),'--root',CONFIG['root']],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(p.parent/'launch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
'''
        original=cfg.copy();cfg.update(pause_script=target,pause_script_sha256=hashlib.sha256(raw).hexdigest())
        result=ctl.execute(client,account,program,timeout=30)
        cfg.clear();cfg.update(original)
        return result
    finally:client.close()

if __name__=='__main__':
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={a:pool.submit(one,a,passwords) for a in ctl.ACCOUNTS}
            results={}
            for a,f in futures.items():
                try:results[a]=f.result()
                except Exception as e:results[a]={'error':str(e),'no_automatic_retry':True}
    finally:passwords.clear()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (LOCAL/(stamp+'_pause_dispatch.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
