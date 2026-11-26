"""Release VRAM ONLY after both local backups passed. Never restart training."""
import concurrent.futures,datetime,json,pathlib,sys
import control_uavdt as ctl
LOCAL=ctl.HERE.parents[1]/'data/07_第二数据集_20261017/用户暂停与本地备份_20261019_v1'
PROGRAM='''import pathlib,json,hashlib,os,signal,time,subprocess,datetime
root=pathlib.Path(CONFIG['root']);dest=root/'pause_user_20261019_v1'
receipt=json.loads((dest/'receipt.json').read_text());assert receipt['state']=='all_saved_paused_waiting_local_backup'
assert PROOF['all_sha256_match'] and PROOF['source_root']==str(root)
assert hashlib.sha256((dest/'backup_manifest.json').read_bytes()).hexdigest()==PROOF['manifest_sha256']
def ident(entry):
 p=pathlib.Path('/proc')/str(entry['pid']);st=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert st[19]==str(entry['start_ticks']) and st[0]=='T'
 args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
 assert any(x.startswith(str(root/'code')+'/') for x in args)
for j in receipt['jobs']:
 ident(j['identity']);cp=root/'outputs'/j['job']/'last.pth'
 assert hashlib.sha256(cp.read_bytes()).hexdigest()==j['checkpoint_sha256']
ident(receipt['controller'])
# The controller is stopped before the trainers so it cannot consume their
# intentional signal exits as failures or advance its old queue.
c=receipt['controller'];os.kill(c['pid'],signal.SIGTERM);os.kill(c['pid'],signal.SIGCONT)
for j in receipt['jobs']:
 pid=j['identity']['pid'];assert os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM);os.killpg(pid,signal.SIGCONT)
time.sleep(2)
for entry in [receipt['controller']]+[j['identity'] for j in receipt['jobs']]:
 p=pathlib.Path('/proc')/str(entry['pid'])
 if p.exists():
  st=(p/'stat').read_text().rsplit(')',1)[1].split();assert st[19]!=str(entry['start_ticks']) or st[0]=='Z','Owned old process still active'
def write(p,v):
 t=p.with_suffix('.next');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
for j in receipt['jobs']:
 p=root/'jobs'/(j['job']+'.json');v=json.loads(p.read_text());v.update(state='paused_by_user_resumable_checkpoint',user_pause=True,paused_completed_epochs=j['completed_epochs'],old_process_terminated_for_vram_release=True,checkpoint_sha256=j['checkpoint_sha256']);write(p,v)
q=json.loads((root/'queue_state.json').read_text());q.update(state='paused_by_user',user_pause=True,active_jobs={},local_backup_verified=PROOF);write(root/'queue_state.json',q)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
assert not gpu.strip(),'Unexpected GPU compute process; do not launch probe'
result={'state':'paused_backup_verified_GPU_released','account':CONFIG['account'],'completed_epochs':{j['job']:j['completed_epochs'] for j in receipt['jobs']},'local_backup':PROOF,'GPU_compute':gpu,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_training_restart':False}
write(dest/'release_receipt.json',result);print(json.dumps(result),flush=True)
'''
def one(a,passwords,proof):
    client=ctl.connect(ctl.ACCOUNTS[a]['port'],passwords)
    try:return ctl.execute(client,a,'PROOF='+repr(proof)+'\n'+PROGRAM,timeout=45)
    finally:client.close()
if __name__=='__main__':
    proofs={a:json.loads((LOCAL/a/'local_backup_verified.json').read_text(encoding='utf8')) for a in ctl.ACCOUNTS}
    assert all(v['status']=='local_backup_verified' and v['all_sha256_match'] for v in proofs.values())
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending={a:pool.submit(one,a,passwords,proofs[a]) for a in ctl.ACCOUNTS};results={a:f.result() for a,f in pending.items()}
    finally:passwords.clear()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(LOCAL/(stamp+'_release.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(results,ensure_ascii=False,indent=2));assert all(r['returncode']==0 for r in results.values())
