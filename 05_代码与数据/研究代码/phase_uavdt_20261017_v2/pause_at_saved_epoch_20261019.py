"""User-authorized external epoch-boundary watcher. Frozen training sources stay untouched.

Stops only validated owned process groups after a new complete evaluation log row,
which the inspected native solver writes AFTER all checkpoint saves. Keeps CUDA
allocations until local backup is verified and a separate release action runs.
"""
import argparse, datetime, hashlib, json, os, pathlib, signal, time, zipfile

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,v):
    t=p.with_suffix(p.suffix+'.next');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def identity(entry,root):
    p=pathlib.Path('/proc')/str(entry['pid'])
    st=(p/'stat').read_text().rsplit(')',1)[1].split()
    args=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
    assert st[19]==str(entry['start_ticks']) and st[0]!='Z','Process identity changed'
    assert any(s.startswith(str(root/'code')+'/') for s in args),'Outside owned code'
    return st[0]
def rows(p):
    result=[]
    for line in p.read_text().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if 'test_uavdt_eval_bbox' in r:result.append(r)
    assert [r['epoch'] for r in result]==list(range(len(result)))
    return result
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(root):
    root=root.resolve();dest=root/'pause_user_20261019_v1';dest.mkdir(exist_ok=False)
    (dest/'before').mkdir()
    plan=read(root/'plan.json');controller=read(root/'launch_receipt.json')
    assert plan['epoches']==96 and len(plan['jobs'])==3
    identity(controller,root)
    jobs=[]
    for arm in plan['jobs']:
        job=f'uavdt_{arm}_full96_seed{plan["seed"]}';status=read(root/'jobs'/(job+'.json'))
        identity(status,root);assert os.getpgid(status['pid'])==status['pid']
        r=rows(root/'outputs'/job/'log.txt');assert r and r[-1]['epoch']<95
        jobs.append({'job':job,'arm':arm,'identity':status,'initial_log_epoch':r[-1]['epoch'],'target_epoch':r[-1]['epoch']+1,'state':'waiting_for_saved_epoch'})
        (dest/'before'/(job+'.json')).write_bytes((root/'jobs'/(job+'.json')).read_bytes())
    for name in ['queue_state.json','launch_receipt.json','queue.lock','plan.json','frozen_manifest.json']:
        p=root/name
        if p.is_file():(dest/'before'/name).write_bytes(p.read_bytes())
    receipt={'root':str(root),'account':plan['account'],'seed':plan['seed'],'started_utc':now(),'state':'waiting_for_epoch_boundaries','controller':controller,'jobs':jobs,'source_sha256':sha(pathlib.Path(__file__)),'automatic_training_restart':False}
    write(dest/'receipt.json',receipt)
    identity(controller,root);os.kill(controller['pid'],signal.SIGSTOP)
    time.sleep(.05);assert identity(controller,root)=='T'
    q=read(root/'queue_state.json');q.update(state='user_pause_pending_epoch_boundary',user_pause_requested_utc=now(),pause_receipt=str(dest/'receipt.json'));write(root/'queue_state.json',q)
    started=time.monotonic()
    while any(j['state']=='waiting_for_saved_epoch' for j in jobs):
        assert time.monotonic()-started<7200,'Epoch-boundary wait exceeded 2 hours; no forced early stop'
        for j in jobs:
            if j['state']!='waiting_for_saved_epoch':continue
            identity(j['identity'],root)
            log=root/'outputs'/j['job']/'log.txt';r=rows(log)
            if r[-1]['epoch']<j['target_epoch']:continue
            identity(j['identity'],root);os.killpg(j['identity']['pid'],signal.SIGSTOP)
            time.sleep(.05);assert identity(j['identity'],root)=='T'
            j.update(state='stopped_after_saved_epoch',saved_epoch=r[-1]['epoch'],completed_epochs=r[-1]['epoch']+1,paused_utc=now(),metrics=r[-1]['test_uavdt_eval_bbox'])
            s=read(root/'jobs'/(j['job']+'.json'));s.update(state='paused_saved_epoch_gpu_held',user_pause=True,paused_completed_epochs=j['completed_epochs'],pause_receipt=str(dest/'receipt.json'));write(root/'jobs'/(j['job']+'.json'),s)
            write(dest/'receipt.json',receipt)
        time.sleep(.1)
    import torch
    for j in jobs:
        cp=root/'outputs'/j['job']/'last.pth'
        with zipfile.ZipFile(cp) as z:assert z.testzip() is None
        value=torch.load(cp,map_location='cpu',weights_only=False)
        assert value['last_epoch']==j['saved_epoch']
        assert {'model','ema','optimizer','scaler','lr_warmup_scheduler'}.issubset(value)
        j.update(checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,checkpoint_keys=sorted(value),model_entries=len(value['model']),optimizer_state_entries=len(value['optimizer']['state']),ema_updates=value['ema']['updates'],zip_crc_passed=True)
        del value
    files=[]
    targets=[root/'plan.json',root/'frozen_manifest.json',root/'queue.lock',root/'launch_receipt.json',root/'final_acceptance.json',root/'capacity_receipt.json',root/'release_after_preflight.json']
    targets+=list((root/'logs').glob('*.log'))
    for j in jobs:
        out=root/'outputs'/j['job'];targets+=list(out.glob('*.pth'))
        targets += [out/'log.txt',out/'args.json',root/'jobs'/(j['job']+'.json')]
    targets += list((root/'code/configs').rglob('*.yml'))+list((root/'code/configs').rglob('*.yaml'))
    for p in sorted(set(targets)):
        assert p.resolve().is_relative_to(root)
        if p.is_file():files.append({'relative':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
    for item in read(root/'frozen_manifest.json')['files']:
        assert sha(root/item['relative'])==item['sha256'],'Frozen source/config changed'
    write(dest/'backup_manifest.json',{'root':str(root),'created_utc':now(),'files':files,'bytes':sum(f['bytes'] for f in files)})
    receipt.update(state='all_saved_paused_waiting_local_backup',finished_utc=now(),backup_manifest_sha256=sha(dest/'backup_manifest.json'),frozen_manifest_verified=True)
    write(dest/'receipt.json',receipt)
    q=read(root/'queue_state.json');q.update(state='paused_by_user_waiting_local_backup',pause_receipt=str(dest/'receipt.json'));write(root/'queue_state.json',q)
    print(json.dumps({'state':receipt['state'],'completed_epochs':{j['job']:j['completed_epochs'] for j in jobs}}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as e:
        dest=a.root/'pause_user_20261019_v1'
        if dest.is_dir():write(dest/'failure.json',{'time':now(),'error':type(e).__name__+': '+str(e),'no_automatic_retry':True})
        raise
