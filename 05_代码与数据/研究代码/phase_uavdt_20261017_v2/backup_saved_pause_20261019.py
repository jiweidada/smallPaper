"""Wait for stopped saved checkpoints, download every listed checkpoint/config/log.
No training starts, no process signals, no remote writes in this transfer stage.
"""
import argparse,concurrent.futures,datetime,hashlib,json,pathlib,sys,time
import control_uavdt as ctl
LOCAL=ctl.HERE.parents[1]/'data/07_第二数据集_20261017/用户暂停与本地备份_20261019_v1'
LOCAL.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def one(account,passwords):
    cfg=ctl.ACCOUNTS[account];client=ctl.connect(cfg['port'],passwords);sftp=client.open_sftp();sftp.get_channel().settimeout(60)
    local=LOCAL/account;local.mkdir(exist_ok=True);start=time.monotonic();last_print=0
    def readremote(p):
        with sftp.open(p,'rb') as f:return f.read()
    try:
        while True:
            try:
                failure=readremote(cfg['root']+'/pause_user_20261019_v1/failure.json')
            except FileNotFoundError:failure=None
            if failure:raise RuntimeError('Pause watcher failed: '+failure.decode())
            try:raw=readremote(cfg['root']+'/pause_user_20261019_v1/receipt.json');receipt=json.loads(raw)
            except FileNotFoundError:receipt={}
            if time.monotonic()-last_print>30:
                print(json.dumps({'account':account,'phase':'waiting_boundary','state':receipt.get('state'),'jobs':[{k:j.get(k) for k in ['arm','state','completed_epochs','target_epoch']} for j in receipt.get('jobs',[])]}),flush=True);last_print=time.monotonic()
            if receipt.get('state')=='all_saved_paused_waiting_local_backup':break
            assert time.monotonic()-start<7500
            time.sleep(5)
        manifest_raw=readremote(cfg['root']+'/pause_user_20261019_v1/backup_manifest.json')
        assert hashlib.sha256(manifest_raw).hexdigest()==receipt['backup_manifest_sha256']
        manifest=json.loads(manifest_raw);assert manifest['root']==cfg['root']
        (local/'pause_receipt.json').write_bytes(raw);(local/'backup_manifest.json').write_bytes(manifest_raw)
        done=0;total=manifest['bytes'];files=[];last_print=0
        for item in manifest['files']:
            rel=pathlib.PurePosixPath(item['relative']);assert not rel.is_absolute() and '..' not in rel.parts
            target=local/'files'/pathlib.Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
            remote=cfg['root']+'/'+str(rel)
            if not target.exists():
                part=target.with_name(target.name+'.part')
                def progress(n,size):
                    nonlocal last_print
                    if time.monotonic()-last_print>20:
                        print(json.dumps({'account':account,'phase':'backup_transfer','file':str(rel),'bytes_done':done+n,'bytes_total':total}),flush=True);last_print=time.monotonic()
                offset=part.stat().st_size if part.exists() else 0
                assert 0<=offset<=item['bytes']
                # Installed Paramiko predates bounded-prefetch arguments. Explicit
                # batches of 32 32KiB read requests bound memory/server pressure.
                with sftp.open(remote,'rb') as source,part.open('ab' if offset else 'xb') as target_stream:
                    while offset<item['bytes']:
                        chunks=[(pos,min(32768,item['bytes']-pos)) for pos in range(offset,min(offset+32*32768,item['bytes']),32768)]
                        for (pos,size),block in zip(chunks,source.readv(chunks)):
                            assert len(block)==size
                            target_stream.write(block);offset+=len(block);progress(offset,item['bytes'])
                assert part.stat().st_size==item['bytes'] and sha(part)==item['sha256']
                part.rename(target)
            assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
            done+=item['bytes'];files.append(item)
        result={'account':account,'status':'local_backup_verified','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes':done,'files':len(files),'all_sha256_match':True,'source_root':cfg['root'],'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'completed_epochs':{j['job']:j['completed_epochs'] for j in receipt['jobs']},'checkpoint_zip_crc_passed':all(j['zip_crc_passed'] for j in receipt['jobs']),'local_path':str(local)}
        (local/'local_backup_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False),flush=True);return result
    finally:client.close()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--account',choices=['A','B','both'],default='both');args=parser.parse_args()
    accounts=list(ctl.ACCOUNTS) if args.account=='both' else [args.account]
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending={a:pool.submit(one,a,passwords) for a in accounts}
            results={}
            for a,f in pending.items():
                try:results[a]=f.result()
                except Exception as e:results[a]={'status':'failed','error':type(e).__name__+': '+str(e)}
    finally:passwords.clear()
    (LOCAL/('备份结果_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    if args.account=='both':(LOCAL/'双账号备份汇总.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    assert all(r['status']=='local_backup_verified' for r in results.values())
