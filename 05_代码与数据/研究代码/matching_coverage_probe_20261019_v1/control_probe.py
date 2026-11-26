"""Task-scoped remote packaging/status, using the existing pinned SSH identities."""
import argparse,concurrent.futures,datetime,hashlib,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;WORK=HERE.parents[1]
sys.path.insert(0,str(WORK/'code/phase_uavdt_20261017_v2'))
import control_uavdt as ctl
LOCAL=WORK/'data/08_匹配聚合覆盖探针_20261019_v1'
ROOTS={a:c['storage']+'/DEIM/experiments/matching_coverage_probe_20261019_v1' for a,c in ctl.ACCOUNTS.items()}
DATA={'A':'/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone/VisDrone2019-DET-train','B':'/24085403014/DEIM/dataset_visdrone/VisDrone2019-DET-train'}
DEPLOY='''import pathlib,json,hashlib,tarfile,shutil
root=pathlib.Path(PROBE['root']);archive=pathlib.Path(PROBE['archive'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==PROBE['archive_sha256']
root.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  rel=pathlib.PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  dest=root/pathlib.Path(*rel.parts);dest.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(member) as source,dest.open('xb') as target:shutil.copyfileobj(source,target)
manifest=json.loads((root/'package_manifest.json').read_text())
for item in manifest['files']:assert hashlib.sha256((root/item['relative']).read_bytes()).hexdigest()==item['sha256'],item['relative']
plan=json.loads((root/'sample_plan.json').read_text());seed=PROBE['seed'];source=pathlib.Path(CONFIG['storage'])/'DEIM/experiments/fine_residual_screen24_20261009_v2/runs'/f'baseline_seed{seed}'/'last.pth'
assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['checkpoints'][str(seed)]['sha256']
(root/'assets').mkdir();dest=root/'assets'/f'baseline24_seed{seed}.pth';shutil.copyfile(source,dest)
assert hashlib.sha256(dest.read_bytes()).hexdigest()==plan['checkpoints'][str(seed)]['sha256']
receipt={'state':'deployed_not_started','account':CONFIG['account'],'seed':seed,'root':str(root),'checkpoint_sha256':plan['checkpoints'][str(seed)]['sha256'],'plan_sha256':manifest['plan_sha256'],'native_math_sources':'30 bound training Python files exact normalized SHA; minimal package __init__ import registries differ, see local source review','data':PROBE['data']}
(root/'deployment_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
'''
STATUS='''import pathlib,json
root=pathlib.Path(PROBE['root'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
result={'account':CONFIG['account'],'probe':read(root/'deployment_receipt.json'),'input_complete':read(root/'inputs/complete.json'),'input_progress':read(root/'inputs/progress.json')}
for seed in [1,2]:result['seed'+str(seed)]={name:read(root/f'results_seed{seed}'/(name+'.json')) for name in ['complete','failure','progress']}
old=pathlib.Path(CONFIG['root'])/'pause_user_20261019_v1'
result['old_pause']=read(old/'receipt.json');result['old_pause_failure']=read(old/'failure.json');print(json.dumps(result),flush=True)
'''
def one(account,passwords,action):
    cfg=ctl.ACCOUNTS[account];client=ctl.connect(cfg['port'],passwords)
    try:
        probe={'root':ROOTS[account],'seed':1 if account=='A' else 2,'data':DATA[account]}
        if action=='deploy':
            path=LOCAL/'probe_package.tar.gz';raw=path.read_bytes();remote=cfg['storage']+'/DEIM/experiments/matching_coverage_probe_20261019_v1.tar.gz'
            sftp=client.open_sftp()
            with sftp.open(remote,'wx') as f:f.write(raw)
            probe.update(archive=remote,archive_sha256=hashlib.sha256(raw).hexdigest());program=DEPLOY
        else:program=STATUS
        return ctl.execute(client,account,'PROBE='+repr(probe)+'\n'+program,timeout=60)
    finally:client.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','status']);args=p.parse_args()
    passwords=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs={a:pool.submit(one,a,passwords,args.action) for a in ROOTS};results={}
            for a,f in fs.items():
                try:results[a]=f.result()
                except Exception as e:results[a]={'returncode':None,'error':type(e).__name__+': '+str(e)}
    finally:passwords.clear()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');path=LOCAL/(stamp+'_'+args.action+'.json');path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'saved':str(path),'results':{a:({'returncode':v.get('returncode'),'error':v.get('error'),'stdout':v.get('stdout','')[-600:]}) for a,v in results.items()}},ensure_ascii=False,indent=2))
