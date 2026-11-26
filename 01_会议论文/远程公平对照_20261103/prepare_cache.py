from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys,shutil,zipfile,traceback,importlib.metadata as metadata,subprocess
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
CACHE=Path('/tmp/mpsma_fair_20261103_data')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()
def save(v):
    p=ROOT/'cache_progress.json.tmp';p.write_text(json.dumps(v,indent=2)+'\n');os.replace(p,ROOT/'cache_progress.json')
report={'status':'VERIFYING_ARCHIVE','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'completed_files':0}
save(report)
try:
    manifest=json.loads((ROOT/'dataset_bundle_manifest.json').read_text())
    assert sha(ROOT/'verified_dataset.zip')==manifest['zip_sha256']
    assert shutil.disk_usage('/tmp').free>manifest['bytes']*2+5*1024**3
    CACHE.mkdir(exist_ok=False)
    with zipfile.ZipFile(ROOT/'verified_dataset.zip') as z:
        assert set(z.namelist())=={r['path'] for r in manifest['files']}
        for i,row in enumerate(manifest['files'],1):
            dest=(CACHE/row['path']).resolve();assert dest.is_relative_to(CACHE)
            dest.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();size=0
            with z.open(row['path']) as src,dest.open('xb') as out:
                while b:=src.read(4*1024*1024):out.write(b);h.update(b);size+=len(b)
            assert h.hexdigest()==row['sha256'] and size==row['bytes'],row['path']
            if i%1000==0:
                report.update({'status':'EXTRACTING_VERIFIED_CACHE','completed_files':i});save(report);print(i,flush=True)
    receipt={'status':'ALL_INPUT_BYTES_VERIFIED','archive_sha256':manifest['zip_sha256'],
             'manifest_sha256':sha(ROOT/'dataset_bundle_manifest.json'),'cache_root':str(CACHE),
             'files':len(manifest['files']),'train_images':6471,'validation_images':548,
             'source':'Previously verified local dataset transferred to the user-authorized server; no pixel or annotation conversion.',
             'completed_utc':datetime.now(timezone.utc).isoformat()}
    (ROOT/'cache_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'execution_configs').mkdir(exist_ok=False)
    for arm in ['BASE','FULL']:
        cfg=json.loads((ROOT/'configs'/f'{arm}.proposed.yml').read_text())
        for split in ['train','val']:
            cfg[f'{split}_dataloader']['dataset']['ann_file']=str(CACHE/split/'annotations'/f'{split}.json')
            cfg[f'{split}_dataloader']['dataset']['img_folder']=str(CACHE/split/'images')
        cfg['output_dir']=str(ROOT/'runs'/arm)
        (ROOT/'execution_configs'/f'{arm}.yml').write_text(json.dumps(cfg,indent=2)+'\n')
    import torch
    env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
         'cudnn':torch.backends.cudnn.version(),'torch_build':torch.__config__.show(),
         'packages':{d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')},
         'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv,noheader'],text=True).strip()}
    (ROOT/'environment.current.json').write_text(json.dumps(env,indent=2)+'\n')
    report.update({'status':'CACHE_AND_CONFIGS_READY','completed_files':len(manifest['files']),'completed_utc':datetime.now(timezone.utc).isoformat()})
    save(report);print(json.dumps(report),flush=True)
except Exception:
    report.update({'status':'CACHE_PREPARATION_FAILED','traceback':traceback.format_exc()});save(report);raise
