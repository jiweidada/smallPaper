"""Hash actual datasets/initialization and environment; no GPU allocation."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(root):
    r=json.loads((root/'runtime.json').read_text())
    result={'time':time.time(),'datasets':{},'init_sha256':sha(r['init']),
            'free_disk_GiB':shutil.disk_usage(root).free/2**30}
    for split in ['train','val']:
        annotation=Path(r[split+'_ann']);images=Path(r[split+'_images'])
        d=json.loads(annotation.read_text());rows=[]
        for im in sorted(d['images'],key=lambda x:x['id']):
            p=images/im['file_name'];rows.append({'id':im['id'],'file':im['file_name'],'bytes':p.stat().st_size,'sha256':sha(p)})
        result['datasets'][split]={'annotation_sha256':sha(annotation),'images':len(rows),
           'image_bytes_manifest_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
           'files':rows}
    command=[r['python'],'-c','import torch,json,platform;print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"python":platform.python_version(),"CUDA_initialized":torch.cuda.is_initialized()}))']
    result['environment']=json.loads(subprocess.check_output(command,text=True))
    result['gpu']=subprocess.check_output(['nvidia-smi','--id',r['GPU_uuid'],'--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True).strip()
    protocol=json.loads((root/'code/protocol_v1.json').read_text())
    result['checks']={'train_annotation_matches':result['datasets']['train']['annotation_sha256']==protocol['data_hashes']['train'],
        'val_annotation_matches':result['datasets']['val']['annotation_sha256']==protocol['data_hashes']['val'],
        'init_matches':result['init_sha256']==protocol['data_hashes']['common_init'],
        'image_counts_match':result['datasets']['train']['images']==6471 and result['datasets']['val']['images']==548,
        'RTX3090_confirmed':'RTX 3090' in result['gpu'],'no_CUDA_work':not result['environment']['CUDA_initialized'],
        'disk_at_least_40GiB':result['free_disk_GiB']>=40}
    result['status']='preflight_passed' if all(result['checks'].values()) else 'preflight_failed'
    with (root/'jobs/preflight.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='datasets'}),flush=True)
    if result['status']!='preflight_passed':raise RuntimeError('Preflight failed')

if __name__=='__main__':run(Path(sys.argv[1]))
