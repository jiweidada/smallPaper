from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile, sys, os, importlib.metadata as metadata, subprocess

root = Path('/25085400120/mpsma_fair_20261103_v1')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()
assert digest(root/'input_bundle.zip') == 'ac08573ba51a11bd6063a1fd29cd16421fa8d5d0d4d20167528648688f7ceb85'
assert not (root/'source_snapshot').exists()
with zipfile.ZipFile(root/'input_bundle.zip') as z:
    for n in z.namelist():
        assert (root/n).resolve().is_relative_to(root)
    z.extractall(root)
for row in json.loads((root/'source_manifest.json').read_text()):
    assert digest(root/row['snapshot']) == row['sha256']
proposal=json.loads((root/'proposed_protocol.json').read_text())
data_root=Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
report={'status':'CHECKING','created_utc':datetime.now(timezone.utc).isoformat(),'source_files':229,'splits':{}}
for split in ['train','val']:
    folder=data_root/('VisDrone2019-DET-'+split)
    annotation=folder/'annotations'/f'{split}.json'
    expected=proposal['data'][split]['annotation_sha256']
    rows=json.loads((root/('training_image_manifest.json' if split=='train' else 'validation_image_manifest.json')).read_text())
    if isinstance(rows,dict): rows=rows['images']
    mismatches=[]
    for row in rows:
        p=folder/'images'/row['file_name']
        if not p.is_file() or digest(p)!=row['sha256']: mismatches.append(row['file_name'])
    actual=digest(annotation) if annotation.exists() else None
    report['splits'][split]={'images':len(rows),'image_root':str(folder/'images'),'mismatches':mismatches,
                            'annotation':str(annotation),'annotation_expected_sha256':expected,
                            'annotation_actual_sha256':actual,'annotation_matches':actual==expected}
report['status']='INPUTS_VERIFIED' if all(not r['mismatches'] and r['annotation_matches'] for r in report['splits'].values()) else 'INPUT_MISMATCH_BEFORE_MODEL_EXECUTION'
(root/'input_verification.json').write_text(json.dumps(report,indent=2)+'\n')
import torch
env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
     'cudnn':torch.backends.cudnn.version(),'torch_build':torch.__config__.show(),
     'packages':{d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')},
     'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv,noheader'],text=True).strip()}
(root/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
if report['status']=='INPUTS_VERIFIED':
    for arm in ['BASE','FULL']:
        cfg=json.loads((root/'configs'/f'{arm}.proposed.yml').read_text())
        for split in ['train','val']:
            cfg[f'{split}_dataloader']['dataset']['ann_file']=report['splits'][split]['annotation']
            cfg[f'{split}_dataloader']['dataset']['img_folder']=report['splits'][split]['image_root']
        cfg['output_dir']=str(root/'runs'/arm)
        (root/'configs'/f'{arm}.yml').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({'status':report['status'],'images':{k:r['images'] for k,r in report['splits'].items()},
                  'mismatch_counts':{k:len(r['mismatches']) for k,r in report['splits'].items()},
                  'annotation_matches':{k:r['annotation_matches'] for k,r in report['splits'].items()},
                  'python':sys.executable,'torch':env['torch'],'cuda':env['cuda']}))
