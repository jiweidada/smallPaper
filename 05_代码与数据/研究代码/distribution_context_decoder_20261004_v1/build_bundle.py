"""Local bundle construction; no credentials and no mutation of upstream."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

code = Path(__file__).resolve().parent
workspace = code.parents[2]
out = workspace/'英文小论文/data/00_实验计划与证据索引/分布上下文模块实施_20261004_v1'
out.mkdir(exist_ok=False)
remote = '/25085400120/DEIM/experiments/distribution_context_seed1_batch8_pair96_20261004_v1'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
up = json.loads((code/'upstream_snapshot_receipt.json').read_text())
original = workspace/up['source']
assert all(sha(original/k)==v for k,v in up['files'].items())
for p in code.glob('*.py'):
    ast.parse(p.read_text(encoding='utf8'))
runtime = {'root':remote, 'GPU_uuid':'GPU-d29c55c5-02ab-1e69-c978-0b28f2102003',
           'python':'/25085400120/envs/deim/bin/python',
           'init':'/25085400120/DEIM/experiments/visdrone_light_20260928_090716/artifacts/common_init_seed0.pth'}
data = '/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone'
for split in ['train','val']:
    runtime[split+'_images'] = data+'/VisDrone2019-DET-'+split+'/images'
    runtime[split+'_ann'] = data+'/VisDrone2019-DET-'+split+'/annotations/'+split+'.json'
(out/'runtime.json').write_text(json.dumps(runtime,indent=2), encoding='utf8')
files = [(p, 'code/'+p.relative_to(code).as_posix()) for p in sorted(code.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yml','.yaml','.txt')]
files.append((out/'runtime.json','runtime.json'))
manifest = {'files':[{'path':name,'sha256':sha(p),'bytes':p.stat().st_size} for p,name in files],
            'upstream_unchanged_files':len(up['files']), 'status':'sealed_after_CPU_and_technical_self_review'}
(out/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
arc = out/'bundle.tar.gz'
with tarfile.open(arc,'w:gz') as tar:
    for p,name in files:
        tar.add(p, arcname=name, recursive=False)
    tar.add(out/'bundle_manifest.json', arcname='bundle_manifest.json', recursive=False)
receipt = {'remote_root':remote,'archive_sha256':sha(arc),'manifest_sha256':sha(out/'bundle_manifest.json'),
           'files':len(files),'bytes':arc.stat().st_size,'upstream_unchanged':True}
(out/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt))
