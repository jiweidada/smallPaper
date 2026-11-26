"""Independent CPU-only validation of source image hashes and intervention domain."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
import torch
torch.set_num_threads(2)
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
S3=ROOT/'code/unit_instance_completion_20261021_v1/342_codex_stage3_20261022_v1'
VAL=Path(r'C:/Users/jiwei/Desktop/DEIM/dataset_visdrone/VisDrone2019-DET-val')
ds=json.loads((VAL/'annotations/val.json').read_text('utf-8'))
image_hash={im['id']:hashlib.sha256((VAL/'images'/im['file_name']).read_bytes()).hexdigest() for im in ds['images']}
result={}
for seed in (1,2):
    bad=[];source_errors=[];nrows={300:0,600:0};lowest={300:1.,600:1.}
    for im in ds['images']:
        e=torch.load(S3/f'342_raw_val_seed{seed}/342_{im["id"]:06d}.pt',map_location='cpu',weights_only=True)
        if e['source_image_sha256']!=image_hash[im['id']]:source_errors.append(im['id'])
        scores=e['final300_logits'].sigmoid().flatten()
        for K in (300,600):
            s,f=scores.topk(K);q=f//10;k=s>.001
            b=e['final300_boxes'][q[k]]
            invalid=~((b[:,2:]>0).all(-1)&torch.isfinite(b).all(-1))
            nrows[K]+=int(k.sum());lowest[K]=min(lowest[K],float(s[k].min()))
            for z in torch.where(invalid)[0].tolist():bad.append({'image_id':im['id'],'K':K,'query':int(q[k][z]),'flat':int(f[k][z]),'score':float(s[k][z]),'box':b[z].tolist()})
    result[str(seed)]={'image_hash_checks':548,'source_image_errors':source_errors,'intervention_selected_invalid_rows':bad,'exported_row_counts':nrows,'lowest_exported_score':lowest}
with open(OUT/'input_domain_validation.json','x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
assert all(not r['source_image_errors'] and not r['intervention_selected_invalid_rows'] for r in result.values())
