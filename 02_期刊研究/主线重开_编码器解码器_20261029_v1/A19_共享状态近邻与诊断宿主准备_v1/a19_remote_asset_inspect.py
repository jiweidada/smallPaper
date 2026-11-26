from pathlib import Path
import json,hashlib,zipfile,pickletools,torch
p=Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/vssd_tiny_host_assets_20261030_v1/vssd_tiny_mesa.pth')
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''): h.update(b)
assert p.stat().st_size==453140982
assert h.hexdigest()=='853f5719558dffcf8a629f4eb07b18914e397dc33f122525b1e2933d138cd545'
with zipfile.ZipFile(p) as z:
    names=[n for n in z.namelist() if n.endswith('data.pkl')]
    assert len(names)==1
    ops=list(pickletools.genops(z.read(names[0])))
    globals_=sorted({str(arg) for op,arg,pos in ops if op.name=='GLOBAL'})
r={'asset_sha256':h.hexdigest(),'asset_bytes':p.stat().st_size,'pickle_GLOBAL_names':globals_,'model_instantiated':False,'model_forward_calls':0,'optimizer_updates':0}
try:
    ck=torch.load(p,map_location='cpu',weights_only=True)
    r['safe_CPU_checkpoint_load_ok']=True
    r['top_level_keys']=list(ck) if isinstance(ck,dict) else [type(ck).__name__]
    groups={}
    if isinstance(ck,dict):
        for k,v in ck.items():
            if isinstance(v,dict):
                ts={n:t for n,t in v.items() if isinstance(t,torch.Tensor)}
                if ts:
                    groups[k]={'tensors':len(ts),'numel':sum(t.numel() for t in ts.values()),'sample_shapes':{n:list(t.shape) for n,t in ts.items() if n in ['head.weight','head.bias','norm.weight','patch_embed.proj.0.weight']}}
    r['tensor_groups']=groups
except Exception as e:
    r['safe_CPU_checkpoint_load_ok']=False
    r['load_error_type']=type(e).__name__
    r['load_error']=str(e)[:3500]
(p.parent/'A19_remote_weight_inspection.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r,ensure_ascii=False))

