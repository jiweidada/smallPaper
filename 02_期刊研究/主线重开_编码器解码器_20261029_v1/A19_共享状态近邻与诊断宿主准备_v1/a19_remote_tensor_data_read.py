from pathlib import Path
import json,torch,hashlib
class MetadataCfgNode(dict):
    pass
p=Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1/vssd_tiny_host_assets_20261030_v1/vssd_tiny_mesa.pth')
with torch.serialization.safe_globals([(MetadataCfgNode,'yacs.config.CfgNode')]):
    ck=torch.load(p,map_location='cpu',weights_only=True)
assert isinstance(ck,dict)
groups={}
for k,v in ck.items():
    if isinstance(v,dict):
        ts={n:t for n,t in v.items() if isinstance(t,torch.Tensor)}
        if ts:
            groups[k]={'tensor_keys':len(ts),'tensor_numel_including_buffers':sum(t.numel() for t in ts.values()),'head_shapes':{n:list(t.shape) for n,t in ts.items() if n in ['head.weight','head.bias','norm.weight','patch_embed.proj.0.weight']}}
        elif 'module' in v and isinstance(v['module'],dict):
            ts={n:t for n,t in v['module'].items() if isinstance(t,torch.Tensor)}
            if ts: groups[k+'.module']={'tensor_keys':len(ts),'tensor_numel_including_buffers':sum(t.numel() for t in ts.values()),'head_shapes':{n:list(t.shape) for n,t in ts.items() if n in ['head.weight','head.bias','norm.weight','patch_embed.proj.0.weight']}}
assert 'model' in groups or 'model_ema' in groups or 'model_ema.module' in groups
r={'status':'CPU_TENSOR_DATA_LOAD_OK_NOT_MODEL_STRICT_LOAD','weights_only':True,'config_metadata_mapping':'yacs.config.CfgNode -> inert local dict subclass with no custom behaviour; unused for model construction','top_level_keys':list(ck),'tensor_groups':groups,'model_instantiated':False,'model_forward_calls':0,'optimizer_updates':0}
(p.parent/'A19_remote_weight_tensor_data.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r,ensure_ascii=False))

