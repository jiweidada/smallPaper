from __future__ import annotations
import hashlib,json,pathlib,sys,tarfile,time

R=pathlib.Path(__file__).resolve().parent
PIN='d8752b75c9beaecfcdd0e5cd207bae3b6946ae23'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(name,x):
    (R/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
    print(json.dumps(x),flush=True)

assert digest(R/'edgecrafter_pinned_source.tar.gz')=='c39629b2d3cb51f8016315ab7c44803b0edff5c312575a2bf48fca5391c988e7'
assert digest(R/'ecdet_s_coco_official.pth')=='35340a45101d24a094becf8776aeaa327c9e6b37453e5e81dfd2923c134c4619'
dest=R/'source'
dest.mkdir(exist_ok=False)
with tarfile.open(R/'edgecrafter_pinned_source.tar.gz','r:gz') as tf:
    for member in tf.getmembers():
        p=pathlib.Path(member.name)
        assert not p.is_absolute() and '..' not in p.parts
        if member.isdir():continue
        assert member.isfile()
        target=dest/p
        target.parent.mkdir(parents=True,exist_ok=True)
        with tf.extractfile(member) as src:target.write_bytes(src.read())
source=dest/('EdgeCrafter-'+PIN)/'ecdetseg'
ecvit=source/'engine/edgecrafter/ecvit.py'
before=digest(ecvit)
ecvit.write_text('from __future__ import annotations\n'+ecvit.read_text(),encoding='utf-8')
sys.path.insert(0,str(source))
import torch
from engine.core import YAMLConfig
torch.set_num_threads(4)
torch.manual_seed(20261031)
cfg=YAMLConfig(str(source/'configs/ecdet/ecdet_s.yml'))
cfg.yaml_cfg['num_classes']=80
cfg.yaml_cfg['remap_mscoco_category']=False
cfg.yaml_cfg['ViTAdapter']['skip_load_backbone']=True
model=cfg.model.cpu().eval()
obj=torch.load(R/'ecdet_s_coco_official.pth',map_location='cpu',weights_only=True)
if isinstance(obj,dict) and isinstance(obj.get('ema'),dict) and 'module' in obj['ema']:
    state=obj['ema']['module']; selection='ema.module'
elif isinstance(obj,dict) and isinstance(obj.get('model'),dict):state=obj['model'];selection='model'
elif isinstance(obj,dict) and all(isinstance(v,torch.Tensor) for v in obj.values()):state=obj;selection='root_state_dict'
else:raise RuntimeError('Unsupported safe checkpoint structure: '+str(type(obj)))
model.load_state_dict(state,strict=True)
assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
emit('strict_official_80class_load.json',{'status':'PASS','checkpoint_selection':selection,'checkpoint_top_keys':list(obj)[:12],'state_keys':len(state),'parameter_count':sum(p.numel() for p in model.parameters()),'all_float_weights_finite':True,'device':'cpu','model_forwards':0,'source_root':str(source),'python':sys.version,'torch':torch.__version__,'ecvit_before_future_annotation_sha256':before,'ecvit_after_sha256':digest(ecvit)})
del model,cfg
torch.manual_seed(20261031)
cfg=YAMLConfig(str(source/'configs/ecdet/ecdet_s.yml'))
cfg.yaml_cfg['num_classes']=10
cfg.yaml_cfg['remap_mscoco_category']=False
cfg.yaml_cfg['ViTAdapter']['skip_load_backbone']=True
model=cfg.model.cpu()
own=model.state_dict()
mismatch=[k for k,v in own.items() if k in state and v.shape!=state[k].shape]
assert mismatch and all('score_head' in k or 'denoising_class_embed' in k for k in mismatch),mismatch
loadable={k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
res=model.load_state_dict(loadable,strict=False)
assert set(res.missing_keys)==set(mismatch) and not res.unexpected_keys
out=R/'common_init_10class.pth'
torch.save({'model':model.state_dict(),'seed':20261031,'source_checkpoint_sha256':digest(R/'ecdet_s_coco_official.pth'),'new_class_parameters':mismatch},out)
base=pathlib.Path('/root/c10_region_restore_20261029_v1')
vp=base/'architecture_probe_v1/a22_vssd_dome_pilot12_20261030_v1/val_inputs'
emit('common_initialization_and_paths.json',{'status':'PASS','file':str(out),'sha256':digest(out),'num_classes':10,'new_class_parameters':mismatch,'inherited_keys':len(loadable),'parameters':sum(p.numel() for p in model.parameters()),'criterion_type':type(cfg.criterion).__name__,'postprocessor_type':type(cfg.postprocessor).__name__,'train_images':str(base/'dataset/train/images'),'train_annotations':str(base/'dataset/train/annotations'),'val_inputs_children':[p.name for p in vp.iterdir()],'model_forwards':0,'optimizer_updates':0})
