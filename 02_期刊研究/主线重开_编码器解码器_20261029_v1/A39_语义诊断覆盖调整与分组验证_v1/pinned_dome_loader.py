"""Author model-only load with non-numerical import compatibility changes.

Author files remain byte-identical. Training/data package initializers are not
executed. Active constructor/layer arithmetic and the original registry/YAML are
used. No detector forward is performed by this module's command-line entry point.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import __future__
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import types
import zipfile
import torch

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
IMPORTS=[]
os.environ['SAVE_INTERMEDIATE_VISUALIZE_RESULT']='False'
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(ROOT/'mplconfig')

def hash_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verified_source():
    manifest=json.loads((ROOT/'asset_verification.json').read_text(encoding='utf-8'))
    assert manifest['source_commit']=='2dde3bc1946a3e9fad9abd0612b59fc39bd6b861'
    assert hash_file(ROOT/'author_source.zip')==manifest['source_zip_sha256']
    assert hash_file(ROOT/'dome-s-visdrone_converted.pth')==manifest['weight']['sha256']
    prefix='Dome-DETR-'+manifest['source_commit']+'/'
    with zipfile.ZipFile(ROOT/'author_source.zip') as z:
        for row in manifest['source_files']:
            rel=row['path'];p=PurePosixPath(rel)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in rel
            dest=(SOURCE/rel).resolve()
            assert dest.is_relative_to(SOURCE.resolve())
            if not dest.exists():
                data=z.read(prefix+rel)
                assert hashlib.sha256(data).hexdigest()==row['sha256']
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            assert hash_file(dest)==row['sha256'],rel
    assert len(manifest['source_files'])==293
    return manifest

class AuthorLoader(importlib.abc.Loader):
    def __init__(self,path):self.path=path
    def create_module(self,spec):return None
    def exec_module(self,module):
        data=self.path.read_bytes();tree=ast.parse(data,filename=str(self.path))
        changes=['deferred_annotations_for_python39_only']
        if self.path==SOURCE/'src/zoo/dome/get_roi_features.py':
            unused={'label','regionprops'}
            assert not any(isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load) and x.id in unused for x in ast.walk(tree))
            before=len(tree.body)
            tree.body=[x for x in tree.body if not (isinstance(x,ast.ImportFrom) and x.module=='skimage.measure')]
            assert len(tree.body)==before-1
            changes.append('omit_provably_unused_skimage_measure_import_only')
        module.__file__=str(self.path)
        IMPORTS.append({'path':str(self.path.relative_to(SOURCE)),
                        'source_sha256':hashlib.sha256(data).hexdigest(),'changes':changes})
        exec(compile(tree,str(self.path),'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),module.__dict__)

class AuthorFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if not (fullname.startswith('src.') or fullname.startswith('tools.')):return None
        stem=SOURCE.joinpath(*fullname.split('.'))
        if stem.is_dir() and (stem/'__init__.py').exists():
            return importlib.util.spec_from_file_location(fullname,stem/'__init__.py',
                 loader=AuthorLoader(stem/'__init__.py'),submodule_search_locations=[str(stem)])
        p=stem.with_suffix('.py')
        if p.is_file():return importlib.util.spec_from_file_location(fullname,p,loader=AuthorLoader(p))
        return None

def namespace(name,path):
    module=types.ModuleType(name);module.__package__=name;module.__path__=[str(path)]
    sys.modules[name]=module
    return module

def state_digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        v=value.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(v.dtype).encode()+b'\0')
        h.update(str(tuple(v.shape)).encode()+b'\0'+v.numpy().tobytes())
    return h.hexdigest()

def input_transform():
    """Use the author ConvertPILImage definition and the exact T.Resize alias."""
    import PIL
    import PIL.Image
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as F
    from torchvision.tv_tensors import Image
    source=SOURCE/'src/data/transforms/_transforms.py'
    data=source.read_bytes();tree=ast.parse(data,filename=str(source))
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ConvertPILImage')
    node.decorator_list=[]  # registry side effect only; class operations unchanged.
    scope={'PIL':PIL,'torch':torch,'T':T,'F':F,'Image':Image}
    body=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(body,str(source),'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),scope)
    resize=T.Resize([800,800])
    convert=scope['ConvertPILImage'](dtype='float32',scale=True)
    receipt={'source_sha256':hashlib.sha256(data).hexdigest(),
             'author_class':'ConvertPILImage','registry_decorator_only_omitted':True,
             'resize_alias_source_line':33,'resize':repr(resize),'normalize':None,
             'dtype':'float32','scale':True}
    def transform(image):
        value=convert(resize(image))
        return value.as_subclass(torch.Tensor).unsqueeze(0)
    return transform,receipt

def load_model(receipt_path=None):
    manifest=verified_source()
    sys.meta_path.insert(0,AuthorFinder())
    for name,rel in [('src','src'),('src.core','src/core'),('src.nn','src/nn'),
      ('src.nn.backbone','src/nn/backbone'),('src.zoo','src/zoo'),('src.zoo.dome','src/zoo/dome'),('tools','tools')]:
        namespace(name,SOURCE/rel)
    core=sys.modules['src.core']
    workspace=importlib.import_module('src.core.workspace')
    core.register=workspace.register
    yu=importlib.import_module('src.core.yaml_utils')
    for module in ['src.nn.backbone.hgnetv2','src.zoo.dome.hybrid_encoder',
                   'src.zoo.dome.dome_decoder','src.zoo.dome.dome','src.zoo.dome.postprocessor']:
        importlib.import_module(module)
    cfg=yu.load_config(str(SOURCE/'configs/dome/Dome-S-VisDrone.yml'))
    assert cfg['num_classes']==12 and cfg['eval_spatial_size']==[800,800]
    assert cfg['HGNetv2']['name']=='B0'
    assert cfg['HybridEncoder']['use_deformable'] is False
    assert cfg['HybridEncoder']['mwas_window_size']==10
    cfg['HGNetv2']['pretrained']=False  # all state tensors strictly overwritten below.
    global_cfg=yu.merge_config(cfg,inplace=False,overwrite=False)
    model=workspace.create(cfg['model'],global_cfg)
    postprocessor=workspace.create(cfg['postprocessor'],global_cfg)
    checkpoint=torch.load(ROOT/'dome-s-visdrone_converted.pth',map_location='cpu',weights_only=True)
    state=checkpoint['model']
    assert state['decoder.enc_score_head.weight'].shape==(12,256)
    assert len(state)==1033
    incompatible=model.load_state_dict(state,strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.eval();postprocessor.eval()
    assert model.decoder.num_classes==12 and len(model.decoder.decoder.layers)==3
    assert model.encoder.out_strides==[4,8,16,32]
    assert model.decoder.min_num_select==250 and model.decoder.max_num_select==500
    for name,value in model.state_dict().items():assert torch.equal(value.cpu(),state[name].cpu()),name
    transform,preprocess=input_transform()
    receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'STRICT_AUTHOR_DOME_S_VISDRONE_MODEL_LOADED_CPU',
      'source_commit':manifest['source_commit'],'weight_sha256':manifest['weight']['sha256'],
      'state_keys':len(state),'parameters':sum(p.numel() for p in model.parameters()),
      'state_digest':state_digest(model),'all_state_tensor_values_equal_checkpoint':True,
      'num_classes':12,'valid_VisDrone_category_ids':list(range(1,11)),
      'query_count_range':[250,500],'decoder_layers':3,'preprocessing':preprocess,
      'author_postprocess':'Flat TopQ over class sigmoid scores, Q=raw query count; no fixed Top300 override.',
      'imports':IMPORTS,'config':cfg,'initialization_override':'HGNetv2 pretrained=False; strict full detector state overwrites all initialization.',
      'numerical_source_edits':0,'deploy_conversion':False,'new_full_detector_forwards':0,'main_detector_updates':0}
    Path(receipt_path or ROOT/'strict_model_load.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    model.requires_grad_(False)
    return model,postprocessor,transform,receipt

if __name__=='__main__':
    _,_,_,receipt=load_model()
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['imports','config']},ensure_ascii=False))
