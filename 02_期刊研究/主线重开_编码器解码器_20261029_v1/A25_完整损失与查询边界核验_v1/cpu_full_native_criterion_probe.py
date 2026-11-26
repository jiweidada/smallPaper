"""Synthetic CPU calls to the complete pinned native criterion, no detector."""
import ast
import copy
import datetime
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types
import torch

AP = Path('/root/c10_region_restore_20261029_v1/architecture_probe_v1')
spec = importlib.util.spec_from_file_location('a25_source_loader', AP / 'dome_assets_v1/dome_probe_loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
loader.verified_source()
sys.meta_path.insert(0, loader.AuthorFinder())
for name, rel in [('src','src'),('src.core','src/core'),('src.nn','src/nn'),
                  ('src.zoo','src/zoo'),('src.zoo.dome','src/zoo/dome'),('src.misc','src/misc')]:
    loader.namespace(name, loader.SOURCE / rel)
workspace = importlib.import_module('src.core.workspace')
sys.modules['src.core'].register = workspace.register
yaml_utils = importlib.import_module('src.core.yaml_utils')
dist_path = loader.SOURCE / 'src/misc/dist_utils.py'
tree = ast.parse(dist_path.read_text(encoding='utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['is_dist_available_and_initialized','get_world_size']]
assert len(functions) == 2
dist_module = types.ModuleType('src.misc.dist_utils')
dist_module.torch = torch
sys.modules['src.misc.dist_utils'] = dist_module
exec(compile(ast.Module(body=functions, type_ignores=[]), str(dist_path), 'exec'), dist_module.__dict__)
importlib.import_module('src.zoo.dome.matcher')
native = importlib.import_module('src.zoo.dome.dome_criterion')
cfg = yaml_utils.load_config(str(loader.SOURCE / 'configs/dome/Dome-S-VisDrone.yml'))
global_cfg = yaml_utils.merge_config(cfg, inplace=False, overwrite=False)
source_path = Path(native.__file__)
source = source_path.read_text(encoding='utf-8')
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
assert source_hash == '357e8997241ca12ca146ccaab8d51df28ed563ae5bac53797825ab697b3474bc'
tree = ast.parse(source)
klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DomeCriterion')
method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'loss_local')
patched = copy.deepcopy(method)
hits = [node for node in ast.walk(patched) if isinstance(node, ast.BinOp) and ast.unparse(node) == 'pred_corners.sum() * 0']
assert len(hits) == 1
hits[0].left.keywords.append(ast.keyword(arg='dtype',value=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='float32',ctx=ast.Load())))
ast.fix_missing_locations(patched)
namespace = dict(native.__dict__)
exec(compile(ast.Module(body=[patched],type_ignores=[]),str(source_path),'exec'),namespace)
stable_method = namespace['loss_local']
assert not torch.cuda.is_initialized()
matcher_calls = [0]

def build_criterion(stable=False):
    criterion = workspace.create('DomeCriterion', global_cfg)
    criterion.matcher.register_forward_pre_hook(lambda *_: matcher_calls.__setitem__(0,matcher_calls[0]+1))
    if stable:
        criterion.loss_local = types.MethodType(stable_method,criterion)
    return criterion

def fixture(corner_dtype):
    batch,queries,classes = 2,500,12
    box = torch.tensor([0.5,0.5,0.05,0.05])
    logits = torch.zeros((batch,queries,classes),dtype=torch.float32,requires_grad=True)
    boxes = box.repeat(batch,queries,1).requires_grad_()
    corners = torch.ones((batch,queries,132),dtype=corner_dtype,requires_grad=True)
    refs = box.repeat(batch,queries,1)
    outputs = {'pred_logits':logits,'pred_boxes':boxes,'pred_corners':corners,
               'ref_points':refs,'up':torch.tensor([0.5]),'reg_scale':torch.tensor([4.0])}
    outputs['aux_outputs'] = [
        {'pred_logits':logits,'pred_boxes':boxes,'pred_corners':corners,'ref_points':refs,
         'teacher_corners':corners.detach(),'teacher_logits':logits.detach()} for _ in range(2)]
    outputs['pre_outputs'] = {'pred_logits':logits,'pred_boxes':boxes}
    outputs['enc_aux_outputs'] = [{'pred_logits':logits,'pred_boxes':boxes}]
    outputs['enc_meta'] = {'class_agnostic':False}
    outputs['batch_queries_num'] = [queries]*batch
    targets = [{'labels':torch.tensor([1]),'boxes':box[None,:].clone()} for _ in range(batch)]
    return outputs,targets,[logits,boxes,corners]

results = []
for name,dtype,stable in [('native_FP16',torch.float16,False),
                          ('stable_zero_only_FP16',torch.float16,True),
                          ('native_FP32',torch.float32,False)]:
    criterion = build_criterion(stable)
    outputs,targets,leaves = fixture(dtype)
    raw = []
    original = torch.nan_to_num
    def observe(value,*args,**kwargs):
        caller = inspect.currentframe().f_back
        if caller.f_code.co_filename == str(source_path) and 'k' in caller.f_locals:
            raw.append({'key':caller.f_locals['k'],'dtype':str(value.dtype),
                        'finite':bool(torch.isfinite(value).all()),
                        'value_text':str(float(value.detach()))})
        return original(value,*args,**kwargs)
    torch.nan_to_num = observe
    try:
        losses = criterion(outputs,targets,epoch=0)
    finally:
        torch.nan_to_num = original
    assert raw and all(value.ndim == 0 for value in losses.values())
    bad = [row['key'] for row in raw if not row['finite']]
    gradients = None
    if stable or dtype == torch.float32:
        assert not bad
        sum(losses.values()).backward()
        gradients = {'all_present':all(value.grad is not None for value in leaves),
                     'all_finite':all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in leaves)}
        assert gradients['all_present'] and gradients['all_finite']
    results.append({'case':name,'corner_dtype':str(dtype),'complete_native_criterion_called':True,
                    'only_engineering_method_change': 'FP32 accumulator for algebraically zero DDF branch' if stable else None,
                    'raw_loss_values_before_sanitation':raw,'nonfinite_raw_loss_keys':bad,
                    'native_returned_losses_all_finite':all(bool(torch.isfinite(v)) for v in losses.values()),
                    'synthetic_input_gradients':gradients})
assert results[0]['nonfinite_raw_loss_keys']
assert all('ddf' in key for key in results[0]['nonfinite_raw_loss_keys'])

# A separate synthetic set-boundary check, not evidence of real-image prevalence.
criterion = build_criterion()
target = [{'labels':torch.tensor([1]),'boxes':torch.tensor([[0.5,0.5,0.02,0.02]])}]
padded = {'pred_logits':torch.zeros((1,2,12)),
          'pred_boxes':torch.tensor([[[0.05,0.05,0.02,0.02],[0.5,0.5,0.5,0.5]]]),
          'batch_queries_num':[1]}
idx = criterion.matcher(padded,target)['indices'][0][0].tolist()
truncated = {'pred_logits':padded['pred_logits'][:,:1],
             'pred_boxes':padded['pred_boxes'][:,:1],'batch_queries_num':[1]}
control_idx = criterion.matcher(truncated,target)['indices'][0][0].tolist()
assert idx == [1] and control_idx == [0]
assert not torch.cuda.is_initialized()
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
result = {
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'COMPLETE_CRITERION_SYNTHETIC_RAW_DDF_FAILURE_AND_ZERO_FIX_VERIFIED',
    'device':'cpu','CUDA_initialized':False,'torch_version':torch.__version__,
    'source':str(source_path),'source_sha256':source_hash,
    'criterion_native_class':type(criterion).__name__,
    'criterion_losses':criterion.losses,'criterion_weights':criterion.weight_dict,
    'complete_native_criterion_forward_calls':3,'native_matcher_forward_calls':matcher_calls[0],
    'detector_forwards':0,'optimizer_updates':0,'real_images':0,'trained_weights_loaded':False,
    'cases':results,
    'query_boundary_fixture':{'valid_query_count':1,'padded_query_count':1,
        'native_matching_selected_indices':idx,'truncated_control_selected_indices':control_idx,
        'evidence_scope':'Native matcher can select a padded row despite batch_queries_num metadata on this fixture; not actual frequency or AP harm.'},
    'limitations':['No replay of A22 lost trained state or actual failing loss.',
                   'No claim that a zero-expression fix stabilizes every loss branch.',
                   'No detector accuracy or algorithm novelty evidence.',
                   'This fixture omits DN and density-map outputs; it executes the full criterion forward for the supplied branch set.'],
    'frozen_A22_modified':False,'training_jobs_launched':0,
}
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
