"""A native target-interpolation counterexample; no detector construction/calls."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE.parent/'code/dn_pre_frozen_diagnostic_20261020_v1/runtime'))
import torch
source=BASE/'A43_ECDet三组蒸馏最小验证_20261031_v1/source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg/engine/edgecrafter/utils.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['weighting_function','translate_gt']]
assert len(nodes)==2
namespace={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
up=torch.tensor([.5],dtype=torch.float32);reg_scale=4.;reg_max=32
reg_scale_tensor=torch.tensor([reg_scale],dtype=torch.float32)
project=namespace['weighting_function'](reg_max,up,reg_scale_tensor)
reference_width_pixels=100.;target_width_pixels=6.;target_center_offset_pixels=20.
gt_left=target_center_offset_pixels-target_width_pixels/2
gt_right=target_center_offset_pixels+target_width_pixels/2
targets=torch.tensor([(-gt_left/reference_width_pixels-.5)*reg_scale,
                      (gt_right/reference_width_pixels-.5)*reg_scale],dtype=torch.float32)
idx,right,left=namespace['translate_gt'](targets,reg_max,reg_scale_tensor,up)
prob=[]
for j in range(2):
    p=torch.zeros(len(project),dtype=torch.float64);i=int(idx[j]);p[i]=left[j];p[i+1]=right[j]
    prob.append(p/p.sum())
pl,pr=prob
native_width=reference_width_pixels*(1+((pl+pr)*project).sum()/reg_scale)
joint=pl[:,None]*pr[None,:]
pair_width=reference_width_pixels*(1+(project[:,None]+project[None,:])/reg_scale)
valid=pair_width>0
bad=float(joint[~valid].sum());conditional=joint*valid;conditional/=conditional.sum()
conditioned_width=float((conditional*pair_width).sum())
assert abs(float(native_width)-target_width_pixels)<1e-5
assert bad>0 and abs(conditioned_width-target_width_pixels)>1e-3
result={'status':'COUNTEREXAMPLE_FOR_NAIVE_ORDER_CONDITIONING_WITH_UNCHANGED_NATIVE_TARGETS',
 'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'reading_scope':'Pinned native weighting_function and translate_gt only.',
 'reg_max':reg_max,'reg_scale':reg_scale,'up':.5,
 'toy_not_dataset_observation':True,'reference_width_pixels':reference_width_pixels,'target_width_pixels':target_width_pixels,
 'target_center_offset_pixels':target_center_offset_pixels,'GT_left_right_pixels_relative_to_reference_center':[gt_left,gt_right],
 'native_interpolated_target_width_pixels':float(native_width),'native_target_nonzero_project_values':[project[p>0].tolist() for p in prob],
 'native_target_nonzero_probabilities':[p[p>0].tolist() for p in prob],'independent_product_nonpositive_width_mass':bad,
 'positive_order_conditioned_mean_width_pixels':conditioned_width,
 'conditioned_minus_GT_width_pixels':conditioned_width-target_width_pixels,
 'conclusion':'A valid GT with a nearly exact native interpolation mean can still produce invalid combinations in a fictitious independent product. Removing that product mass changes the otherwise correct mean. Marginals are not a calibrated joint box posterior.',
 'decision':'Do not select unchanged-native-targets plus naive product-order-conditioned readout as the next candidate. A different joint-target/readout method would require its own substantive proposal, not a rescue sweep.',
 'new_model_forwards':0,'new_optimizer_updates':0,'new_COCO_evaluate_calls':0,
 'development_note':'Native function calls require a one-element reg_scale tensor and FP32 inputs because translate_gt creates FP32 target weights. These interface issues were corrected. A symmetric centered fixture did not produce the proposed strict counterexample because of the zero-width boundary; the reported counterexample is the valid off-center target specified above, not a dataset estimate. No model/evaluator was called in these development checks.',
 'scope_limit':'No estimate of how often this occurs in trained predictions; no claim that every constrained joint model is ineffective.'}
out=ROOT/'08_合法性条件读出_原生插值反例_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
