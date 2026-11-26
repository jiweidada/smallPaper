"""Bind all available original-user seed2 endpoints, with margin engineering fixtures."""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
ROOT=Path.cwd();code=ROOT/'英文小论文/code/user_class_margin_geometry12_20261011_v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ast.parse((code/'analyze.py').read_text(encoding='utf8'))
names=['英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py','英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json','英文小论文/code/user_backbone_primary_collection12_20261011_v2/verify_primary.py']
for arm in ('backbone_silu','p2_skip','backbone_attention'):
    names+=[f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{arm}_v1/primary_verification_v1.json',f'outputs/user_backbone_primary_analysis12_20261011_v2/seed2_{arm}/manifest.json']
spec=dict(schema='user_seed2_geometry_conditioned_relative_class_observation_v1',analysis_sha256=sha(code/'analyze.py'),dependencies_sha256={n:sha(ROOT/n) for n in names},run_tags=['BASE12','backbone_silu','p2_skip','backbone_attention'],IoU_thresholds=[.5,.75],class_margin_epsilon=1e-5,geometry_source='unchanged bound float32 native-coordinate transform and pairwiseIoU',training_allowed=False,APs_causal_decomposition_allowed=False,all10_categories_required=True,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
with (code/'protocol.json').open('x',encoding='utf8') as f:json.dump(spec,f,indent=2)
s=importlib.util.spec_from_file_location('margin_engineering',code/'analyze.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
logits=np.array([[3.,1.,-2.],[1.,4.,0.],[-1.,-1.,-2.]])
got=m.relative_margin(logits,np.array([0,1,2]))
expected=np.array([[2.,-3.,0.],[-2.,3.,0.],[-5.,-4.,-1.]])
assert np.array_equal(got,expected)
# A same-query scalar shifts absolute scores but cannot alter its class margins.
assert np.array_equal(m.relative_margin(logits+np.array([7.,-5.,2.])[:,None],np.array([0,1,2])),expected)
# Correct relative class is not necessary for retaining a true-class multilabel entry.
prob=1/(1+np.exp(-np.array([4.,3.,-1.])));assert prob[1]>.9 and np.argmax(prob)!=1
review=dict(status='PASS_MARGIN_ENGINEERING_SCALAR_INVARIANCE_AND_MULTILABEL_COUNTEREXAMPLE',dependency_files_verified=len(names),files={p.name:sha(p) for p in code.iterdir() if p.is_file()},actual_scientific_analysis_pending=True,root_or_AP_claim=False)
with (ROOT/'outputs/user_class_margin_geometry12_20261011_v1_source_review.json').open('x',encoding='utf8') as f:json.dump(review,f,indent=2)
print(review['status'])
