"""Fix overlap categories, tolerance and conservative uncertainty rules before reading outcomes."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path.cwd();code=ROOT/'英文小论文/code/native_fp_geometry12_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ast.parse((code/'analyze.py').read_text(encoding='utf8'))
spec=importlib.util.spec_from_file_location('fp_geometry_fixtures',code/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=[
    ('invalid_box',[0.9,0.0],[0,1],[10,11],set(),False,1),
    ('eligible_own_GT_already_used',[0.7,0.0],[0,1],[10,11],{10},True,2),
    ('eligible_own_GT_not_used',[0.7,0.0],[0,1],[10,11],set(),True,3),
    ('other_class_overlap',[0.2,0.7],[0,1],[10,11],set(),True,4),
    ('insufficient_any_class_overlap',[0.2,0.4],[0,1],[10,11],set(),True,5),
    ('low_overlap_all_GT',[0.02,0.05],[0,1],[10,11],set(),True,6),
    ('own_IoU_boundary',[0.5,0.1],[0,1],[10,11],set(),True,7),
    ('other_IoU_boundary',[0.2,0.5],[0,1],[10,11],set(),True,7),
    ('one_used_one_unused_own_GT',[0.7,0.6],[0,0],[10,11],{10},True,3),
    ('all_eligible_own_GT_used',[0.7,0.6],[0,0],[10,11],{10,11},True,2),
    ('no_GT',[],[],[],set(),True,6),
    ('low_overlap_boundary',[0.1,0.0],[0,1],[10,11],set(),True,7),
]
proof=[]
for name,ious,cats,ids,used,valid,expected in cases:
    got=m.fp_label(np.array(ious),np.array(cats),np.array(ids),0,used,.5,1e-5,valid)
    assert got==expected,name
    reverse=m.fp_label(np.array(ious)[::-1],np.array(cats)[::-1],np.array(ids)[::-1],0,used,.5,1e-5,valid)
    assert reverse==expected,name
    proof.append(dict(name=name,expected=m.LABELS[expected],passed=True,GT_order_invariant=True))
# Score ties that lead to different categories are never resolved by selecting the convenient box.
labels={m.fp_label(np.array([v]),np.array([0]),np.array([10]),0,{10},.5,1e-5) for v in (.7,.02)}
assert labels=={2,6} and len(labels)>1
proof.append(dict(name='multiple_close_scores_with_conflicting_geometries_remain_ambiguous',passed=True))
ranking=read(ROOT/'英文小论文/code/stem_init_stage_analysis_20261011_v2/ranking_protocol.json')
deps={n:sha(ROOT/n) for n in (
    '英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py',
    '英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json',
    '英文小论文/code/stem_init_stage_analysis_20261011_v2/ranking_protocol.json',
    '英文小论文/code/scale_protection12_monitor_20261010_v1/geometry_core.py',
    '英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py',
    'outputs/warm_stem_both_seeds_review12_20261011_v1/report.json')}
assert deps['英文小论文/code/scale_protection12_monitor_20261010_v1/geometry_core.py']==ranking['geometry_code_sha256']
protocol=dict(schema='native_all_area_FP_geometry12_v1',analysis_sha256=sha(code/'analyze.py'),dependencies_sha256=deps,
    score_abs_tolerance=ranking['native_score_abs_tolerance'],IoU_boundary_epsilon=1e-5,
    low_overlap_boundary=.1,labels=list(m.LABELS),prefixes=list(m.PREFIXES),all10_classes_and10_IoUs_required=True,
    no_NATIVE_matches_recomputed=True,all300_near_score_queries_considered=True,
    independent_saved_EMA_replay_not_substituted=True,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
with (code/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2,allow_nan=False)
result=dict(status='PASS_CONSERVATIVE_NATIVE_FP_OVERLAP_RULES_AND_GT_ORDER_FIXTURES',fixtures=proof,
    actual_scientific_runs=0,source_sha256=sha(__file__),analysis_sha256=sha(code/'analyze.py'),protocol_sha256=sha(code/'protocol.json'),
    no_model_or_evaluator_import=True,new_model_forwards=0,new_BASE_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
assert 'torch' not in sys.modules and not any(x.startswith('pycocoeval') for x in sys.modules)
with (ROOT/'outputs/native_fp_geometry12_20261011_v1_CPU_review.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=result['status'],cases=len(proof),scientific_runs=0)))
