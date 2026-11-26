from pathlib import Path
import ast,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path.cwd();HERE=ROOT/'英文小论文/code/native_fp_query_trace12_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
ast.parse((HERE/'analyze.py').read_text(encoding='utf8'))
s=importlib.util.spec_from_file_location('query_trace_fixture',HERE/'analyze.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
cases=[('initial_support',[.7,.2,.8],1),('pre_support',[.2,.7,.8],2),('final_support',[.2,.3,.8],3),
       ('initial_boundary',[.5,.7,.8],4),('pre_boundary',[.2,.5,.8],4),('guaranteed_initial_despite_other_near',[.7,.2,.8],1)]
proof=[]
for name,values,expected in cases:
    data=[np.array([x]) for x in values]
    if name=='guaranteed_initial_despite_other_near':data[0]=np.array([.7,.5])
    assert m.first_support(data,.5,1e-5)==expected,name
    proof.append(dict(name=name,passed=True))
old=read(ROOT/'英文小论文/code/native_fp_geometry12_20261011_v1/protocol.json')
deps={n:sha(ROOT/n) for n in ('英文小论文/code/native_fp_geometry12_20261011_v1/analyze.py',
    '英文小论文/code/native_fp_geometry12_20261011_v1/protocol.json',
    '英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py',
    '英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json')}
for seed in (1,2):
    for tag in ('BASE12','Warm12'):
        folder=ROOT/f'outputs/native_fp_geometry12_20261011_v1/seed{seed}_{tag}'
        for n,h in read(folder/'manifest.json')['files'].items():assert sha(folder/n)==h,n
        deps[(folder/'manifest.json').relative_to(ROOT).as_posix()]=sha(folder/'manifest.json')
protocol=dict(schema='native_FP_within_run_query_trace12_v1',analysis_sha256=sha(HERE/'analyze.py'),dependencies_sha256=deps,
    score_abs_tolerance=old['score_abs_tolerance'],IoU_boundary_epsilon=old['IoU_boundary_epsilon'],
    first_support_labels=list(m.STAGE_LABELS),same_query_native_other_category_TP_witness_only=True,
    no_cross_run_query_identity=True,no_NATIVE_matches_recomputed=True,new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)
with (HERE/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
assert 'torch' not in sys.modules
with (ROOT/'outputs/native_fp_query_trace12_20261011_v1_CPU_review.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS_SAME_GT_FIRST_SUPPORT_BOUNDARY_FIXTURES',fixtures=proof,scientific_trace_runs=0,source_sha256=sha(__file__),analysis_sha256=sha(HERE/'analyze.py'),protocol_sha256=sha(HERE/'protocol.json')),f,indent=2)
print(json.dumps(dict(status='NATIVE_FP_QUERY_TRACE_RULES_FROZEN',cases=len(proof),scientific_trace_runs=0)))
