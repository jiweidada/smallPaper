"""Use the same within-run trace on the two actual original-user FP outputs."""
from pathlib import Path
import ast,difflib,hashlib,json
ROOT=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=ROOT/'英文小论文/code/native_fp_query_trace12_20261011_v1';new=ROOT/'英文小论文/code/native_fp_query_trace12_20261011_v2';new.mkdir(exist_ok=False)
text=(old/'analyze.py').read_text(encoding='utf8').replace("assert seed in (1,2) and tag in ('BASE12','Warm12')","assert seed==2 and tag in ('backbone_silu','p2_skip')")
text=text.replace('native_fp_geometry12_20261011_v1','native_fp_geometry12_20261011_v2').replace('outputs/native_fp_query_trace12_20261011_v1/','outputs/native_fp_query_trace12_20261011_v2/').replace("choices=('BASE12','Warm12')","choices=('backbone_silu','p2_skip')")
ast.parse(text);(new/'analyze.py').write_text(text,encoding='utf8')
spec=json.loads((old/'protocol.json').read_bytes());spec['schema']='actual_user_native_FP_within_run_query_trace12_v2';spec['analysis_sha256']=sha(new/'analyze.py')
spec['dependencies_sha256']={n:h for n,h in spec['dependencies_sha256'].items() if 'native_fp_geometry' not in n}
for n in ('英文小论文/code/native_fp_geometry12_20261011_v2/analyze.py','英文小论文/code/native_fp_geometry12_20261011_v2/protocol.json','outputs/native_fp_geometry12_20261011_v2/seed2_backbone_silu/manifest.json','outputs/native_fp_geometry12_20261011_v2/seed2_p2_skip/manifest.json'):
    spec['dependencies_sha256'][n]=sha(ROOT/n)
spec['permitted_actual_runs']=[dict(seed=2,tag=t) for t in ('backbone_silu','p2_skip')]
(new/'protocol.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
(new/'adaptation.diff').write_text(''.join(difflib.unified_diff((old/'analyze.py').read_text(encoding='utf8').splitlines(True),text.splitlines(True),fromfile='v1/analyze.py',tofile='v2/analyze.py')),encoding='utf8')
def function(src,name):return ast.dump(next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
assert function(text,'first_support')==function((old/'analyze.py').read_text(encoding='utf8'),'first_support')
with (ROOT/'outputs/user_native_fp_query_trace12_20261011_v2_source_review.json').open('x',encoding='utf8') as f:json.dump(dict(status='TWO_ACTUAL_USER_FP_MANIFESTS_BOUND_SAME_TRACE_RULES',files={p.name:sha(p) for p in new.iterdir() if p.is_file()},permitted_actual_runs=spec['permitted_actual_runs'],new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0),f,indent=2)
print('PASS_ACTUAL_USER_FP_BOUND_SAME_TRACE_RULES')
