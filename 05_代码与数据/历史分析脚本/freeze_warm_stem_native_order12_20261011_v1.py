from pathlib import Path
import ast,hashlib,json
root=Path('英文小论文/code/warm_stem_native_order12_20261011_v1');s=root/'analyze.py';ast.parse(s.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dependencies=['英文小论文/code/stem_native_rank_primary12_20261011_v1/analyze.py','outputs/stem_native_rank_primary12_20261011_v1/report.json',
'英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py','英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json',
'英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py']
protocol=dict(schema='warm_native_order12_v1',analysis_sha256=sha(s),dependencies_sha256={n:sha(n) for n in dependencies},
 prefixes=[100,500,1000,2000,5000,10000,20000],evaluator_calls=0,model_forwards=0,
 scope='Exact existing all-area native evaluation accounting; small-GT subsets remain all-area matching. No APs raw-score-prefix proof or causal root claim.')
with (root/'protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,indent=2)
print(json.dumps({'status':'NATIVE_ORDER_ANALYSIS_FROZEN','source_sha256':sha(s)}))
