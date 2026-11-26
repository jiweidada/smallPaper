from pathlib import Path
import ast,hashlib,json
root=Path('英文小论文/code/warm_stem_primary_analysis12_20261011_v1')
s=root/'analyze.py';ast.parse(s.read_text(encoding='utf8'))
dependencies=['英文小论文/code/stem_init_stage_analysis_20261011_v2/compare_saved_stages.py','英文小论文/code/stem_init_stage_analysis_20261011_v2/protocol.json',
'英文小论文/code/stem_init_stage_analysis_20261011_v2/ranking_core.py','英文小论文/code/stem_init_stage_analysis_20261011_v2/ranking_protocol.json',
'英文小论文/code/stem_joint_error_analysis_20261011_v2/analyze.py','英文小论文/code/stem_joint_error_analysis_20261011_v2/protocol.json',
'英文小论文/code/warm_stem_primary_collection12_20261011_v1/verify_primary.py','英文小论文/code/warm_stem_primary_collection12_20261011_v1/collect.py',
'英文小论文/code/warm_stem_primary_collection12_20261011_v1/protocol.json']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=dict(schema='warm_stem_primary_stage_analysis12_v1',analysis_sha256=sha(s),dependencies_sha256={n:sha(n) for n in dependencies},
 permitted_seeds=[1,2],BASE_source='Existing fully source-bound stage/ranking caches only',independent_EMA_replay_required_for_formal_endpoint=True,
 first_boundary_only_selected300_regressed_boxes=True,scientific_root_or_novelty_proven=False,new_model_forwards=0,new_training_epochs=0)
with (root/'protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,indent=2)
print(json.dumps({'status':'WARM_PRIMARY_ANALYSIS_SOURCE_BOUND','analysis_sha256':p['analysis_sha256']}))
