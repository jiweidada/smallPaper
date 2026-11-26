"""Extend unchanged native FP geometry rules to two actually verified original-user endpoints."""
from pathlib import Path
import ast,difflib,hashlib,json
ROOT=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=ROOT/'英文小论文/code/native_fp_geometry12_20261011_v1';new=ROOT/'英文小论文/code/native_fp_geometry12_20261011_v2';new.mkdir(exist_ok=False)
text=(old/'analyze.py').read_text(encoding='utf8')
a=text.index('    else:\n        vpath=ROOT/f');b=text.index("    for n,h in read(rank_folder/'manifest.json')",a)
text=text[:a]+'''    else:
        assert seed==2 and tag in ('backbone_silu','p2_skip')
        vpath=ROOT/f'outputs/user_backbone_primary_collection12_20261011_v2/seed{seed}_{tag}_v1/primary_verification_v1.json'
        v=read(vpath);assert v['status']=='PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'
        assert v['all_collected_files_verified'] and v['actual_stage_images']==548 and v['files_verified']==617
        assert v['verification_script_sha256']==sha(ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v2/verify_primary.py')
        assert v['identity']['seed']==seed and v['identity']['arm']==tag
        packet=Path(v['downloaded_packet']);assert sha(packet/'primary_manifest.json')==v['primary_manifest_sha256']
        folder=packet/'run';proof=sha(vpath)
        rank_folder=ROOT/f'outputs/user_backbone_primary_analysis12_20261011_v2/seed{seed}_{tag}/candidate_ranking'
'''+text[b:]
text=text.replace("assert seed in (1,2) and tag in ('BASE12','Warm12')", "assert seed==2 and tag in ('backbone_silu','p2_skip')")
text=text.replace('outputs/native_fp_geometry12_20261011_v1/','outputs/native_fp_geometry12_20261011_v2/')
text=text.replace("choices=('BASE12','Warm12')","choices=('backbone_silu','p2_skip')")
ast.parse(text);(new/'analyze.py').write_text(text,encoding='utf8')
protocol=json.loads((old/'protocol.json').read_bytes());protocol['schema']='actual_user_native_all_area_FP_geometry12_v2';protocol['analysis_sha256']=sha(new/'analyze.py')
protocol['dependencies_sha256']={n:h for n,h in protocol['dependencies_sha256'].items() if 'warm_stem' not in n}
for n in ('英文小论文/code/user_backbone_primary_collection12_20261011_v2/verify_primary.py','英文小论文/code/user_backbone_primary_collection12_20261011_v2/protocol.json','outputs/user_backbone_seed2_review12_20261011_v1/report.json'):
    protocol['dependencies_sha256'][n]=sha(ROOT/n)
protocol['permitted_actual_runs']=[dict(seed=2,tag=t) for t in ('backbone_silu','p2_skip')]
for t in ('backbone_silu','p2_skip'):
    for n in (f'outputs/user_backbone_primary_collection12_20261011_v2/seed2_{t}_v1/primary_verification_v1.json',f'outputs/user_backbone_primary_analysis12_20261011_v2/seed2_{t}/manifest.json'):
        protocol['dependencies_sha256'][n]=sha(ROOT/n)
(new/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
(new/'adaptation.diff').write_text(''.join(difflib.unified_diff((old/'analyze.py').read_text(encoding='utf8').splitlines(True),text.splitlines(True),fromfile='v1/analyze.py',tofile='v2/analyze.py')),encoding='utf8')
# Scientific FP labels and their tolerance policy are unchanged.
def function(src,name):return ast.dump(next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
assert function(text,'fp_label')==function((old/'analyze.py').read_text(encoding='utf8'),'fp_label')
with (ROOT/'outputs/user_native_fp_geometry12_20261011_v2_source_review.json').open('x',encoding='utf8') as f:json.dump(dict(status='ACTUAL_USER_SOURCES_BOUND_FP_LABEL_AST_UNCHANGED',files={p.name:sha(p) for p in new.iterdir() if p.is_file()},permitted_actual_runs=protocol['permitted_actual_runs'],new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0),f,indent=2)
print('PASS_ACTUAL_USER_PRIMARY_BOUND_SAME_FP_RULES')
