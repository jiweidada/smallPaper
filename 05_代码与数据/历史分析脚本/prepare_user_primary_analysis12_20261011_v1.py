"""Prepare separately sealed readers for all three existing user arms, using unchanged proven CPU cores."""
from pathlib import Path
import ast, difflib, hashlib, json, re
ROOT=Path.cwd();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)
ARMS="('backbone_silu','p2_skip','backbone_attention')"
oldstage=ROOT/'英文小论文/code/warm_stem_primary_analysis12_20261011_v1'
oldorder=ROOT/'英文小论文/code/warm_stem_native_order12_20261011_v1'
collection='英文小论文/code/user_backbone_primary_collection12_20261011_v1'
stage=ROOT/'英文小论文/code/user_backbone_primary_analysis12_20261011_v1'
order=ROOT/'英文小论文/code/user_backbone_native_order12_20261011_v1'
stage.mkdir(exist_ok=False);order.mkdir(exist_ok=False)
assert sha(oldstage/'analyze_v2.py')=='daadbc2921c04952c024830868317ec8286fa27569763e1c468e06f2e44fa2b6'
assert sha(oldorder/'analyze.py')=='8d5364d3dd6664685c879119bfac157f8b2002459301aa07919b6a03f9b5fcdc'

s=(oldstage/'analyze_v2.py').read_text(encoding='utf8')
for a,b in [
    ("def analyze(seed):", "def analyze(seed,arm):"),
    ("assert seed in (1,2)", "assert seed in (1,2) and arm in "+ARMS),
    ("HERE/'protocol_v2.json'", "HERE/'protocol.json'"),
    ('warm_stem_primary_collection12_20261011_v1/seed{seed}_warm_stem_v1', 'user_backbone_primary_collection12_20261011_v1/seed{seed}_{arm}_v1'),
    ('PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS', 'PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'),
    ('warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py', 'user_backbone_primary_collection12_20261011_v1/verify_primary.py'),
    ("identity['arm']=='warm_stem'", "identity['arm']==arm"),
    ('warm_stem_primary_analysis12_20261011_v1/seed{seed}', 'user_backbone_primary_analysis12_20261011_v1/seed{seed}_{arm}'),
    ("rank.analyze(packet,'warm_stem'", "rank.analyze(packet,arm"),
    ('SAME_GT_WARM_VS_CACHED_BASE_PRIMARY12_DESCRIPTION_SINGLE_SEED_REPLAY_PENDING', 'SAME_GT_USER_ARM_VS_CACHED_BASE_PRIMARY12_DESCRIPTION_SINGLE_SEED_REPLAY_PENDING'),
    ("seed=seed,identity=identity", "seed=seed,arm=arm,identity=identity"),
    ("a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=(1,2),required=True);analyze(a.parse_args().seed)", "a=argparse.ArgumentParser();a.add_argument('--seed',type=int,choices=(1,2),required=True);a.add_argument('--arm',choices="+ARMS+",required=True);args=a.parse_args();analyze(args.seed,args.arm)"),
]:s=replace(s,a,b)
s=s.replace('warm12','candidate12').replace('warm_ranking','candidate_ranking').replace('warm_stage_per_GT','candidate_stage_per_GT').replace('BASE_to_warm','BASE_to_candidate').replace('gained_TP_BASE_outcomes','gained_TP_BASE_outcomes').replace('lost_TP_warm_outcomes','lost_TP_candidate_outcomes')
s=re.sub(r'\bwarm\b','candidate',s).replace('Warm12','Candidate12')
ast.parse(s);(stage/'analyze.py').write_text(s,encoding='utf8')
(stage/'adaptation.diff').write_text(''.join(difflib.unified_diff((oldstage/'analyze_v2.py').read_text().splitlines(keepends=True),s.splitlines(keepends=True),fromfile='Warm_saved_stage_v2.py',tofile='user_saved_stage.py')),encoding='utf8')
deps={n:h for n,h in read(oldstage/'protocol_v2.json')['dependencies_sha256'].items() if '/warm_stem_primary_collection' not in n}
for name in ('collect.py','verify_primary.py','guards.py','protocol.json'):deps[collection+'/'+name]=sha(ROOT/collection/name)
for n,h in deps.items():assert sha(ROOT/n)==h,n
write(stage/'protocol.json',dict(schema='user_backbone_primary_stage_analysis12_v1',analysis_sha256=sha(stage/'analyze.py'),
    dependencies_sha256=deps,permitted_seeds=[1,2],permitted_arms=['backbone_silu','p2_skip','backbone_attention'],
    new_model_forwards=0,new_training_epochs=0,new_evaluator_calls=0,
    first_boundary_only_selected300_regressed_boxes=True,BASE_source='Previously verified cached same-seed BASE12 only',
    independent_saved_EMA_replay_required_for_formal_endpoint=True,scientific_root_or_novelty_proven=False))

s=(oldorder/'analyze.py').read_text(encoding='utf8')
for a,b in [
    ("def analyze(seed):", "def analyze(seed,arm):"),
    ("assert seed in (1,2) and sha(__file__)", "assert seed in (1,2) and arm in "+ARMS+" and sha(__file__)"),
    ('warm_stem_primary_collection12_20261011_v1/seed{seed}_warm_stem_v1', 'user_backbone_primary_collection12_20261011_v1/seed{seed}_{arm}_v1'),
    ('PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS', 'PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'),
    ('warm_stem_primary_collection12_20261011_v1/verify_primary_v2.py', 'user_backbone_primary_collection12_20261011_v1/verify_primary.py'),
    ("assert summary['identity']==verification['identity']", "assert summary['identity']==verification['identity'] and summary['identity']['arm']==arm and summary['identity']['seed']==seed"),
    ('warm_stem_native_order12_20261011_v1/seed{seed}', 'user_backbone_native_order12_20261011_v1/seed{seed}_{arm}'),
    ('WARM_AND_CACHED_BASE_NATIVE_ALL_AREA_PR20200_CELLS_AND_GT_PREFIXES_REPRODUCED_EXACTLY', 'USER_ARM_AND_CACHED_BASE_NATIVE_ALL_AREA_PR20200_CELLS_AND_GT_PREFIXES_REPRODUCED_EXACTLY'),
    ('seed=seed,classes=classes', 'seed=seed,arm=arm,classes=classes'),
    ('people / IoU0.50（最大负APs贡献类别，但这里是全面积的实际前段）', 'people / IoU0.50（示例类别；所有10类和10个IoU均保留；这里是全面积前段）'),
    ("p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);analyze(p.parse_args().seed)", "p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--arm',choices="+ARMS+",required=True);args=p.parse_args();analyze(args.seed,args.arm)"),
]:s=replace(s,a,b)
s=s.replace('Warm','Candidate').replace('warm','candidate')
s=replace(s,"'原生分数前段分析：Candidate种子'+str(seed)+'对已有BASE12'","'原生分数前段分析：'+arm+'种子'+str(seed)+'对已有BASE12'")
ast.parse(s);(order/'analyze.py').write_text(s,encoding='utf8')
(order/'adaptation.diff').write_text(''.join(difflib.unified_diff((oldorder/'analyze.py').read_text().splitlines(keepends=True),s.splitlines(keepends=True),fromfile='Warm_native_order.py',tofile='user_native_order.py')),encoding='utf8')
deps={n:h for n,h in read(oldorder/'protocol.json')['dependencies_sha256'].items() if '/warm_stem_primary_collection' not in n}
for name in ('verify_primary.py','guards.py','protocol.json'):deps[collection+'/'+name]=sha(ROOT/collection/name)
for n,h in deps.items():assert sha(ROOT/n)==h,n
write(order/'protocol.json',dict(schema='user_backbone_native_order12_v1',analysis_sha256=sha(order/'analyze.py'),
    dependencies_sha256=deps,prefixes=[100,500,1000,2000,5000,10000,20000],permitted_arms=['backbone_silu','p2_skip','backbone_attention'],
    evaluator_calls=0,model_forwards=0,new_training_epochs=0,
    scope='Exact all-area native score-order accounting only. Small-GT subset of all-area matches is not exact native APs prefix accounting. No cross-run box alignment or causal root.'))
proof=dict(status='USER_PRIMARY_STAGE_AND_NATIVE_ORDER_READERS_PREPARED_SOURCE_BOUND_NOT_RUN',
    source_sha256=sha(__file__),sources={p.relative_to(ROOT).as_posix():sha(p) for d in (stage,order) for p in d.iterdir() if p.is_file()},
    unchanged_geometry_ranking_and_native_accounting_cores=True,
    source_verified_Warm_analysis_seeds=[1,2],pending_real_user_endpoint_readers=6,
    new_candidate_model_forwards=0,new_BASE_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0,
    actual_scientific_endpoints_verified=0,actual_user_AP_improvement_proven=False)
write(ROOT/'outputs/user_primary_analysis12_20261011_v1_source_review.json',proof)
print(json.dumps(dict(status=proof['status'],source_files=len(proof['sources']),scientific_endpoints=0)))
