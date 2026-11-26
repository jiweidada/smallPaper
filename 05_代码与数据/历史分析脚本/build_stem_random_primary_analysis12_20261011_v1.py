"""Reuse the executed all-GT analysis with only the random native-Stem evidence adapter."""
from pathlib import Path
import difflib,hashlib,json

source=Path('英文小论文/code/stem_remaining_primary_analysis12_20261011_v1/analyze.py')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read('outputs/stem_remaining_primary_analysis12_20261011_v1/seed2_attention_spatial_mean/report.json')['analysis_source_sha256']==sha(source)
text=source.read_text(encoding='utf8');original=text
edits=[
    ('All-GT descriptive comparison of actual primary12 Stem controls; no replay claim.', 'All-GT random native-Stem versus enhanced Stem primary12 comparison; no replay claim.'),
    ("arm in ('attention_spatial_mean','early_final_downsample')", "arm=='native_stem_scratch'"),
    ('outputs/stem_component_primary_collection_20261011_v2/seed{seed}_{arm}_v1', 'outputs/stem_primary_collection_20261011_v1/seed{seed}_{arm}_v1'),
    ('PASS_COMPONENT_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS', 'PASS_RANDOM_STEM_PRIMARY12_ALL_FILES_AND548_STAGE_ARRAYS'),
    ('英文小论文/code/stem_component_primary_verify12_20261011_v2/verify_primary.py', '英文小论文/code/stem_random_primary_collection12_20261011_v1/verify_primary.py'),
    ("readers['control']", "readers['reference']"),
    ('outputs/stem_remaining_primary_analysis12_20261011_v1', 'outputs/stem_random_primary_analysis12_20261011_v1'),
    ('SAME_GT_PRIMARY12_COMPONENT_DESCRIPTION_SINGLE_SEED_REPLAY_PENDING', 'SAME_GT_PRIMARY12_RANDOM_NATIVE_VS_ENHANCED_DESCRIPTION_REPLAY_PENDING'),
    ("control_scope=source_protocol['controls'][arm]", "control_scope='Native Stem default random initialization; all non-Stem native pretrained states retained. Both configurations omit native Stem pretrained states; structure, its initialization and capacity still differ.'"),
    ('other_seed_same_fixed12_endpoint_pending=True', 'other_seed_report_must_be_combined=True'),
    ('Other seed and independent saved-EMA proof remain required.', 'Combine both seed reports; independent saved-EMA proof remains required. This is the net early-learning effect of architecture plus its random initialization, not a unique mechanism or parameter-matched comparison.'),
    ("choices=('attention_spatial_mean','early_final_downsample')", "choices=('native_stem_scratch',)")
]
changes=[]
for before,after in edits:
    count=text.count(before);assert count>0,before
    text=text.replace(before,after);changes.append(dict(before=before,after=after,occurrences=count))
destination=Path('英文小论文/code/stem_random_primary_analysis12_20261011_v1');destination.mkdir(exist_ok=False)
with (destination/'analyze.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
with (destination/'adaptation.diff').open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(destination/'analyze.py'))))
with (destination/'adaptation.json').open('x',encoding='utf8') as f:json.dump(dict(source=str(source),source_sha256=sha(source),analysis_sha256=sha(destination/'analyze.py'),changes=changes,
    pure_geometry_ranking_joint_algorithms_unchanged=True,existing_frozen_model_training_and_terminal_protocols_unchanged=True,formal_terminal_gate_not_relaxed=True),f,indent=2)
print(json.dumps(dict(status='EXISTING_EXECUTED_ALL_GT_ALGORITHM_ADAPTED_TO_RANDOM_NATIVE_PRIMARY',source_sha256=sha(source),analysis_sha256=sha(destination/'analyze.py'))))
