"""Freeze analysis routing and mathematical dependencies before new endpoints."""
import datetime,json
from compare_saved_stages import ROOT,HERE,PROTOCOL,read,sha,write
assert not PROTOCOL.exists()
old=ROOT/'英文小论文/code/scale_protection12_monitor_20261010_v1'
rank=read(ROOT/'outputs/sod_final_selection_20261010_v1/protocol.json')
assert sha(old/'analyze_final_selection.py')==rank['analysis_code_sha256']
assert sha(old/'geometry_core.py')==rank['geometry_code_sha256']
rank.update(analysis_code_sha256=sha(HERE/'ranking_core.py'),frozen_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    preparation_scope='Same mathematical reader; only external postprocessor/annotation paths are supplied for new terminal layouts.')
write(HERE/'ranking_protocol.json',rank)
readers=dict(full='英文小论文/code/original_stem12_monitor_20261011_v1',reference='英文小论文/code/native_stem_scratch12_monitor_20261011_v2',control='英文小论文/code/stem_component12_monitor_20261011_v1')
caches={str(s):dict(stage=f'outputs/scale_protection12_20261010_v1/stage_analysis/seed{s}_target_protect_v1',rank=f'outputs/sod_final_selection_20261010_v1/seed{s}_baseline') for s in (1,2)}
for cache in caches.values():
    for key in ('stage','rank'):cache[key+'_manifest_sha256']=sha(ROOT/cache[key]/'manifest.json')
deps={str((old/n).relative_to(ROOT)).replace('\\','/'):sha(old/n) for n in ('geometry_core.py','analyze_stages.py')}
for path in readers.values():
    for f in (ROOT/path).iterdir():
        if f.is_file():deps[f.relative_to(ROOT).as_posix()]=sha(f)
files=('compare_saved_stages.py','ranking_core.py','register_protocol.py','validate_preparation.py','ranking_protocol.json')
p=dict(schema='stem_init_stage_analysis_v2',analysis_files_sha256={n:sha(HERE/n) for n in files},dependency_files_sha256=deps,
    readers=readers,control_arms=['detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample'],
    baseline_caches=caches,
    scope='Three observed boundaries only: selected300 initial references, decoder-layer0 pre boxes, final boxes. Geometric support is not native one-to-one recall or AP, and its earliest change is not a uniquely identified Stem/backbone/fusion cause. Dense8400 pool, initial logits and every feature-module output are not observed here. Final ranking is a native-score-checked proxy with uncertain ties retained. Matching/duplicate ambiguity and training supervision causes remain unresolved by these saved validation tensors. GT identities are joined across runs; query identities are never aligned. Original fixed12 gates are unchanged.',
    frozen_before_new_fixed12_terminal_consumption=True,new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0)
write(PROTOCOL,p);print('STEM_STAGE_ANALYSIS_FROZEN',sha(PROTOCOL))
