"""Freeze one parameter-free audit before dataset noise sampling."""
from pathlib import Path
from datetime import datetime, timezone
import json
from a10_dn_geometry import sha, write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'A10_跨实例负去噪参考框几何审计_v1'
assert not OUT.exists(), 'Do not overwrite a frozen protocol.'
OUT.mkdir()
old = json.loads((ROOT / 'A6_稀疏编码器前景监督覆盖审计_v1' / 'protocol.json').read_text(encoding='utf-8'))
p = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'status': 'FROZEN_BEFORE_ANY_A10_DATASET_NOISE_DRAWS',
    'question': 'Do negative Dome denoising initial references frequently overlap a different tiny GT with an agreeing input class label?',
    'design': 'One fixed parameter-free Monte Carlo geometry audit on existing100 training-image GT traces. Not a detector experiment or independent confirmation of A9.',
    'source_archive': old['source_archive'], 'source_archive_sha256': old['source_archive_sha256'],
    'trace_manifest': old['trace_manifest'],
    'seed': 20261030, 'draws_per_image': 16,
    'operator': {'author_repo': 'https://github.com/RicePasteM/Dome-DETR', 'commit': '2dde3bc1946a3e9fad9abd0612b59fc39bd6b861', 'num_denoising': 100, 'num_classes': 12, 'label_noise_ratio': .5, 'box_noise_scale': 1., 'geometry': 'Float32 normalized GT reconstructed from cached640-square boxes; native800 size cohort; source abs-size correction,inverse_sigmoid eps1e-5 and sigmoid roundtrip included.'},
    'sampling': 'Native B1 groups=max(100//N,1),but primary uses the first group only to avoid batch-dependent group multiplicity; all B1 groups secondary. Actual training augmentation/batching distribution is not estimated.',
    'primary_cohort': 'sqrt(box area) at800-square<8 pixels; target cohort independent of query origin size.',
    'primary_key': 'one_group_same_input_class_iou0.5',
    'primary_event': 'At least one negative initial reference in first DN group has IoU>=.5 with another valid GT,excluding its own origin,and its perturbed input class equals that target GT class.',
    'aggregation': 'Per-target exposure per draw; average across16draws and GT within each image,then equal-image mean. 2000 prefix-group bootstrap resamples with seed2026103001. Not per-query independent confidence intervals.',
    'gate': 'At least200 tinyGT and15prefix groups. Mean>=.05 and bootstrap low>.02 -> prioritize further source/causal audit only. Bootstrap high<.02 -> deprioritize. Otherwise inconclusive. These are resource-prioritization cutoffs,not proof of harmful supervision or novelty.',
    'iou_thresholds': [.25, .5, .75],
    'secondary': ['Any input class upper-bound exposure', 'All native B1 groups', 'IoU.25/.75 without changing the primary decision', '8-16 and>=16 size strata', 'Own-origin IoU.5 collision rate; cannot substitute it for the cross-instance primary'],
    'interpretation_limits': ['References are decoder inputs,not output predictions; overlap does not prove erroneous final labels or harmful gradients.', 'Source annotation subset excludes ignored/unlabeled objects; rates concern valid cached GT only.', 'No augmentation or actual batch schedule simulated; no full training frequency claim.', 'A9 output geometry shortage is not causally linked to denoising by this audit.', 'Novelty still requires closest-method/source clearance; no module or training is approved by frequency alone.'],
    'stop': 'Exactly100 existing images,16draws each; no seed,threshold,group or cohort search after seeing results. No extra full model forward,optimizer update or fitted head.',
    'requires_before_run': 'PASS_PARAMETER_FREE_AUTHOR_OPERATOR_EQUIVALENCE on fixed synthetic inputs only.',
    'full_model_forwards_remain': 200, 'new_fits': 0, 'main_detector_updates': 0,
}
write(OUT / 'protocol.json', p)
files = [ROOT/'a10_dn_geometry.py', ROOT/'a10_author_operator_equivalence.py', Path(__file__), OUT/'protocol.json']
source = ROOT/'official_assets_v1'/'Dome_DETR_2dde3bc_v1'/'source'/'src'/'zoo'/'dome'
files += [source/'denoising.py', source/'box_ops.py', source/'utils.py']
write(OUT/'freeze_manifest.json', {'utc': datetime.now(timezone.utc).isoformat(), 'files': [{'path': str(f.relative_to(ROOT)), 'sha256': sha(f)} for f in files]})
print(json.dumps({'status': p['status'], 'images': len(p['trace_manifest']), 'draws_per_image': 16, 'new_model_forwards': 0}))
