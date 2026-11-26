"""Describe frozen source memberships and verified discarded/used Mosaic labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd()
P = ROOT / 'outputs/scale_protection12_20261010_v1'
OUT = P / 'exposure_description_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


def summarize(members, profile):
    sizes = [profile[i]['raw_valid_GT'] for i in members]
    risky = [round(profile[i]['raw_valid_GT'] * profile[i]['risky_fraction']) for i in members]
    assert all(abs(profile[i]['raw_valid_GT'] * profile[i]['risky_fraction'] - k) < 1e-8 for i, k in zip(members, risky))
    total = sum(sizes)
    return dict(image_identities=len(members), raw_valid_GT=total, raw_FP32_under4_GT=sum(risky),
        mean_raw_GT_per_image=float(np.mean(sizes)), median_raw_GT_per_image=float(np.median(sizes)),
        raw_GT_weighted_under4_fraction=sum(risky)/total if total else 0,
        target_rule_positive_images=sum(profile[i]['protect'] for i in members))


profile_path = P / 'terminal_collection/seed1_target_protect_v1/extracted/deployment/training_profile.json'
profile = {int(i): v for i, v in read(profile_path).items()}
assert len(profile) == 6471
assert sha(profile_path) == '5149b63006c5af5b3cb2bc8d424608b74652c44780d637395af6f5b0d6ef3be9'
target = sorted(i for i in profile if profile[i]['protect'])
assert len(target) == 995
membership = {}
source_stats = {'all_train': summarize(sorted(profile), profile), 'target_fixed995': summarize(target, profile)}
for seed in (1, 2):
    choices = {}
    exposures = Counter()
    for epoch in range(4, 12):
        ranked = sorted(profile, key=lambda iid: hashlib.sha256(f'scale_protection_control:{seed}:{epoch}:{iid}'.encode()).digest())
        selected = sorted(ranked[:995])
        assert len(selected) == len(set(selected)) == 995
        assert summarize(selected, profile)['image_identities'] == 995
        choices[str(epoch)] = dict(members=selected, source_summary=summarize(selected, profile),
            overlap_with_target_fixed995=len(set(selected) & set(target)))
        exposures.update(selected)
    membership[str(seed)] = choices
    source_stats[f'random_seed{seed}_across8_epochs'] = dict(
        unique_source_images=len(exposures), total_per_epoch_image_memberships=sum(exposures.values()),
        membership_multiplicity_histogram={str(n): sum(v == n for v in exposures.values()) for n in range(1, 9)},
        sum_raw_GT_over_membership_slots=sum(profile[i]['raw_valid_GT'] * n for i, n in exposures.items()),
        sum_raw_FP32_under4_GT_over_membership_slots=sum(round(profile[i]['raw_valid_GT'] * profile[i]['risky_fraction']) * n for i, n in exposures.items()))
    assert source_stats[f'random_seed{seed}_across8_epochs']['total_per_epoch_image_memberships'] == 7960

actual = {}
for seed in (1, 2):
    report_path = P / f'terminal_collection/seed{seed}_target_protect_v1/verification_report_v1.json'
    report = read(report_path)
    assert report['status'] == 'PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'
    assert report['augmentation']['all_policy_decisions_exact']
    totals = Counter()
    for epoch, item in report['augmentation']['Mosaic_events_by_epoch'].items():
        assert 4 <= int(epoch) <= 11
        totals.update(item)
    assert 0 < totals['substitutions'] < totals['events']
    actual[str(seed)] = dict(verified_report_sha256=sha(report_path), **dict(totals),
        labels_removed_from_discarded_Mosaic_outputs=totals['native_mosaic_GT']-totals['used_GT'],
        discarded_Mosaic_label_reduction_percent=100*(totals['native_mosaic_GT']-totals['used_GT'])/totals['native_mosaic_GT'],
        scope='Actual logged Mosaic calls, before sanitation/MixUp/collate/DN. Native Mosaic ran and its result was discarded on substitutions; the total is not a newly executed native training trajectory.')

OUT.mkdir(exist_ok=False)
def write(name, value):
    with (OUT / name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
write('random_source_memberships.json', membership)
report = dict(status='FROZEN_MEMBERSHIP_AND_VERIFIED_TARGET_EVENT_DESCRIPTION',
    source_profile_sha256=sha(profile_path), code_sha256=sha(__file__), source_stats=source_stats,
    target_actual_event_totals=actual,
    distinctions=[
        'Source membership995 is matched per epoch. Realized substituted events/GT exposures are not forced equal.',
        'Target protects a persistent995-image set. Random membership changes each epoch, so lifetime image coverage and protection multiplicity differ.',
        'FP32 clipped-source risk profile differs from FLOAT64 source-label cell diagnosis. Denominators are retained separately.',
        'Logged pre-sanitization labels are not post-collate labels, DN-positive counts or independent GT samples.',
        'The gate can compare these two specified policies. It cannot isolate a pure pixel-size, GT-density, context or persistence cause.'],
    new_model_forwards=0, new_evaluator_calls=0, new_optimizer_updates=0, new_training_epochs=0,
    acceptance_gates_unchanged=True, no_new_control_configuration=True,
    random_actual_event_totals='Pending actual activation and fixed12 completion; no membership simulation is reported as observed runtime events.')
assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
write('report.json', report)
write('manifest.json', dict(files={f.name: sha(f) for f in OUT.iterdir() if f.is_file()}))
print(json.dumps(dict(status=report['status'], source_stats=source_stats, target_actual_event_totals=actual)), flush=True)
