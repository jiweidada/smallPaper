"""Local independent FP32 training-profile and paired source-GT reanalysis."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'outputs/scale_protection12_20261010_v1'
EXPECTED_ZIP = '18f07f06959df8bf64ae88628e35784959e57a1d76814279a2b2691599cdd884'
EXPECTED_DATA = '9912fc0450f76725e597340f6e7e586914bc76deb10c6d157e9faf1815b9378d'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def write(p, v):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)


def profile(annotation):
    images = {int(i['id']):i for i in annotation['images']}
    by_image = {i:[] for i in images}
    for a in annotation['annotations']:
        if a.get('iscrowd', 0):
            continue
        iid = int(a['image_id']); im = images[iid]
        b = np.array(a['bbox'], dtype=np.float32)
        b[2:] += b[:2]
        b[0::2] = np.clip(b[0::2], 0, im['width'])
        b[1::2] = np.clip(b[1::2], 0, im['height'])
        wh = b[2:]-b[:2]
        if (wh > 0).all():
            side = float((wh*np.array([640/im['width'],640/im['height']], dtype=np.float32)).min())
            by_image[iid].append(side)
    result = {}
    for iid, sides in by_image.items():
        frac = sum(x < 4. for x in sides)/len(sides) if sides else 0.
        result[str(iid)] = dict(raw_valid_GT=len(sides), risky_fraction=frac, protect=frac >= .25)
    return result


def main():
    z = OUT/'data_collection_v1.zip'; assert sha(z) == EXPECTED_ZIP
    target = OUT/'data_collection_v1'; assert not target.exists()
    with zipfile.ZipFile(z) as archive:
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in archive.namelist())
        archive.extractall(target)
    manifest = read(target/'data_collection_manifest.json')
    assert manifest['status'] == 'TERMINAL_DATA_ONLY_PACKET'
    for name, item in manifest['files'].items():
        p = target/name
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
    state = read(target/'diagnostics/controller_state.json')
    assert state['status'] == 'COMPLETE' and state['exit_code'] == 0 and state['data_only_physical_gate_passed']
    data_path = target/'diagnostics/data_probe.json'; assert sha(data_path) == EXPECTED_DATA
    d = read(data_path); assert d['status'] == 'COMPLETE_DATA_ONLY_SCALE_PROTECTION'
    assert d['new_training_epochs'] == d['optimizer_steps'] == d['model_forward_calls'] == 0
    assert d['protocol_sha256'] == sha(OUT/'protocol.json') == sha(target/'protocol.json')
    for name, h in d['implementation'].items():
        assert sha(target/'code'/name) == h and sha(Path(__file__).parent/name) == h
    ann_path = ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1/terminal_collection/seed1/extracted/reference/train_annotations.json'
    assert sha(ann_path) == d['identity']['train_annotation_sha256']
    computed = profile(read(ann_path)); assert computed == d['profile']
    rows = []; transitions = {}
    for seed in (1,2):
        native = d['modes'][f'{seed}:native']
        identity = d['modes'][f'{seed}:identity']
        assert all(native[k] == identity[k] for k in ('samples','batches','final_rng','ordinary_hashes'))
        old = {int(r['annotation_id']):r for s in native['samples'] for r in s['primary']}
        assert len(old) == 7537
        for arm in ('native','identity','target_protect','random_protect'):
            mode = d['modes'][f'{seed}:{arm}']
            primary = [r for s in mode['samples'] for r in s['primary']]
            retained = [r for r in primary if r['occurrences'] == 1]
            summary = dict(raw_primary_GT=len(primary), retained_unique_primary_GT=len(retained),
                missing_primary_GT=sum(not r['survived'] for r in primary), duplicate_excluded=sum(r['occurrences'] > 1 for r in primary),
                primary_input_lt4=sum(r['actual_input_shortside'] < 4 for r in retained),
                primary_input_ge4=sum(r['actual_input_shortside'] >= 4 for r in retained),
                exposures_before_mixup=sum(s['exposures_before_mixup'] for s in mode['samples']),
                exposures_after_mixup=sum(b['target_exposures_after_mixup'] for b in mode['batches']),
                Mosaic_events=len(mode['guard_records']) if arm != 'native' else None,
                substituted_events=sum(r['applied'] for r in mode['guard_records']))
            assert summary == mode['summary']
            assert mode['final_rng'] == native['final_rng']
            assert [b['rng_after'] for b in mode['batches']] == [b['rng_after'] for b in native['batches']]
            rows.append(dict(seed=seed, arm=arm, **summary))
            if arm != 'native':
                new = {int(r['annotation_id']):r for s in mode['samples'] for r in s['primary']}
                assert set(new) == set(old)
                common = [g for g in old if old[g]['occurrences'] == new[g]['occurrences'] == 1]
                t = dict(same_retained_GT=len(common),
                    native_retained_trial_missing=sum(v['survived'] and not new[g]['survived'] for g,v in old.items()),
                    native_missing_trial_retained=sum(not v['survived'] and new[g]['survived'] for g,v in old.items()),
                    common_native_lt4_trial_ge4=sum(old[g]['actual_input_shortside'] < 4 <= new[g]['actual_input_shortside'] for g in common),
                    common_native_ge4_trial_lt4=sum(new[g]['actual_input_shortside'] < 4 <= old[g]['actual_input_shortside'] for g in common))
                assert t == d['comparisons'][f'{seed}:{arm}']; transitions[f'{seed}:{arm}'] = t
                for r in mode['guard_records']:
                    iid = str(r['image_id'])
                    assert r['source_GT'] == computed[iid]['raw_valid_GT']
                    assert r['risky_fraction'] == computed[iid]['risky_fraction']
                    if arm == 'target_protect':
                        assert r['applied'] == computed[iid]['protect']
                    if r['applied']:
                        assert r['used_GT'] == r['source_GT']
                    else:
                        assert r['used_GT'] == r['native_mosaic_GT']
    passed = all(d['modes'][f'{s}:target_protect']['summary']['retained_unique_primary_GT'] >= d['modes'][f'{s}:native']['summary']['retained_unique_primary_GT']
        and d['modes'][f'{s}:target_protect']['summary']['primary_input_ge4'] > d['modes'][f'{s}:native']['summary']['primary_input_ge4'] for s in (1,2))
    assert passed == d['data_only_physical_gate_passed'] == True
    write(OUT/'training_profile.json', computed)
    report = dict(status='LOCAL_PROFILE_AND_PAIRED_GT_RECOMPUTATION_PASS', data_only_physical_gate_passed=True,
        archive_sha256=EXPECTED_ZIP, source_data_sha256=EXPECTED_DATA,
        immutable_packet_files_verified=len(manifest['files'])+1, training_images_profile_verified=len(computed),
        protected_training_images=sum(r['protect'] for r in computed.values()), rows=rows, transitions=transitions,
        candidate_AP_available=False, new_detector_forwards=0, optimizer_updates=0,
        scope='Physical feasibility and archived data-side equivalence only. Not model efficacy, exact realized event matching, statistical significance or root-cause identification.')
    write(OUT/'data_feasibility_verified.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
