"""Verify cached A1 features and freeze a new, explicitly exploratory precision probe."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
A1 = ROOT/'A1_冻结特征读出诊断_v1'
OUT = ROOT/'A4_冻结特征精细中心偏移读出_v1'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Do not silently replace a frozen protocol'
    archive = A1/'backup/A1_readout_probe_bounded.tar.gz'
    receipt = json.loads((A1/'backup/archive_receipt.json').read_text(encoding='utf-8'))
    archive_sha = sha(archive.read_bytes())
    assert archive_sha == receipt['sha256'] and len(archive_sha)==64
    verified = []
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile('file_manifest.json'))
        if isinstance(manifest, dict):
            manifest = manifest['files']
        for item in manifest:
            rel = item['file']
            data = tar.extractfile(rel).read()
            assert sha(data)==item['sha256'], rel
            if rel.startswith('features/'):
                target = A1/rel
                assert target.resolve().is_relative_to(A1.resolve())
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    assert target.read_bytes()==data
                else:
                    target.write_bytes(data)
                verified.append(dict(path=str(target), sha256=sha(data), bytes=len(data)))
    assert len(verified)==16
    a1_protocol = json.loads((A1/'protocol.json').read_text(encoding='utf-8'))
    specs = {x['file_name']:x for x in a1_protocol['images']}
    counts = []
    for item in verified:
        with np.load(item['path'], allow_pickle=False) as z:
            name = str(z['file_name'])
            ci = z['gt_center_index']
            eligible = z['gt_probe_eligible'].astype(bool) & (z['gt_sqrt_area']<16) & (ci>=0)
            _, inv, n = np.unique(ci[ci>=0], return_inverse=True, return_counts=True)
            unique = np.zeros(len(ci), bool)
            unique[np.where(ci>=0)[0]] = n[inv]==1
            eligible &= unique
            selected = np.where(eligible)[0]
            centers = (z['gt_boxes'][selected,:2]+z['gt_boxes'][selected,2:])*0.5
            offset = centers-z['xy'][ci[selected]]
            assert np.all(np.abs(offset)<=2.0001)
            item.update(file_name=name, fold=int(z['fold']), group=str(z['group']))
            assert item['fold']==specs[name]['fold'] and item['group']==specs[name]['group']
            counts.append(dict(file_name=name, fold=item['fold'], group=item['group'],
                               eligible=int(eligible.sum()), lt8=int((eligible&(z['gt_sqrt_area']<8)).sum()),
                               from8to16=int((eligible&(z['gt_sqrt_area']>=8)).sum())))
    protocol = dict(
        material_passport=dict(origin_skill='academic-research-suite', origin_mode='bounded exploratory cached-feature precision probe',
                               origin_date=datetime.now(timezone.utc).isoformat(), verification_status='FROZEN_BEFORE_A4_FITS', version_label='A4_v1'),
        status='FROZEN_BEFORE_A4_FITS',
        rationale='A1 local center ranking is not evidence of fine coordinate precision. Test offsets within the correct4px cell using exactly the cached features; no new GPU observation.',
        design_status='New prespecified analysis prompted by earlier A1/A2 and literature; project-level exploratory, not external preregistration or independent replication.',
        source_archive=str(archive), source_archive_sha256=archive_sha, source_files=verified,
        fold_contract='Reuse original4 A1 filename-group folds; no group crosses a fit/test split.',
        cohort='All A1-probe-eligible GT with sqrt resized input area<16; exclude every shared fine-center sample, considering all GT, not just tiny GT.',
        input='192-dimensional feature at the GT-containing4px cell center (4j+2), same input coordinates for P2,P2_down,P3_before,P3_after. No coordinates/class/GT box used as regressor input.',
        target='2D GT center minus4px cell center, in input pixels; both axes in[-2,2]. Correct cell is supplied by GT: conditional information diagnostic only.',
        views=['P2','P2_down','P3_before','P3_after'], ridge=0.01,
        fit='Training-fold image-balanced weighted ridge; training-only per-channel mean/std; unpenalized intercept. Solve real dx/dy and within-image jointly shuffled dx/dy together. Same shuffle across views.',
        predict='Clip all real/null/mean-constant predictions to[-2,2] per axis before radial Euclidean error.',
        fit_accounting=dict(two_response_linear_models=32, real_models=16, shuffled_target_models=16, scalar_coordinate_heads=64, parameters_per_two_response_model=386),
        controls=['training-fold image-balanced constant offset','zero offset','within-image jointly shuffled offset labels'],
        primary_endpoint='Image-mean radial error of P3_after minus P2, input pixels, on fixed<16 unique-cell cohort; positive favors P2.',
        inference=dict(method='filename-group cluster bootstrap of image means', iterations=2000, seed=20261030, confidence=0.95),
        gates=dict(min_gt=200,min_groups=8,min_gt_each_size_bin=50,
                   encoder_precision_gap='primary mean>=0.20px and95%lower>0;both<8 and8-16 differences>0;P2 error advantage over constant AND shuffled each>=0.15px with95%lower>0',
                   fused_precision_readable='If encoder-gap gate fails: P3_after advantage over constant AND shuffled each>=0.15px with95%lower>0',
                   otherwise='No sufficient support from this fixed linear precision probe; do not add an encoder or decoder module from it.'),
        image_counts=counts,
        limits=['Author detector checkpoint saw these training images; fold holdout applies only to the diagnostic readout.',
                'GT supplies the correct4px cell and excludes ambiguous shared cells; this is not full-image detection or instance separation.',
                'A negative linear probe does not prove information destruction; a positive result does not prove native decoder causality.',
                'Fine-position encoding can differ from coarse center saliency; do not interpret either as novel architecture benefit.',
                'No AP or independent validation-set claim; no hyperparameter/seed/feature-neighborhood search.'],
        new_gpu_forwards=0,new_main_detector_updates=0,global_full_forwards_unchanged=185,
        scientific_mainline_validated=False)
    dump(OUT/'protocol.json', protocol)
    shutil.copyfile(ROOT/'a4_precision_probe.py', OUT/'a4_precision_probe.py')
    files=[]
    for name in ('protocol.json','a4_precision_probe.py'):
        data=(OUT/name).read_bytes()
        files.append(dict(path=name,sha256=sha(data),bytes=len(data)))
    dump(OUT/'freeze_manifest.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(), files=files))
    print(json.dumps(dict(status='A4_FROZEN', images=16,eligible=sum(x['eligible'] for x in counts),
                          lt8=sum(x['lt8'] for x in counts),from8to16=sum(x['from8to16'] for x in counts),
                          groups=len({x['group'] for x in counts if x['eligible']}),
                          archive_sha256=archive_sha,new_forwards=0), ensure_ascii=False))

if __name__=='__main__':
    main()
