"""Freeze one existing-math neighborhood control; extract A1 fitted heads,never refit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
A1=ROOT/'A1_冻结特征读出诊断_v1'
A4=ROOT/'A4_冻结特征精细中心偏移读出_v1'
OUT=ROOT/'A5_既有中心场局部Taylor控制_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Frozen protocol already exists'
    receipt=json.loads((A1/'backup/archive_receipt.json').read_text(encoding='utf-8'))
    archive=A1/'backup/A1_readout_probe_bounded.tar.gz'
    assert sha(archive)==receipt['sha256']
    heads=[]
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('file_manifest.json'))['files']
        for item in manifest:
            rel=item['file']
            if not rel.startswith('analysis/head_fold') or not rel.endswith('.npz'):
                continue
            data=tar.extractfile(rel).read()
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            target=A1/rel;target.parent.mkdir(exist_ok=True)
            if target.exists():
                assert target.read_bytes()==data
            else:
                target.write_bytes(data)
            heads.append(dict(path=str(target),sha256=item['sha256']))
    assert len(heads)==16
    a4_protocol=json.loads((A4/'protocol.json').read_text(encoding='utf-8'))
    cohort_file=A4/'per_gt.jsonl'
    cohort=[json.loads(line) for line in cohort_file.read_text(encoding='utf-8').splitlines()]
    byname={}
    for r in cohort:
        byname.setdefault(r['file_name'],set()).add(r['uid'])
    counts=[]
    for spec in a4_protocol['source_files']:
        path=Path(spec['path']);assert sha(path)==spec['sha256']
        with np.load(path,allow_pickle=False) as z:
            name=str(z['file_name']);ids=set(int(v) for v in z['fine_flat_ids'])
            eligible=[]
            for g,uid in enumerate(z['gt_uids']):
                if str(uid) not in byname.get(name,set()):
                    continue
                ci=int(z['gt_center_index'][g]);flat=int(z['fine_flat_ids'][ci]);rr,cc=divmod(flat,160)
                stencil=[(rr+dr)*160+cc+dc for dr in (-2,0,2) for dc in (-2,0,2)]
                if rr>=2 and rr<158 and cc>=2 and cc<158 and all(i in ids for i in stencil):
                    eligible.append(dict(uid=str(uid),size=float(z['gt_sqrt_area'][g])))
            counts.append(dict(file_name=name,group=str(z['group']),fold=int(z['fold']),
                               eligible=len(eligible),lt8=sum(r['size']<8 for r in eligible),
                               from8to16=sum(r['size']>=8 for r in eligible),uids=[r['uid'] for r in eligible]))
    protocol=dict(
        material_passport=dict(origin_skill='academic-research-suite',origin_mode='single fixed cached-neighborhood coordinate control',
                               origin_date=datetime.now(timezone.utc).isoformat(),verification_status='FROZEN_BEFORE_A5_FIELD_EVALUATION',version_label='A5_v1'),
        status='FROZEN_BEFORE_A5_FIELD_EVALUATION',
        motivation='A4 point-feature linear offsets failed. One known analytic neighborhood control tests whether existing A1 OOF center fields carry finer spatial structure;no new fitted head search.',
        primary_source='DARK primary arXiv1910.06278v1 Methodology,Eq9. This sparse unmodulated local Taylor control is NOT a full DARK reproduction.',
        differences_from_DARK=['Only3x3 samples spaced8 input pixels are available;no complete dense heatmap.',
                               'No Gaussian convolution/modulation step or full-image peak search can be reproduced from this cache.',
                               'GT supplies the correct4px cell. Fixed clipping/fallback are conditional diagnostic safeguards,not a deployable detector.'],
        source_archive=str(archive),source_archive_sha256=receipt['sha256'],source_features=a4_protocol['source_files'],
        source_A1_heads=heads,source_cohort=str(cohort_file),source_cohort_sha256=sha(cohort_file),
        fixed_images=counts,views=['P2','P2_down','P3_before','P3_after'],
        cohort='A4 fixed unique-cell<16GT with all9 cached center+8px-neighbor sample points present;no selection by predicted score or result.',
        operator=dict(stencil_spacing_input_px=8,log_floor=1e-12,
                      gradient='central finite differences of log field,axis units=input pixels',
                      hessian='3point diagonal second differences and4corner mixed difference',
                      step='delta=-solve(H,gradient)',accept='positive center score;center is local maximum;max Hessian eigenvalue<-1e-6',
                      safeguards='Otherwise zero displacement;clip accepted delta to[-2,2] per axis,using the GT-provided correct4px cell'),
        controls=['zero displacement','A1 within-image-shuffled-label OOF field using its existing frozen coefficient column',
                  'single-target exact Gaussian for numerical functional check','A1 max-over-GT Gaussian target field for neighborhood interference audit'],
        accounting='No new fits,heads,model forwards,or detector updates. Reuse16 A1 state files containing32 existing scalar models.',
        inference=dict(iterations=2000,seed=20261030,unit='filename-group cluster bootstrap of image means'),
        primary_endpoint='P3_after real-field radial-error advantage over zero and its shuffled-field control,all eligible<16GT.',
        gate=dict(min_gt=200,min_groups=8,min_gt_each_size_bin=50,min_error_advantage_px=.15,
                  pass_rule='Both primary mean advantages>=.15px with95%lower>0;zero-error advantage positive in both<8 and8-16 bins.',
                  fail='Do not add a generic center-field refinement module from these cached diagnostics.'),
        stop='One fixed analytic operator only. No bandwidth,log-floor,stencil,head,seed or fallback search after results. Even a pass is known-method conditional evidence,not novel algorithm/AP benefit.',
        limitations=['Author detector saw original training images;readout folds only isolate A1 linear fits.',
                     'Giving the correct4px cell excludes unknown-location discovery and query ownership.',
                     'Ridge center fields can be negative/non-Gaussian and the sparse cache prevents full DARK modulation;failure does not falsify full DARK or all neighborhood readouts.',
                     'Existing A1 center-field fits used all sample points/scales;A4 offset heads used different targets/data counts,so A5 is not a capacity-matched A4 architecture comparison.'],
        global_full_model_forwards_unchanged=185,new_fits=0,new_main_detector_updates=0,scientific_mainline_validated=False)
    write(OUT/'protocol.json',protocol)
    shutil.copyfile(ROOT/'a5_local_taylor.py',OUT/'a5_local_taylor.py')
    files=[dict(file=name,sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size) for name in ('protocol.json','a5_local_taylor.py')]
    write(OUT/'freeze_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),files=files))
    print(json.dumps(dict(status='A5_FROZEN',eligible=sum(r['eligible'] for r in counts),
                          lt8=sum(r['lt8'] for r in counts),from8to16=sum(r['from8to16'] for r in counts),
                          existing_A1_heads_extracted=16,new_fits=0,new_forwards=0)))

if __name__=='__main__':
    main()
