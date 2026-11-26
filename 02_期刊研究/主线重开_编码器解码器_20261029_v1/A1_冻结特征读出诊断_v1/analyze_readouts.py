"""A1 prespecified32 CPU linear fits. No torch,CUDA,or architecture training."""
from pathlib import Path
import hashlib
import json
import os
import time
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
from readout_common import VIEWS, fit_readouts, predict, gt_pair_wins, paired_group_ci, selfcheck

ROOT = Path(__file__).resolve().parent
COHORTS = ('all','lt8','8to16','16to32','ge32','lt16','lt16_encoder_miss50')

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def cohort_mask(row,name):
    size = row['gt_sqrt_area']
    masks = dict(all=np.ones(len(size),bool),lt8=size<8,**{'8to16':(size>=8)&(size<16),
                 '16to32':(size>=16)&(size<32)},ge32=size>=32,lt16=size<16,
                 lt16_encoder_miss50=(size<16)&(row['gt_all_encoder_max_iou']<.5))
    return row['gt_probe_eligible'] & masks[name]

def main():
    manifest = json.loads((ROOT/'freeze_manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest() == item['sha256']
    protocol = json.loads((ROOT/'protocol.json').read_text())
    extraction = json.loads((ROOT/'receipts/extraction_complete.json').read_text())
    assert extraction['status'] == 'FROZEN_FEATURE_EXTRACTION_COMPLETE'
    assert extraction['full_model_forwards_cumulative'] == 185
    assert extraction['main_model_state_unchanged'] and extraction['full_model_training_updates'] == 0
    result_dir = ROOT/'analysis'
    result_dir.mkdir(exist_ok=True)
    with (result_dir/'analysis_started.json').open('x') as handle:
        json.dump({'pid':os.getpid(),'started_unix':time.time(),'prespecified_linear_fits':32},handle)
    print(json.dumps({'status':'CPU_ANALYSIS_STARTED','selfcheck':selfcheck(),'pid':os.getpid()}),flush=True)
    rows = []
    for spec in protocol['images']:
        file = ROOT/'features'/(Path(spec['file_name']).stem+'.npz')
        receipt = next(r for r in extraction['comparisons'] if r['file_name']==spec['file_name'])
        assert hashlib.sha256(file.read_bytes()).hexdigest() == receipt['sha256']
        with np.load(file,allow_pickle=False) as z:
            row = {k:z[k].copy() for k in z.files}
        row.update(file_name=str(row['file_name']),group=str(row['group']),fold=int(row['fold']))
        assert row['fold']==spec['fold'] and row['group']==spec['group']
        assert all(row['features_'+view].shape == (len(row['target']),192) for view in VIEWS)
        rows.append(row)
    assert len(rows)==16 and set(r['fold'] for r in rows)==set(range(4))
    for group in {r['group'] for r in rows}:
        assert len({r['fold'] for r in rows if r['group']==group})==1
    outcomes = {r['file_name']:{} for r in rows}
    fit_count = 0
    model_hashes = []
    began = time.monotonic()
    for fold in range(4):
        train,test = [r for r in rows if r['fold']!=fold],[r for r in rows if r['fold']==fold]
        assert not ({r['group'] for r in train}&{r['group'] for r in test})
        for view in VIEWS:
            fitted = fit_readouts(train,view,ridge=.01,fold=fold)
            fit_count += 2
            path = result_dir/(f'head_fold{fold}_{view}.npz')
            np.savez_compressed(path,**fitted)
            model_hashes.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for row in test:
                scores = predict(row,view,fitted)
                assert np.isfinite(scores).all()
                outcomes[row['file_name']][view] = gt_pair_wins(row,scores)
        print(json.dumps({'status':'FOLD_COMPLETE','fold':fold,'train_images':len(train),
                          'test_images':len(test),'cumulative_readout_fits':fit_count}),flush=True)
    assert fit_count == 32
    per_image,per_gt = [],[]
    for row in rows:
        values = outcomes[row['file_name']]
        native = gt_pair_wins(row,row['native_score'])[:,0]
        img = dict(file_name=row['file_name'],group=row['group'],fold=row['fold'],
                   total_gt=len(row['gt_boxes']),eligible_gt=int(row['gt_probe_eligible'].sum()),points=len(row['target']))
        for cohort in COHORTS:
            mask = cohort_mask(row,cohort)
            img['n_'+cohort] = int(mask.sum())
            for view in VIEWS:
                for j,kind in enumerate(('real','null')):
                    img[view+'_'+kind+'_'+cohort] = float(values[view][mask,j].mean()) if mask.any() else None
            img['native_'+cohort] = float(native[mask].mean()) if mask.any() else None
            for a,b in (('P2','P3_after'),('P2','P2_down'),('P3_before','P3_after')):
                img[a+'_minus_'+b+'_'+cohort] = float((values[a][mask,0]-values[b][mask,0]).mean()) if mask.any() else None
        per_image.append(img)
        for g,uid in enumerate(row['gt_uids']):
            rec = dict(uid=str(uid),file_name=row['file_name'],group=row['group'],fold=row['fold'],
                       sqrt_input_area=float(row['gt_sqrt_area'][g]),class_id=int(row['gt_labels'][g]),
                       eligible=bool(row['gt_probe_eligible'][g]),
                       all_encoder_max_iou=float(row['gt_all_encoder_max_iou'][g]),
                       topk_max_iou=float(row['gt_topk_max_iou'][g]),
                       same16cell_crowd=bool(row['gt_crowd'][g]))
            for view in VIEWS:
                for j,kind in enumerate(('real','null')):
                    val = values[view][g,j]
                    rec[view+'_'+kind+'_pair_win'] = float(val) if np.isfinite(val) else None
            per_gt.append(rec)
    estimates = []
    for cohort in COHORTS:
        for view in VIEWS:
            for kind in ('real','null'):
                ci = paired_group_ci(per_image,view+'_'+kind+'_'+cohort)
                estimates.append(dict(cohort=cohort,view=view,kind=kind,
                                      gt=sum(r['n_'+cohort] for r in per_image),**ci))
    primary = paired_group_ci(per_image,'P2_minus_P3_after_lt16')
    residual = paired_group_ci(per_image,'P2_minus_P3_after_lt16_encoder_miss50')
    size_contrasts = {c:paired_group_ci(per_image,'P2_minus_P3_after_'+c) for c in ('lt8','8to16')}
    n_primary = sum(r['n_lt16'] for r in per_image)
    n_residual = sum(r['n_lt16_encoder_miss50'] for r in per_image)
    null_means = {v:next(r['mean'] for r in estimates if r['cohort']=='lt16' and r['view']==v and r['kind']=='null') for v in ('P2','P3_after')}
    checks = dict(enough_primary_gt=n_primary>=50,enough_groups=primary['groups']>=8,
                  primary_effect_at_least_005=primary['mean'] is not None and primary['mean']>=.05,
                  primary_interval_above_zero=primary['low'] is not None and primary['low']>0,
                  both_size_bins_nonnegative=all(v['mean'] is not None and v['mean']>=0 for v in size_contrasts.values()),
                  enough_residual_gt=n_residual>=30,
                  residual_positive=residual['mean'] is not None and residual['mean']>0,
                  null_controls_below_060=all(v is not None and v<.6 for v in null_means.values()))
    gate = 'SUPPORTS_FURTHER_LOCAL_READOUT_INVESTIGATION_NOT_ARCHITECTURE_BENEFIT' if all(checks.values()) else 'DOES_NOT_PASS_PRESPECIFIED_LINEAR_READOUT_GATE'
    summary = dict(status='A1_PRESPECIFIED_CPU_READOUT_ANALYSIS_COMPLETE',images=16,
        groups=len({r['group'] for r in rows}),folds=4,feature_views=list(VIEWS),linear_readouts_fitted=32,
        real_readouts_fitted=16,shuffled_label_readouts_fitted=16,parameters_per_readout=193,
        total_gt=len(per_gt),eligible_gt=sum(r['eligible'] for r in per_gt),
        primary_lt16_gt=n_primary,residual_lt16_gt=n_residual,
        primary_P2_minus_P3_after=primary,residual_P2_minus_P3_after=residual,
        size_contrasts=size_contrasts,null_controls=null_means,gate_checks=checks,decision_gate=gate,
        estimates=estimates,elapsed_cpu_analysis_seconds=time.monotonic()-began,
        full_model_forwards_cumulative=185,main_model_training_updates=0,
        models=model_hashes,
        scope='Conditional local-center ranking at GT-derived candidate neighborhoods. This is not object AP,instance separation proof,generalization,or causal architecture improvement. Equal-head capacity does not equal receptive field or intrinsic feature capacity.',
        limitations=['Author model saw original training images;probe folds only hold out readout-fitting image groups.',
                     'Filename prefix grouping is a correlation-reduction heuristic,not independently verified scene identity.',
                     'Shallow/learned-downsampled/fused representations differ in receptive field and intrinsic dimension.',
                     'Same fine-cell GT can share a center sample;this probe cannot establish within-cell instance separability.',
                     'A negative linear probe does not exclude nonlinear accessible information.',
                     'Only one primary contrast;other view/scale contrasts are descriptive,not alternative positives.'])
    dump(result_dir/'summary.json',summary)
    dump(result_dir/'per_image.json',per_image)
    with (result_dir/'per_gt.jsonl').open('w',encoding='utf-8') as f:
        for row in per_gt:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    dump(result_dir/'completed.json',dict(status='ANALYSIS_COMPLETE',linear_readouts_fitted=32,
                                        summary_sha256=hashlib.sha256((result_dir/'summary.json').read_bytes()).hexdigest()))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('estimates','models','limitations')},allow_nan=False),flush=True)

if __name__=='__main__':
    main()
