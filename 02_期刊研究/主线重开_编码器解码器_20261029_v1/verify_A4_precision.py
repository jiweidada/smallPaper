"""Independent prediction/reaggregation audit of frozen A4 models; no refits."""
from pathlib import Path
import hashlib
import json
import tarfile
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parent
A4=ROOT/'A4_冻结特征精细中心偏移读出_v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    frozen=json.loads((A4/'freeze_manifest.json').read_text(encoding='utf-8'))
    for row in frozen['files']:
        assert sha(A4/row['path'])==row['sha256']
    protocol=json.loads((A4/'protocol.json').read_text(encoding='utf-8'))
    summary=json.loads((A4/'summary.json').read_text(encoding='utf-8'))
    completed=json.loads((A4/'completed.json').read_text(encoding='utf-8'))
    assert sha(A4/'summary.json')==completed['summary_sha256']
    gt=[json.loads(line) for line in (A4/'per_gt.jsonl').read_text(encoding='utf-8').splitlines()]
    byuid={row['uid']:row for row in gt}
    assert len(byuid)==len(gt)==472
    per_image=json.loads((A4/'per_image.json').read_text(encoding='utf-8'))
    checked=0
    constant_by_fold={}
    source_rows=[]
    for spec in protocol['source_files']:
        path=Path(spec['path'])
        assert sha(path)==spec['sha256']
        with np.load(path,allow_pickle=False) as z:
            source_rows.append({k:z[k].copy() for k in z.files})
    for fold in range(4):
        means=[]
        for row in source_rows:
            if int(row['fold'])==fold:
                continue
            ci=row['gt_center_index'];size=row['gt_sqrt_area']
            allowed=row['gt_probe_eligible']&(size<16)&(ci>=0)
            counts={int(i):int((ci==i).sum()) for i in ci[ci>=0]}
            allowed &= np.array([counts.get(int(i),0)==1 for i in ci])
            chosen=np.flatnonzero(allowed)
            if len(chosen):
                actual=(row['gt_boxes'][chosen,:2]+row['gt_boxes'][chosen,2:])*.5
                means.append((actual-row['xy'][ci[chosen]]).mean(0))
        constant_by_fold[fold]=np.mean(means,axis=0)
    for model in summary['model_files']:
        assert sha(A4/model['path'])==model['sha256']
    for row in source_rows:
        name=str(row['file_name']);fold=int(row['fold'])
        records=[r for r in gt if r['file_name']==name]
        for rec in records:
            index=np.flatnonzero(row['gt_uids']==rec['uid'])
            assert len(index)==1
            g=int(index[0]);ci=int(row['gt_center_index'][g])
            actual=(row['gt_boxes'][g,:2]+row['gt_boxes'][g,2:])*.5
            target=actual-row['xy'][ci]
            np.testing.assert_allclose(target,rec['target_offset'],rtol=0,atol=1e-12)
            assert int(row['fold'])==rec['fold'] and str(row['group'])==rec['group']
            for view in protocol['views']:
                with np.load(A4/f'models/fold{fold}_{view}.npz',allow_pickle=False) as state:
                    np.testing.assert_allclose(state['constant'],constant_by_fold[fold],rtol=0,atol=2e-14)
                    x=row['features_'+view][ci].astype(np.float64)
                    z=np.r_[(x-state['mean'])/state['std'],1.]
                    prediction=np.maximum(-2.,np.minimum(2.,state['coef'].T@z))
                    errors=[float(np.hypot(*(prediction[:2]-target))),float(np.hypot(*(prediction[2:]-target))),
                            float(np.hypot(*(np.clip(state['constant'],-2,2)-target)))]
                for metric,error in zip(('error','null_error','constant_error'),errors):
                    assert abs(error-rec[view][metric])<2e-10,(rec['uid'],view,metric)
                    checked+=1
    for image in per_image:
        records=[r for r in gt if r['file_name']==image['file_name']]
        for cohort in ('lt16','lt8','from8to16','encoder_miss50'):
            selected=[r for r in records if cohort=='lt16' or
                      (cohort=='lt8' and r['size']<8) or (cohort=='from8to16' and r['size']>=8)]
            if cohort=='encoder_miss50':
                continue  # cohort membership is in source; primary+size strata audited here
            assert len(selected)==image[cohort+'_n']
            if selected:
                diff=np.mean([r['P3_after']['error']-r['P2']['error'] for r in selected])
                assert abs(diff-image[cohort+'_P3minusP2'])<1e-12
                for view in ('P2','P3_after'):
                    for control in ('constant','null'):
                        value=np.mean([r[view][control+'_error']-r[view]['error'] for r in selected])
                        assert abs(value-image[cohort+'_'+view+'_advantage_'+control])<1e-12
    eligible=[r for r in per_image if r['lt16_n']]
    assert abs(np.mean([r['lt16_P3minusP2'] for r in eligible])-summary['estimates']['lt16_P3minusP2']['mean'])<1e-12
    assert summary['new_model_forwards']==summary['new_main_detector_updates']==0
    assert summary['two_response_linear_models_fitted']==32 and summary['scalar_coordinate_heads']==64
    assert not summary['encoder_gap_gate'] and not summary['fused_readable_gate']
    result=dict(status='PASS_FROZEN_FILES_ALL472_GT_5664_ERROR_VALUES_AND_IMAGE_PRIMARY_REAGGREGATION',
                utc=datetime.now(timezone.utc).isoformat(),error_values_checked=checked,
                model_files_verified=len(summary['model_files']),source_files_verified=len(source_rows),
                no_refits=True,new_forwards=0,scientific_mainline_validated=False)
    (A4/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    backup=A4/'backup';backup.mkdir(exist_ok=True)
    files=[p for p in sorted(A4.rglob('*')) if p.is_file() and 'backup' not in p.relative_to(A4).parts and '__pycache__' not in p.relative_to(A4).parts]
    manifest=[dict(file=p.relative_to(A4).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    (backup/'file_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=backup/'A4_cached_precision_probe.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:
            tar.add(p,arcname=p.relative_to(A4).as_posix(),recursive=False)
        tar.add(backup/'file_manifest.json',arcname='file_manifest.json',recursive=False)
    result.update(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,hashed_files=len(files))
    (backup/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
