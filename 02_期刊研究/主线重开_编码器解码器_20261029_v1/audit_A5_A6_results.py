"""Reaggregate cached A5 results and independently enumerate every A6 lattice support."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib
import json
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def check_freeze(folder):
    for row in read(folder/'freeze_manifest.json')['files']:
        assert sha(folder/row['file'])==row['sha256']
    assert sha(folder/'summary.json')==read(folder/'completed.json')['summary_sha256']

def archive(folder,result):
    backup=folder/'backup';backup.mkdir(exist_ok=True)
    files=[p for p in sorted(folder.rglob('*')) if p.is_file() and not set(p.relative_to(folder).parts)&{'backup','__pycache__'}]
    manifest=[dict(file=p.relative_to(folder).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(backup/'file_manifest.json',manifest)
    target=backup/(folder.name.split('_')[0]+'_bounded_result.tar.gz')
    with tarfile.open(target,'w:gz') as tar:
        for path in files:
            tar.add(path,arcname=path.relative_to(folder).as_posix(),recursive=False)
        tar.add(backup/'file_manifest.json',arcname='file_manifest.json',recursive=False)
    with tarfile.open(target) as tar:
        for row in manifest:
            assert hashlib.sha256(tar.extractfile(row['file']).read()).hexdigest()==row['sha256']
    result.update(archive=str(target),archive_sha256=sha(target),archive_bytes=target.stat().st_size,archive_files_verified=len(files))
    write(backup/'receipt.json',result)

def audit_A5():
    folder=ROOT/'A5_既有中心场局部Taylor控制_v1';check_freeze(folder)
    summary=read(folder/'summary.json');images=read(folder/'per_image.json')
    gt=[json.loads(line) for line in (folder/'per_gt.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(gt)==461 and len({r['uid'] for r in gt})==461
    checked=0
    for row in gt:
        y=np.array(row['target_offset'])
        assert abs(np.linalg.norm(y)-row['zero_error'])<1e-12
        for view in ('P2','P2_down','P3_before','P3_after'):
            for kind in ('real','null'):
                assert abs(np.linalg.norm(np.array(row[view][kind+'_offset'])-y)-row[view][kind+'_error'])<1e-12
                checked+=1
    for image in images:
        rows=[r for r in gt if r['file_name']==image['file_name']]
        for cohort in ('lt16','lt8','from8to16'):
            use=[r for r in rows if cohort=='lt16' or (r['size']<8 if cohort=='lt8' else r['size']>=8)]
            assert len(use)==image[cohort+'_n']
            if use:
                for control in ('zero','null'):
                    value=np.mean([(r['zero_error'] if control=='zero' else r['P3_after']['null_error'])-r['P3_after']['real_error'] for r in use])
                    assert abs(value-image[cohort+'_P3_after_advantage_'+control])<1e-12
    assert not summary['positive_gate'] and summary['pure_Gaussian_max_error']<1e-12
    for control in ('zero','null'):
        values=[r['lt16_P3_after_advantage_'+control] for r in images if r['lt16_n']]
        assert abs(np.mean(values)-summary['estimates']['lt16_P3_after_advantage_'+control]['mean'])<1e-12
    result=dict(status='PASS_461_GT_3688_RADIAL_ERRORS_AND_IMAGE_PRIMARY_REAGGREGATION',utc=datetime.now(timezone.utc).isoformat(),
                verified_error_values=checked,no_refits=True,new_full_detector_forwards=0,
                scope='Integrity,numerical synthetic controls and stored-coordinate radial-error arithmetic;not full DARK or detection AP.')
    write(folder/'verification.json',result);archive(folder,result)
    return result

def enumerate_level(box,height,width,stride,low,high):
    b=np.array(box,dtype=np.float32)
    xs=np.arange(int(np.ceil(width/stride)),dtype=np.float32)*np.float32(stride)+np.float32(stride//2)
    ys=np.arange(int(np.ceil(height/stride)),dtype=np.float32)*np.float32(stride)+np.float32(stride//2)
    xi=np.flatnonzero((xs>b[0])&(xs<b[2]));yi=np.flatnonzero((ys>b[1])&(ys<b[3]))
    if not len(xi) or not len(yi):
        return []
    points=[]
    for y in yi:
        x=xs[xi]
        distances=np.stack([x-b[0],np.full_like(x,ys[y]-b[1]),b[2]-x,np.full_like(x,b[3]-ys[y])])
        furthest=distances.max(axis=0)
        for i in xi[(furthest>low)&(furthest<=high)]:
            points.append((int(y),int(i)))
    return points

def audit_A6():
    folder=ROOT/'A6_稀疏编码器前景监督覆盖审计_v1';check_freeze(folder)
    summary=read(folder/'summary.json');images=read(folder/'per_image.json');p=read(folder/'protocol.json')
    assert summary['source_equivalence_reference']['status']=='PASS_CPU_AUTHOR_TARGET_FUNCTION_EQUIVALENCE'
    gt=[json.loads(line) for line in (folder/'per_gt.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(gt)==13473 and len({r['uid'] for r in gt})==4491
    byimage=defaultdict(list)
    for row in gt:
        byimage[(row['file_name'],row['scenario'])].append(row)
    checked=0
    for image in images:
        rows=byimage[(image['file_name'],image['scenario'])]
        ranges=p['paper_overlap_sensitivity'] if 'overlap' in image['scenario'] else p['source_ranges']
        union=set();local={}
        for row in rows:
            levels=[enumerate_level(row['source_box'],image['height'],image['width'],s,*limits) for s,limits in zip((8,16,32,64),ranges)]
            counts=[len(v) for v in levels]
            assert counts==row['level_counts'] and sum(counts)==row['positive_tokens'],row['uid']
            union.update(levels[0]);local[row['uid']]=row
            checked+=1
        for row in rows:
            b=np.array(row['source_box'],dtype=np.float32)
            center=(b[:2]+b[2:])*np.float32(.5);col,rr=np.floor(center/8).astype(int)
            has_support=any((rr+dy,col+dx) in union for dy in (-1,0,1) for dx in (-1,0,1))
            assert row['no_nearby_P3_foreground']==(not has_support)
        for cohort in ('all','fixed640_lt8','fixed640_lt16','native_lt8','native_lt16'):
            if cohort=='all':use=rows
            else:
                size_key='size640' if cohort.startswith('fixed') else 'native_size'
                cutoff=8 if cohort.endswith('lt8') else 16
                use=[r for r in rows if r[size_key]<cutoff]
            assert len(use)==image[cohort+'_n']
            if use:
                for suffix in ('no_positive','no_positive_no_nearby'):
                    value=np.mean([r['positive_tokens']==0 and (r['no_nearby_P3_foreground'] if suffix.endswith('no_nearby') else True) for r in use])
                    assert abs(value-image[cohort+'_'+suffix])<1e-12
    assert all(summary['checks'].values())
    pooled={}
    for scenario in p['scenarios']:
        use=[r for r in gt if r['scenario']==scenario and r['size640']<8]
        pooled[scenario]=dict(gt=len(use),no_positive=sum(r['positive_tokens']==0 for r in use),
                              no_positive_no_nearby=sum(r['positive_tokens']==0 and r['no_nearby_P3_foreground'] for r in use))
    result=dict(status='PASS_INDEPENDENT_LATTICE_ENUMERATION_ALL13473_GT_SCENARIOS_AND_NEARBY_UNION',
                utc=datetime.now(timezone.utc).isoformat(),gt_scenarios_verified=checked,images=100,unique_gt=4491,pooled_primary_counts=pooled,
                new_fits=0,new_full_detector_forwards=0,
                scope='Source-bound target geometry only;not measured Focus-DETR predictions,encoder update choices,AP or novel algorithm efficacy.')
    write(folder/'verification.json',result);archive(folder,result)
    return result

if __name__=='__main__':
    print(json.dumps(dict(A5=audit_A5(),A6=audit_A6()),ensure_ascii=False))
