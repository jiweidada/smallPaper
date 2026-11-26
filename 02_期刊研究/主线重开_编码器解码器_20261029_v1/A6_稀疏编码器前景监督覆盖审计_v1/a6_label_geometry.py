"""Fixed source-equivalent foreground-label geometry audit;no detector or fitted head."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import io
import json
import tarfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
STRIDES=(8,16,32,64)
SOURCE_RANGES=((-1,64),(64,128),(128,256),(256,999999))
PAPER_OVERLAP_RANGES=((-1,64),(64,256),(128,512),(256,999999))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def source_roundtrip(boxes,height,width):
    g=np.asarray(boxes,dtype=np.float32)
    size=np.array([width,height,width,height],dtype=np.float32)
    normalized=g/size
    center=(normalized[:,:2]+normalized[:,2:])*np.float32(.5)
    extent=normalized[:,2:]-normalized[:,:2]
    normalized_cxcywh=np.concatenate([center,extent],axis=1)
    pixel=normalized_cxcywh*size
    return np.concatenate([pixel[:,:2]-pixel[:,2:]*np.float32(.5),pixel[:,:2]+pixel[:,2:]*np.float32(.5)],axis=1)

def level_eligibility(boxes,height,width,stride,limits):
    boxes=np.asarray(boxes,dtype=np.float32)
    rr,cc=np.meshgrid(np.arange(int(np.ceil(height/stride))),np.arange(int(np.ceil(width/stride))),indexing='ij')
    coords=np.stack([cc.ravel(),rr.ravel()],axis=1).astype(np.float32)*np.float32(stride)+np.float32(stride//2)
    left=coords[:,None,0]-boxes[None,:,0]
    top=coords[:,None,1]-boxes[None,:,1]
    right=boxes[None,:,2]-coords[:,None,0]
    bottom=boxes[None,:,3]-coords[:,None,1]
    offsets=np.stack([left,top,right,bottom],axis=-1)
    eligibility=(offsets.min(axis=-1)>0)&(offsets.max(axis=-1)>limits[0])&(offsets.max(axis=-1)<=limits[1])
    return coords,eligibility

def resize_reference(boxes640,original_height,original_width):
    scale=min(800/min(original_height,original_width),1333/max(original_height,original_width))
    height=int(original_height*scale+.5);width=int(original_width*scale+.5)
    # Cached boxes are clipped original coordinates independently scaled to640.
    boxes=np.asarray(boxes640,dtype=np.float64)*np.array([width/640,height/640,width/640,height/640])
    return boxes,height,width

def group_ci(rows,key,seed=20261030,iterations=2000):
    kept=[r for r in rows if r.get(key) is not None]
    if not kept:
        return dict(mean=None,low=None,high=None,images=0,groups=0)
    groups=sorted({r['group'] for r in kept});v=[[r[key] for r in kept if r['group']==g] for g in groups]
    rng=np.random.default_rng(seed)
    boot=[np.mean([q for i in rng.integers(len(groups),size=len(groups)) for q in v[i]]) for _ in range(iterations)]
    return dict(mean=float(np.mean([r[key] for r in kept])),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),images=len(kept),groups=len(groups))

def main():
    manifest=json.loads((ROOT/'freeze_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        assert sha(ROOT/item['file'])==item['sha256']
    p=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_BEFORE_A6_LABEL_COUNTS'
    reference=json.loads((ROOT/'author_source_equivalence.json').read_text(encoding='utf-8'))
    assert reference['status']=='PASS_CPU_AUTHOR_TARGET_FUNCTION_EQUIVALENCE'
    (ROOT/'started.json').open('x',encoding='utf-8').write(json.dumps(dict(pid=__import__('os').getpid(),utc=datetime.now(timezone.utc).isoformat())))
    began=time.monotonic();metadata=json.loads((ROOT/'dataset_geometry_metadata.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'dataset_geometry_metadata.json')==p['geometry_metadata_sha256']
    geometry={r['file_name']:r for r in metadata['rows']}
    archive=Path(p['source_archive']);assert sha(archive)==p['source_archive_sha256']
    gt_rows=[];image_rows=[];verified=[]
    with tarfile.open(archive) as tar:
        for spec in p['trace_manifest']:
            data=tar.extractfile(spec['path']).read();assert hashlib.sha256(data).hexdigest()==spec['sha256']
            with np.load(io.BytesIO(data),allow_pickle=False) as z:
                original={k:z[k].copy() for k in ('gt_boxes','gt_labels','gt_sqrt_area','gt_uids')}
            name=Path(spec['path']).stem+'.jpg';meta=geometry[name]
            verified.append(dict(path=spec['path'],sha256=spec['sha256']))
            boxes=original['gt_boxes'];base_size=original['gt_sqrt_area']
            reference_boxes,reference_h,reference_w=resize_reference(boxes,meta['height'],meta['width'])
            scenarios=[('square640_source',boxes,640,640,SOURCE_RANGES),
                       ('aspect800_reference_source',reference_boxes,reference_h,reference_w,SOURCE_RANGES),
                       ('square640_paper_overlap_sensitivity',boxes,640,640,PAPER_OVERLAP_RANGES)]
            for scenario,pixel_boxes,height,width,ranges in scenarios:
                normalized=source_roundtrip(pixel_boxes,height,width)
                counts=np.zeros(len(boxes),dtype=np.int64);level_counts=[]
                p3_coords=p3_union=None
                for stride,limits in zip(STRIDES,ranges):
                    coords,eligibility=level_eligibility(normalized,height,width,stride,limits)
                    n=eligibility.sum(axis=0);counts+=n;level_counts.append(n)
                    if stride==8:
                        p3_coords=coords;p3_union=eligibility.any(axis=1)
                centers=(normalized[:,:2]+normalized[:,2:])*np.float32(.5)
                nearest=np.floor(centers/8).astype(np.int64)
                point_cols=np.round((p3_coords[:,0]-4)/8).astype(np.int64)
                point_rows=np.round((p3_coords[:,1]-4)/8).astype(np.int64)
                no_nearby=[]
                for col,row in nearest:
                    nearby=(np.abs(point_cols-col)<=1)&(np.abs(point_rows-row)<=1)
                    no_nearby.append(not p3_union[nearby].any())
                no_nearby=np.asarray(no_nearby,bool)
                current_size=np.sqrt(np.prod(np.maximum(normalized[:,2:]-normalized[:,:2],0),axis=1))
                image=dict(file_name=name,group=name.split('_')[0],scenario=scenario,height=height,width=width,
                           tokens=sum(int(np.ceil(height/s))*int(np.ceil(width/s)) for s in STRIDES),total_gt=len(boxes))
                masks=dict(all=np.ones(len(boxes),bool),fixed640_lt8=base_size<8,fixed640_lt16=base_size<16,
                           native_lt8=current_size<8,native_lt16=current_size<16)
                for cohort,mask in masks.items():
                    image[cohort+'_n']=int(mask.sum())
                    if mask.any():
                        image[cohort+'_no_positive']=float(np.mean(counts[mask]==0))
                        image[cohort+'_no_positive_no_nearby']=float(np.mean((counts[mask]==0)&no_nearby[mask]))
                image_rows.append(image)
                for i,uid in enumerate(original['gt_uids']):
                    gt_rows.append(dict(uid=str(uid),file_name=name,group=image['group'],scenario=scenario,
                                        size640=float(base_size[i]),native_size=float(current_size[i]),class_id=int(original['gt_labels'][i]),
                                        positive_tokens=int(counts[i]),level_counts=[int(v[i]) for v in level_counts],
                                        no_nearby_P3_foreground=bool(no_nearby[i]),source_box=normalized[i].tolist()))
    assert len(verified)==100 and len(gt_rows)==4491*3
    estimates={}
    for scenario in p['scenarios']:
        these=[r for r in image_rows if r['scenario']==scenario]
        keys=sorted({key for r in these for key in r if key.endswith('_no_positive') or key.endswith('_no_positive_no_nearby')})
        estimates[scenario]={key:group_ci(these,key) for key in keys}
    primary=estimates['square640_source'];sensitivity=estimates['aspect800_reference_source']
    checks=dict(
        enough_gt=sum(r['fixed640_lt8_n'] for r in image_rows if r['scenario']=='square640_source')>=200,
        enough_groups=primary['fixed640_lt8_no_positive']['groups']>=15,
        point_hole=primary['fixed640_lt8_no_positive']['mean']>=.20 and primary['fixed640_lt8_no_positive']['low']>.10,
        no_nearby=primary['fixed640_lt8_no_positive_no_nearby']['mean']>=.10 and primary['fixed640_lt8_no_positive_no_nearby']['low']>.05,
        aspect_robust=sensitivity['fixed640_lt8_no_positive']['mean']>=.10)
    decision='GEOMETRIC_SUPERVISION_GAP_WORTH_BOUNDED_ENCODER_COVERAGE_TEST_NOT_NEW_ALGORITHM' if all(checks.values()) else 'DOES_NOT_PASS_ALL_COVERAGE_PRIORITIZATION_GATES'
    result=dict(status='A6_FOREGROUND_LABEL_GEOMETRY_AUDIT_COMPLETE',decision=decision,checks={k:bool(v) for k,v in checks.items()},
                images=100,unique_gt=4491,scenario_gt_rows=len(gt_rows),estimates=estimates,
                source_equivalence_reference=reference,source_protocol='Inspected Focus-DETR targets only;not Focus detector execution or AP evaluation.',
                new_full_detector_forwards=0,new_fits=0,detector_updates=0,global_forwards_unchanged=185,
                elapsed_seconds=time.monotonic()-began,limitations=p['limitations'],scientific_mainline_validated=False)
    write(ROOT/'summary.json',result);write(ROOT/'per_image.json',image_rows);write(ROOT/'trace_verification.json',verified)
    with (ROOT/'per_gt.jsonl').open('w',encoding='utf-8') as handle:
        for row in gt_rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(ROOT/'completed.json',dict(status='TERMINAL_SUCCESS',summary_sha256=sha(ROOT/'summary.json'),new_fits=0,new_full_detector_forwards=0))
    print(json.dumps(dict(status=result['status'],decision=decision,checks=result['checks'],
                         primary_no_positive=primary['fixed640_lt8_no_positive'],
                         primary_no_positive_no_nearby=primary['fixed640_lt8_no_positive_no_nearby'],
                         aspect_reference=sensitivity['fixed640_lt8_no_positive'],elapsed_seconds=result['elapsed_seconds'])))

if __name__=='__main__':
    main()
