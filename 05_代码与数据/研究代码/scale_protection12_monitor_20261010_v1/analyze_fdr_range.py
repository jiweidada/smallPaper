"""GT-assisted algebraic range oracle on existing BASE12 traces, no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import geometry_core as geometry


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f: json.dump(x,f,indent=2,allow_nan=False,ensure_ascii=False)


def analyze(protocol_path,output):
    protocol_path,output=Path(protocol_path),Path(output)
    assert not output.exists()
    p=read(protocol_path);assert p['schema']=='sod_fdr_range_v1'
    assert sha(__file__)==p['analysis_source_sha256'] and sha(geometry.__file__)==p['geometry_reader_sha256']
    scale,up,regmax=p['reg_scale'],p['up'],p['reg_max'];bound=abs(up)*abs(scale)*2
    assert (scale,up,regmax)==(4.,.5,32) and bound==4.
    step=(abs(up*scale)+1)**(2/(regmax-2))
    weights=np.array([-bound]+[-step**i+1 for i in range(regmax//2-1,0,-1)]+[0.]+
                     [step**i-1 for i in range(1,regmax//2)]+[bound])
    assert len(weights)==33 and (np.diff(weights)>0).all()
    output.mkdir()
    reports={};bindings={}
    for seed,spec in p['inputs'].items():
        root=Path(spec['packet']);stage=root/'baseline_eval/stages_epoch12'
        for relative,h in spec['files'].items(): assert sha(root/relative)==h,relative
        manifest=read(stage/'manifest.json')
        ann=read(root/'reference/val_annotations.json')
        images={int(x['id']):x for x in ann['images']};group={k:[] for k in images}
        for a in ann['annotations']:group[int(a['image_id'])].append(a)
        assert len(images)==548 and len(ann['annotations'])==38759
        arrays=[];sources={};maximum_roundtrip=0.
        for iid,im in sorted(images.items()):
            path=stage/f'image_{iid:06d}.npz'
            assert sha(path)==manifest['files'][path.name];sources[path.name]=manifest['files'][path.name]
            with np.load(path,allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
            ref=obs['pre_after_first_layer_boxes'].astype(np.float64)
            assert ref.shape==(300,4) and np.isfinite(ref).all() and (ref[:,2:]>0).all()
            entries=sorted(group[iid],key=lambda a:int(a['id']))
            gt=np.array([a['bbox'] for a in entries],dtype=np.float64)
            normal=gt/np.array([im['width'],im['height'],im['width'],im['height']])
            x1,y1=normal[:,0,None],normal[:,1,None]
            x2,y2=x1+normal[:,2,None],y1+normal[:,3,None]
            cx,cy,w,h=(ref[:,j][None,:] for j in range(4))
            distances=np.stack((scale*(cx-x1)/w-scale*.5,scale*(cy-y1)/h-scale*.5,
                                scale*(x2-cx)/w-scale*.5,scale*(y2-cy)/h-scale*.5),axis=-1)
            magnitude=np.max(np.abs(distances),axis=-1)
            closed=magnitude<=bound
            strict=magnitude<bound-p['strict_interior_margin']
            # Verify inverse algebra for every GT/query pair without clipping the oracle.
            decoded=np.stack((cx-(.5*scale+distances[...,0])*w/scale,
                              cy-(.5*scale+distances[...,1])*h/scale,
                              cx+(.5*scale+distances[...,2])*w/scale,
                              cy+(.5*scale+distances[...,3])*h/scale),axis=-1)
            target=np.stack((x1[:,0],y1[:,0],x2[:,0],y2[:,0]),axis=-1)
            error=np.abs(decoded-target[:,None,:])*np.array([im['width'],im['height'],im['width'],im['height']])
            maximum_roundtrip=max(maximum_roundtrip,float(error.max()))
            metrics=geometry.image_metrics(im,entries,obs)
            best=metrics['best_query_within_run'][:,2];r=np.arange(len(entries))
            for j,a in enumerate(entries):
                arrays.append(dict(annotation_id=int(a['id']),image_id=iid,category_id=int(a['category_id']),
                    original_area=float(a['area']),input_shortside=float(min(normal[j,2:]*640)),
                    any_closed_range=bool(closed[j].any()),any_strict_interior=bool(strict[j].any()),
                    reachable_query_count=int(closed[j].sum()),best_final_query_in_range=bool(closed[j,best[j]]),
                    minimum_max_abs_distance=float(magnitude[j].min()),
                    max_iou=metrics['max_iou'][j].tolist(),best_final_query=int(best[j])))
        arrays.sort(key=lambda x:x['annotation_id'])
        assert len(arrays)==38759 and maximum_roundtrip<=p['roundtrip_pixel_tolerance']
        data={k:np.array([r[k] for r in arrays]) for k in arrays[0]}
        assert len(set(data['annotation_id'].tolist()))==38759
        np.savez_compressed(output/f'seed{seed}_per_GT.npz',**data)
        side=data['input_shortside'];m=data['max_iou']
        masks=dict(all=np.ones(len(side),dtype=bool),COCO_small=data['original_area']<=1024,
                   input_under4=side<4,input_4_to_under8=(side>=4)&(side<8),input_ge8=side>=8)
        assert masks['input_under4'].sum()==5096 and masks['COCO_small'].sum()==26586
        groups={}
        for name,mask in masks.items():
            failed=mask & (m[:,2]<.75);never=mask & ~(m>=.75).any(1)
            groups[name]=dict(GT=int(mask.sum()),closed_range_reachable=int((mask&data['any_closed_range']).sum()),
                strict_interior_reachable=int((mask&data['any_strict_interior']).sum()),
                final_IoU75_failed=int(failed.sum()),
                failed_and_range_reachable=int((failed&data['any_closed_range']).sum()),
                failed_and_best_final_query_in_range=int((failed&data['best_final_query_in_range']).sum()),
                never_IoU75_supported=int(never.sum()),never_supported_but_range_reachable=int((never&data['any_closed_range']).sum()),
                reachable_query_count_median=float(np.median(data['reachable_query_count'][mask])))
        reports[seed]=dict(groups=groups,inverse_roundtrip_max_error_pixels=maximum_roundtrip)
        bindings[seed]=dict(stage_manifest_sha256=sha(stage/'manifest.json'),stage_files=sources,
                            annotation_sha256=sha(root/'reference/val_annotations.json'))
    report=dict(status='GT_ASSISTED_FDR_RANGE_ORACLE_COMPLETED',protocol_sha256=sha(protocol_path),
        seeds=reports,bindings=bindings,weights=weights.tolist(),new_model_forwards=0,new_evaluator_calls=0,
        new_optimizer_updates=0,scope=p['scope'],interpretation=p['interpretation'])
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files={q.name:sha(q) for q in output.iterdir() if q.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();print(json.dumps(analyze(a.protocol,a.output),ensure_ascii=False),flush=True)
