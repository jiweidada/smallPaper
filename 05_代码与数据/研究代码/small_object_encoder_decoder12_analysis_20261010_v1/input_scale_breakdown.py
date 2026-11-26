"""Describe saved GT outcomes by actual 640-input size; no model execution."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from results import read,write,sha
import geometry_core as geometry

VAL_SHA='0329595b1f5aa8153732407a1d906de0dd5db7b7e8e005b8941ff19e90e774bd'
EDGES=(0.,2.,4.,8.,16.,32.,float('inf'))
LABELS=('under2','2_to_under4','4_to_under8','8_to_under16','16_to_under32','32_and_above')


def load_saved(folder):
    folder=Path(folder);manifest=read(folder/'manifest.json')
    for name,h in manifest['files'].items():assert sha(folder/name)==h,name
    report=read(folder/'report.json');assert report['annotation_sha256']==VAL_SHA
    assert report['native_TP_geometry_disagreement_cells']==0
    arrays={}
    for arm in ('baseline','candidate'):
        p=folder/f'{arm}_per_GT.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:arrays[arm]={k:z[k].copy() for k in z.files}
    assert 'baseline' in arrays
    return report,arrays


def run(stage_analysis,annotation_path,output):
    output=Path(output);assert not output.exists()
    assert sha(annotation_path)==VAL_SHA
    annotation=read(annotation_path);report,arrays=load_saved(stage_analysis)
    images={int(i['id']):i for i in annotation['images']}
    annotations={int(a['id']):a for a in annotation['annotations']}
    base=arrays['baseline'];ids=base['annotation_id'];assert len(ids)==len(set(ids))==len(annotations)==38759
    assert len(images)==548 and set(map(int,ids))==set(annotations)
    projected=[]
    for index,g in enumerate(ids):
        a=annotations[int(g)];im=images[int(a['image_id'])]
        assert a['image_id']==int(base['image_id'][index]) and a['category_id']==int(base['category_id'][index])
        assert a['area']==base['original_area'][index] and not a.get('iscrowd',0) and not a.get('ignore',0)
        assert im['width']>0 and im['height']>0
        projected.append([float(a['bbox'][2])*640/im['width'],float(a['bbox'][3])*640/im['height']])
    wh=np.asarray(projected,dtype=np.float64);assert wh.shape==(len(ids),2) and np.isfinite(wh).all() and (wh>0).all()
    side=wh.min(1);bands={label:(side>=a)&(side<b) for label,a,b in zip(LABELS,EDGES[:-1],EDGES[1:])}
    assert np.all(np.stack(list(bands.values())).sum(0)==1)
    masks={'all':np.ones(len(ids),dtype=bool),'COCO_small':base['original_area']<=1024}
    rows={};paired={}
    for scope,scope_mask in masks.items():
        rows[scope]={};paired[scope]={}
        for label,band in bands.items():
            mask=scope_mask&band;n=int(mask.sum());stats={}
            for arm,values in arrays.items():
                for k in ('annotation_id','image_id','category_id','original_area'):
                    assert np.array_equal(base[k],values[k]),k
                counts=[]
                for threshold_index in (0,5):
                    threshold=float(geometry.THRESHOLDS[threshold_index]);support=values['max_iou'][mask]>=threshold
                    hits=values['native_TP'][mask,threshold_index]
                    assert not (hits&~support[:,2]).any()
                    missed=int((~hits).sum());no_box=int((~support[:,2]).sum());box_no_hit=int((support[:,2]&~hits).sum())
                    assert missed==no_box+box_no_hit
                    counts.append({'IoU':threshold,'geometry_support_counts_initial_pre_final':support.sum(0).tolist(),
                        'geometry_support_percent_initial_pre_final':(100*support.mean(0)).tolist() if n else [None]*3,
                        'native_all_area_TP_count':int(hits.sum()),'native_all_area_TP_GT_micro_percent':100*float(hits.mean()) if n else None,
                        'native_missed_GT':missed,'missed_without_final_geometry_support':no_box,
                        'missed_with_final_geometry_support':box_no_hit})
                stats[arm]=counts
            rows[scope][label]={'GT':n,'GT_fraction_of_scope_percent':100*n/int(scope_mask.sum()),
                'COCO_small_GT':int((mask&(base['original_area']<=1024)).sum()),'outcomes':stats}
            if 'candidate' in arrays:
                metric=geometry.pair_metrics(base,arrays['candidate'],mask);metric.pop('transition_annotation_ids')
                paired[scope][label]=metric
    classes={str(c):{label:int((band&(base['category_id']==c)).sum()) for label,band in bands.items()} for c in range(10)}
    for arm in arrays:
        for idx,threshold_index in enumerate((0,5)):
            assert sum(rows['all'][label]['outcomes'][arm][idx]['native_all_area_TP_count'] for label in LABELS)==report[arm]['all']['native_TP_by_threshold'][threshold_index]
    result={'status':'INPUT_SCALE_DESCRIPTIVE_BREAKDOWN','source_stage_analysis':str(Path(stage_analysis).resolve()),
        'stage_report_sha256':sha(Path(stage_analysis)/'report.json'),'annotation_sha256':VAL_SHA,
        'seed':report['seed'],'input_transform':'Direct non-aspect-preserving Resize640x640, verified by upstream stage reader.',
        'size_definition':'min(original_bbox_width*640/original_image_width, original_bbox_height*640/original_image_height); fractional pixels; no rounding or clipping.',
        'bands_pixels':[[a,None if np.isinf(b) else b] for a,b in zip(EDGES[:-1],EDGES[1:])],
        'band_names':list(LABELS),'feature_strides_pixels':{'P2':4,'P3':8},'outcome_breakdown':rows,
        'class_counts_by_input_min_side':classes,'paired_changes_by_input_min_side':paired if 'candidate' in arrays else None,
        'new_model_forwards':0,'new_evaluator_calls':0,'training_updates':0,
        'scope':'Supplementary GT-micro geometry/native all-area matching description, not APs, COCO area-specific recall or a causal test. Size/category/occlusion/density may covary. Does not change the frozen fixed12 gates.'}
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    output.mkdir(parents=True,exist_ok=False);write(output/'report.json',result)
    np.savez_compressed(output/'input_sizes_per_GT.npz',annotation_id=ids,width640=wh[:,0],height640=wh[:,1],min_side640=side)
    write(output/'manifest.json',{'implementation_sha256':sha(__file__),'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('stage-analysis','annotation','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=run(a.stage_analysis,a.annotation,a.output)
    print(json.dumps({'status':r['status'],'seed':r['seed'],'all_GT_bins':{k:v['GT'] for k,v in r['outcome_breakdown']['all'].items()},'new_model_forwards':0}))
