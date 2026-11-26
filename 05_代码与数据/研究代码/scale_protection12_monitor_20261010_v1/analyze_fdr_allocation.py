"""GT-assisted one-query-per-GT range allocation with minimum-cover certificates."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def maximum(edges,order,annotation_ids):
    adjacency=[np.flatnonzero(r).tolist() for r in edges]
    assigned={}
    def augment(left,seen):
        for right in adjacency[left]:
            if right in seen: continue
            seen.add(right)
            if right not in assigned or augment(assigned[right],seen):
                assigned[right]=left
                return True
        return False
    for left in order: augment(int(left),set())
    matched={left:right for right,left in assigned.items()}
    active=set(map(int,order));assert set(matched)<=active
    visited_left=active-set(matched);visited_right=set();queue=deque(sorted(visited_left))
    while queue:
        left=queue.popleft()
        for right in adjacency[left]:
            if matched.get(left)==right or right in visited_right: continue
            visited_right.add(right)
            if right in assigned and assigned[right] not in visited_left:
                visited_left.add(assigned[right]);queue.append(assigned[right])
    cover_left=active-visited_left;cover_right=visited_right
    assert len(cover_left)+len(cover_right)==len(matched)
    # A matching and equal-cardinality vertex cover independently certify optimality.
    for left in active:
        assert left in cover_left or all(r in cover_right for r in adjacency[left])
    assert len(set(matched.values()))==len(matched)
    assert all(edges[l,r] for l,r in matched.items())
    return matched,dict(matching=[[int(annotation_ids[l]),int(r)] for l,r in sorted(matched.items())],
        cover_left_annotation_ids=[int(annotation_ids[l]) for l in sorted(cover_left)],
        cover_right_query_rows=sorted(cover_right),optimal_cardinality=len(matched),
        reachable_GT=sum(bool(adjacency[l]) for l in active),active_GT=len(active),
        every_edge_covered=True,matching_and_cover_size_equal=True)


def analyze(protocol_path,output):
    protocol_path,output=Path(protocol_path),Path(output)
    assert not output.exists()
    p=read(protocol_path);assert p['schema']=='sod_fdr_allocation_v1' and sha(__file__)==p['analysis_code_sha256']
    range_protocol=Path(p['range_protocol']);assert sha(range_protocol)==p['range_protocol_sha256']
    reference=read(range_protocol);output.mkdir()
    totals,provenance={},{}
    for seed,spec in reference['inputs'].items():
        root=Path(spec['packet']);annotation=read(root/'reference/val_annotations.json')
        assert sha(root/'reference/val_annotations.json')==spec['files']['reference/val_annotations.json']
        images={int(x['id']):x for x in annotation['images']};grouped={k:[] for k in images}
        for a in annotation['annotations']: grouped[int(a['image_id'])].append(a)
        stage=root/'baseline_eval/stages_epoch12';manifest=read(stage/'manifest.json')
        assert sha(stage/'manifest.json')==spec['files']['baseline_eval/stages_epoch12/manifest.json']
        with np.load(Path(p['range_analysis'])/f'seed{seed}_per_GT.npz',allow_pickle=False) as z:
            prior={k:z[k].copy() for k in z.files}
        prior_index={int(g):i for i,g in enumerate(prior['annotation_id'])}
        certificates={};records=[];summary={}
        for iid,im in sorted(images.items()):
            path=stage/f'image_{iid:06d}.npz';assert sha(path)==manifest['files'][path.name]
            with np.load(path,allow_pickle=False) as z:ref=z['pre_after_first_layer_boxes'].astype(np.float64)
            assert ref.shape==(300,4) and (ref[:,2:]>0).all() and np.isfinite(ref).all()
            entries=sorted(grouped[iid],key=lambda a:int(a['id']));ids=np.array([int(a['id']) for a in entries],dtype=np.int64)
            original=np.array([a['bbox'] for a in entries],dtype=np.float64).reshape(-1,4)
            gt=original/np.array([im['width'],im['height'],im['width'],im['height']])
            lo,hi=gt[:,:2],gt[:,:2]+gt[:,2:];c,w=ref[:,:2],ref[:,2:]
            edges=((lo[:,None,:]>=c[None,:,:]-1.5*w[None,:,:]) &
                   (lo[:,None,:]<=c[None,:,:]+.5*w[None,:,:]) &
                   (hi[:,None,:]>=c[None,:,:]-.5*w[None,:,:]) &
                   (hi[:,None,:]<=c[None,:,:]+1.5*w[None,:,:])).all(-1)
            row=np.array([prior_index[int(g)] for g in ids],dtype=np.int64)
            assert np.array_equal(edges.any(1),prior['any_closed_range'][row])
            assert np.array_equal(edges.sum(1),prior['reachable_query_count'][row])
            tiny=gt[:,2:].min(1)*640<4
            small=np.array([a['area'] for a in entries])<=1024
            priority=np.concatenate((np.flatnonzero(tiny),np.flatnonzero(~tiny)))
            tiny_match,tiny_cert=maximum(edges,np.flatnonzero(tiny),ids)
            small_match,small_cert=maximum(edges,np.flatnonzero(small),ids)
            all_match,all_cert=maximum(edges,priority,ids)
            assert sum(tiny[l] for l in all_match)==len(tiny_match)
            assert len(tiny_match)<=int((tiny&edges.any(1)).sum())
            certificates[str(iid)]=dict(edge_sha256=hashlib.sha256(edges.tobytes()).hexdigest(),
                image_GT=len(ids),one_query_per_GT=True,under4_only=tiny_cert,
                COCO_small_only=small_cert,all_tiny_priority=all_cert)
            for j,g in enumerate(ids):
                records.append(dict(annotation_id=int(g),image_id=iid,input_shortside=float(gt[j,2:].min()*640),
                    original_area=float(entries[j]['area']),range_reachable=bool(edges[j].any()),
                    under4_only_assigned=j in tiny_match,COCO_small_only_assigned=j in small_match,
                    all_tiny_priority_assigned=j in all_match,assigned_query_row=all_match.get(j,-1),
                    final_IoU75_failed=bool(prior['max_iou'][row[j],2]<.75)))
            for name,cert in (('under4_only',tiny_cert),('COCO_small_only',small_cert),('all_tiny_priority',all_cert)):
                s=summary.setdefault(name,dict(GT=0,individually_reachable=0,maximum_assignable=0))
                s['GT']+=cert['active_GT'];s['individually_reachable']+=cert['reachable_GT'];s['maximum_assignable']+=cert['optimal_cardinality']
        records.sort(key=lambda x:x['annotation_id'])
        assert len(records)==38759
        data={k:np.array([r[k] for r in records]) for k in records[0]}
        np.savez_compressed(output/f'seed{seed}_per_GT.npz',**data)
        write(output/f'seed{seed}_certificates.json',certificates)
        assert summary['under4_only']['GT']==5096 and summary['COCO_small_only']['GT']==26586
        for s in summary.values():
            s['range_excluded']=s['GT']-s['individually_reachable']
            s['additional_one_to_one_deficit']=s['individually_reachable']-s['maximum_assignable']
            s['not_simultaneously_assignable']=s['GT']-s['maximum_assignable']
        summary['under4_only']['matching_choices_not_unique']=True
        totals[seed]=summary
        provenance[seed]=dict(stage_manifest_sha256=sha(stage/'manifest.json'),images=548,
            certificate_sha256=sha(output/f'seed{seed}_certificates.json'),
            all_three_matching_sizes_certified_by_equal_vertex_covers=True)
    report=dict(status='GT_ASSISTED_FDR_ALLOCATION_ORACLE_COMPLETED',seeds=totals,provenance=provenance,
        protocol_sha256=sha(protocol_path),new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
        scope=p['scope'],interpretation=p['interpretation'])
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files={q.name:sha(q) for q in output.iterdir() if q.is_file()}))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    return totals


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(analyze(a.protocol,a.output),ensure_ascii=False),flush=True)
