"""Validate maximum-allocation certificates without rerunning matching search."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(protocol_path,analysis,output):
    protocol_path,analysis,output=map(Path,(protocol_path,analysis,output))
    assert not output.exists()
    protocol=read(protocol_path);report=read(analysis/'report.json')
    assert report['protocol_sha256']==sha(protocol_path)
    for name,h in read(analysis/'manifest.json')['files'].items():assert sha(analysis/name)==h
    bound=read(protocol['range_protocol']);checks={}
    for seed,spec in bound['inputs'].items():
        root=Path(spec['packet']);ann=read(root/'reference/val_annotations.json')
        images={int(im['id']):im for im in ann['images']};grouped={i:[] for i in images}
        for a in ann['annotations']:grouped[int(a['image_id'])].append(a)
        certs=read(analysis/f'seed{seed}_certificates.json')
        with np.load(analysis/f'seed{seed}_per_GT.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        index={int(g):i for i,g in enumerate(data['annotation_id'])}
        assert set(index)=={int(a['id']) for a in ann['annotations']}
        sums={k:dict(GT=0,individually_reachable=0,maximum_assignable=0) for k in
              ('under4_only','COCO_small_only','all_tiny_priority')};verified=0
        for iid,im in sorted(images.items()):
            entries=sorted(grouped[iid],key=lambda a:int(a['id']))
            ids=[int(a['id']) for a in entries];local={g:i for i,g in enumerate(ids)}
            row=np.array([index[g] for g in ids],dtype=np.int64)
            with np.load(root/f'baseline_eval/stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:
                ref=z['pre_after_first_layer_boxes'].astype(np.float64)
            boxes=np.array([a['bbox'] for a in entries],dtype=np.float64).reshape(-1,4)
            normalized=boxes/np.array([im['width'],im['height'],im['width'],im['height']])
            start,end=normalized[:,:2],normalized[:,:2]+normalized[:,2:]
            left_top=4*(ref[None,:,:2]-start[:,None,:])/ref[None,:,2:]-2
            right_bottom=4*(end[:,None,:]-ref[None,:,:2])/ref[None,:,2:]-2
            edges=(np.abs(np.concatenate((left_top,right_bottom),axis=-1))<=4).all(-1)
            cert=certs[str(iid)]
            assert hashlib.sha256(edges.tobytes()).hexdigest()==cert['edge_sha256']
            assert np.array_equal(edges.any(1),data['range_reachable'][row])
            tiny=normalized[:,2:].min(1)*640<4
            masks=dict(under4_only=tiny,COCO_small_only=np.array([a['area'] for a in entries])<=1024,
                       all_tiny_priority=np.ones(len(ids),dtype=bool))
            for name,mask in masks.items():
                proof=cert[name];active={g for g,yes in zip(ids,mask) if yes}
                pairs=proof['matching'];left=[int(g) for g,q in pairs];right=[int(q) for g,q in pairs]
                assert len(left)==len(set(left))==len(right)==len(set(right))
                assert set(left)<=active and all(0<=q<300 for q in right)
                assert all(edges[local[g],q] for g,q in pairs)
                cover_left=set(proof['cover_left_annotation_ids']);cover_right=set(proof['cover_right_query_rows'])
                assert cover_left<=active and all(0<=q<300 for q in cover_right)
                assert len(cover_left)+len(cover_right)==len(pairs)==proof['optimal_cardinality']
                for g in active-cover_left:assert set(np.flatnonzero(edges[local[g]]))<=cover_right
                assert proof['active_GT']==int(mask.sum())
                assert proof['reachable_GT']==int((mask&edges.any(1)).sum())
                for k,source in [('GT','active_GT'),('individually_reachable','reachable_GT'),('maximum_assignable','optimal_cardinality')]:
                    sums[name][k]+=proof[source]
                column={'under4_only':'under4_only_assigned','COCO_small_only':'COCO_small_only_assigned',
                        'all_tiny_priority':'all_tiny_priority_assigned'}[name]
                assert np.array_equal(data[column][row],np.array([g in set(left) for g in ids]))
                if name=='all_tiny_priority':
                    mapping=dict(pairs)
                    assert data['assigned_query_row'][row].tolist()==[mapping.get(g,-1) for g in ids]
                verified+=1
            tiny_all=sum(tiny[local[int(g)]] for g,q in cert['all_tiny_priority']['matching'])
            assert tiny_all==cert['under4_only']['optimal_cardinality']
        for name,s in sums.items():
            saved=report['seeds'][seed][name]
            for key,v in s.items():assert saved[key]==v
            assert saved['range_excluded']==s['GT']-s['individually_reachable']
            assert saved['additional_one_to_one_deficit']==s['individually_reachable']-s['maximum_assignable']
        assert verified==548*3
        checks[seed]=dict(images=548,GT=38759,maximum_matching_certificates_independently_verified=verified,
            matching_edges_and_unique_assignment_valid=True,equal_cardinality_cover_covers_every_edge=True,
            tiny_priority_preserves_maximum_tiny_count=True,sums=sums)
    result=dict(status='PASS_INDEPENDENT_MAXIMUM_ALLOCATION_CERTIFICATES',checks=checks,
        protocol_sha256=sha(protocol_path),report_sha256=sha(analysis/'report.json'),
        verifier_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
        material_passport=dict(data='Frozen VisDroneval548/38759GT',weights='Two existing BASE12 EMA traces',
            hypothesis='Individual range reach versus one-query-per-GT allocation capacity',scope=protocol['scope']),
        statistical_fallacy_checks=dict(total=11,checked=11,notes=[
            'Oracle not deployed','Not an AP upper bound without stated one-to-one assumption',
            'No class/ranking information inferred','No root cause inferred','No feature recoverability inferred',
            'No significance claim','No repeated-seed independence assumption','Overlapping groups not summed',
            'Both seeds retained','Specific unmatched GT depend on matching ties','No efficacy gate changed']))
    assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
    with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('protocol','analysis','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(verify(a.protocol,a.analysis,a.output)),flush=True)
