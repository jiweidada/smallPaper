"""Check GT partitions and stable-rank selections by direct cutoff inequalities."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(packet,arm,protocol_path,analysis,output):
    packet,protocol_path,analysis,output=map(Path,(packet,protocol_path,analysis,output));assert not output.exists()
    p=read(protocol_path);r=read(analysis/'report.json');assert r['protocol_sha256']==sha(protocol_path)
    for name,h in read(analysis/'manifest.json')['files'].items():assert sha(analysis/name)==h
    folder=packet/'baseline_eval' if arm=='baseline' else packet/'run';ep=folder/'endpoint_epoch12'
    with np.load(analysis/'per_GT.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(ep/'GT_hits.npz',allow_pickle=False) as z:flags={k:z[k].copy() for k in z.files}
    assert np.array_equal(data['annotation_id'],flags['annotation_id']) and np.array_equal(data['category_id'],flags['category_id'])
    checks=read(analysis/'image_score_checks.json');ann=read(packet/'reference/val_annotations.json')
    assert set(data['annotation_id'].tolist())=={int(a['id']) for a in ann['annotations']}
    raw_hits=np.zeros_like(flags['hits']);index={int(g):i for i,g in enumerate(flags['annotation_id'])}
    with gzip.open(ep/'matching_all_area.json.gz','rt',encoding='utf8') as f:matching=json.load(f)
    for row in matching:
        match,ignore=np.array(row['dtMatches']),np.array(row['dtIgnore'],dtype=bool)
        for ti in range(10):
            for g in match[ti][(match[ti]!=0)&~ignore[ti]].astype(int):raw_hits[index[int(g)],ti]=True
    assert np.array_equal(raw_hits,flags['hits'])
    tested=0;floor=np.log(p['score_threshold']/(1-p['score_threshold']))
    for iid in range(548):
        check=checks[str(iid)]
        if check['ranking_ambiguous']:continue
        assert not check['native_score_list_mismatches'] and check['native_score_list_max_abs_error']<=p['native_score_abs_tolerance']
        with np.load(folder/f'stages_epoch12/image_{iid:06d}.npz',allow_pickle=False) as z:
            logits=z['final_logits'].astype(np.float64);boxes=z['final_boxes'].copy();hw=z['original_hw'].copy()
        center=boxes[:,:2].astype(np.float32);size=boxes[:,2:].astype(np.float32)
        xy=np.concatenate((center-size*.5,center+size*.5),axis=1)
        xy*=np.array([hw[1],hw[0],hw[1],hw[0]],dtype=np.float32)
        xy[:,2:]-=xy[:,:2];xy=xy.astype(np.float64)
        valid=(boxes[:,2:]>0).all(1)
        rows=np.flatnonzero(data['image_id']==iid)
        by_id={int(a['id']):a for a in ann['annotations'] if a['image_id']==iid}
        for row in rows:
            a=by_id[int(data['annotation_id'][row])];gt=np.array(a['bbox'],dtype=np.float64)
            overlap=np.maximum(0,np.minimum(gt[:2]+gt[2:],xy[:,:2]+xy[:,2:])-np.maximum(gt[:2],xy[:,:2]))
            inter=overlap.prod(1);union=gt[2:].prod()+xy[:,2:].prod(1)-inter
            iou=np.zeros(300);iou[valid]=inter[valid]/union[valid]
            cls=int(a['category_id']);global_cut=check['global_logit_cutoff'];class_cut=check['class100_logit_cutoffs'][cls]
            global_keep=(logits[:,cls]>=global_cut)&(logits[:,cls]>floor)
            class_keep=(logits[:,cls]>=max(global_cut,class_cut))&(logits[:,cls]>floor)
            for threshold,ti,tag in ((.5,0,'50'),(.75,5,'75')):
                support=iou>=threshold
                assert bool(support.any())==bool(data[f'geometry_{tag}'][row])
                assert bool((support&global_keep).any())==bool(data[f'global_true_class_{tag}'][row])
                assert bool((support&class_keep).any())==bool(data[f'retained_true_class_{tag}'][row])
            tested+=1
    for tag,ti in (('50',0),('75',5)):
        assert np.array_equal(data[f'native_TP_{tag}'],raw_hits[:,ti])
        for j in range(len(data['annotation_id'])):
            if data[f'native_TP_{tag}'][j]:expected='native_TP'
            elif not data[f'geometry_{tag}'][j]:expected='no_final_geometric_support'
            elif data['ranking_ambiguous'][j]:expected='uncertain_ranking_reconstruction'
            elif not data[f'global_true_class_{tag}'][j]:expected='no_true_class_support_after_top300_and_score_threshold'
            elif not data[f'retained_true_class_{tag}'][j]:expected='true_class_support_removed_by_class100'
            else:expected='true_class_retained_geometry_without_native_TP'
            assert data[f'cell_{tag}'][j]==expected
    assert not r['score_mismatch_images'] and r['native_TP_proxy_conflicts']==0
    result=dict(status='PASS_PARTITION_AND_STABLE_RANK_CUTOFF_CHECKS',seed=r['identity']['seed'],arm=arm,
        GT=38759,stable_image_GT_geometry_and_rank_checks=tested,ambiguous_images=r['ambiguous_images'],
        native_GT_flags_rebuilt_exact=True,all_GT_partitions_exact=True,protocol_sha256=sha(protocol_path),
        report_sha256=sha(analysis/'report.json'),verifier_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
        scope='Exact partition/count arithmetic and stable-logit cutoff checks. Native CUDA query identities are not archived; near-cutoff images remain uncertain. This validates a bounded diagnostic proxy, not a new detector or causal claim.',
        statistical_fallacy_checks=dict(total=11,checked=11,notes=['NotAP','Notrootcause','No bestepoch selection',
            'No native flags rewritten','Ambiguous ranks preserved','Both seeds retained','Overlapping groups not summed',
            'Nativezero-match sentinel retained','No unique duplicate/greedy cause inferred',
            'No model/evaluator rerun','No primary or mechanism gate changed']))
    assert 'torch' not in sys.modules
    with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('packet','arm','protocol','analysis','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(verify(a.packet,a.arm,a.protocol,a.analysis,a.output)),flush=True)
