"""Independent CPU verification of saved reassignments and their coverage gate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    root=Path.cwd(); packet=Path(a.packet); seed=a.seed
    base=root/f'outputs/native_augmented_supervision_20261010_v1/collection_seed{seed}/capture_v1/endpoint12'
    result=packet/f'seed{seed}_probe'
    manifest=read(result/'manifest.json')
    for name,digest in manifest['files'].items():assert sha(result/name)==digest,name
    report=read(result/'report.json'); protocol=root/'outputs/support_matching_20261011_v1/protocol.json'
    assert report['protocol_sha256']==sha(protocol)==sha(packet/'protocol.json')
    assert read(packet/'completion.json')['exit_code']==0
    assert all(report[k]==0 for k in ['model_forwards','BASE_forwards','backwards','optimizer_steps','training_epochs'])
    with (result/'per_image_groups.csv').open(encoding='utf8') as f:
        reported={(int(r['batch']),int(r['slot']),r['scope'],r['arm'],int(r['band'])):r for r in csv.DictReader(f)}
    assignments=read(result/'assignments.json'); assert len(assignments)==16*4*2*2
    group_by_batch={b:[r for r in assignments if r['batch']==b] for b in range(16)}
    verified=0; pairs=0; independent=[]
    for batch,records in group_by_batch.items():
        p=base/f'batch_{batch:03d}';ev=read(p/'evidence.json');_,_,h,w=ev['metadata']['input_shape']
        with np.load(p/'arrays.npz',allow_pickle=False) as z:
            for rec in records:
                s=rec['slot'];q=np.array(rec['query_rows'],dtype=int);g=np.array(rec['GT_rows'],dtype=int)
                gt=z[f'target.{s}.boxes'].astype(np.float64);ref=z['model.pre_outputs.pred_boxes'][s].astype(np.float64)
                assert len(q)==len(g)==len(np.unique(q)) and (q>=0).all() and (q<300).all()
                assert (g>=0).all() and (g<len(gt)).all()
                if rec['scope']=='MAL':assert len(g)==min(len(gt),300) and len(np.unique(g))==len(g)
                assert rec['source_annotation_ids']==z[f'target.{s}.verified_source_annotation_ids'][g].tolist()
                # Invert native interval endpoints directly, independently of
                # max(abs(required_distance)) used by the producer.
                selected_ref=ref[q]; gg=gt[g]
                gt_edges=np.concatenate((gg[:,:2]-gg[:,2:]/2,gg[:,:2]+gg[:,2:]/2),axis=1)
                lower=np.concatenate((selected_ref[:,:2]-1.5*selected_ref[:,2:],selected_ref[:,:2]-.5*selected_ref[:,2:]),axis=1)
                upper=np.concatenate((selected_ref[:,:2]+.5*selected_ref[:,2:],selected_ref[:,:2]+1.5*selected_ref[:,2:]),axis=1)
                reachable=((gt_edges>=lower)&(gt_edges<=upper)).all(1)
                short=(gt[:,2:]*np.array([w,h])).min(1)
                bands=np.select([short<4,short<8,short<16],[0,1,2],default=3)
                for band in range(4):
                    m=bands[g]==band; vals=dict(pairs=int(m.sum()),outside_pairs=int((~reachable[m]).sum()),
                        supervised_GT_rows=len(np.unique(g[m])),target_rows=int((bands==band).sum()))
                    key=(batch,s,rec['scope'],rec['arm'],band);r=reported[key]
                    for k,v in vals.items():assert int(r[k])==v,(key,k,r[k],v)
                    independent.append(dict(scope=rec['scope'],arm=rec['arm'],band=band,**vals));verified+=1
                pairs+=len(q)
    assert verified==len(reported)==1024
    tiny={}
    for arm in ('native','candidate'):
        selected=[r for r in independent if (r['scope'],r['arm'],r['band'])==('GO',arm,0)]
        totals={k:sum(r[k] for r in selected) for k in ('pairs','outside_pairs','supervised_GT_rows','target_rows')}
        totals['outside_fraction']=totals['outside_pairs']/totals['pairs'];tiny[arm]=totals
    gate=report['comparator_certified'] and tiny['candidate']['outside_fraction']<tiny['native']['outside_fraction'] and tiny['candidate']['supervised_GT_rows']>=tiny['native']['supervised_GT_rows']
    assert gate==report['manipulation_coverage_gate']
    value=dict(status='PASS_INDEPENDENT_INTERVAL_AND_COVERAGE_VERIFICATION',seed=seed,
        source_rows_verified=verified,assignment_pair_exposures_verified=pairs,GO_under4=tiny,
        preregistered_manipulation_coverage_gate=bool(gate),report_sha256=sha(result/'report.json'),
        verifier_sha256=sha(__file__),native_replay_certification=report['comparisons'],
        model_forwards=0,training_epochs=0,actual_optimizer_history=False,AP_measured=False,
        fallacy_scan_coverage='11/11',
        limits=['All predefined size bands and both seeds retained; no selected positive subgroup.',
            'Augmented GT exposures may repeat source annotations; no independent sample or population estimate.',
            'Reassignment uses known GT during training only; physical support is not attained IoU or AP.',
            'Changing the matched target can worsen current box/class quality; report those tradeoffs.',
            'Frozen BASE12 fixture is not evidence of a new optimizer trajectory or causal root.',
            'No parameter sweep, best-arm selection, repeated model execution, p-value, significance, or novelty claim.'])
    with Path(a.output).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
