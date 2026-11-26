"""Execute only matcher/criterion algebra on preserved tensors, never a detector."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)


def numpy_penalty(ref, gt):
    r, g = ref.astype(np.float64), gt.astype(np.float64)
    xy1, xy2 = g[:, :2] - g[:, 2:] / 2, g[:, :2] + g[:, 2:] / 2
    left = (r[:, None, :2] - xy1[None]) / (r[:, None, 2:] / 4 + 1e-16) - 2
    right = (xy2[None] - r[:, None, :2]) / (r[:, None, 2:] / 4 + 1e-16) - 2
    d = np.concatenate((left, right), -1)
    return np.clip((np.max(np.abs(d), -1) - 4) / 4, 0, 1), np.max(np.abs(d), -1) <= 4


def aligned_iou(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    a1, a2 = a[:, :2]-a[:, 2:]/2, a[:, :2]+a[:, 2:]/2
    b1, b2 = b[:, :2]-b[:, 2:]/2, b[:, :2]+b[:, 2:]/2
    wh = np.maximum(0, np.minimum(a2, b2)-np.maximum(a1, b1))
    inter = wh.prod(1)
    return inter / (np.maximum(0, a[:, 2:]).prod(1)+b[:, 2:].prod(1)-inter)


def pairs_equal(a, b):
    return all(np.array_equal(x.cpu().numpy(), y.cpu().numpy()) for aa, bb in zip(a, b) for x, y in zip(aa, bb))


def main(a):
    root, prefix = Path(a.root), Path(a.prefix)
    out = root/f'seed{a.seed}_probe'
    assert not out.exists()
    protocol = read(root/'protocol.json')
    for name, digest in protocol['source_code_sha256'].items():
        assert sha(root/'code'/name) == digest, name
    native = prefix/'DEIM/experiments/fine_residual_screen24_20261009_v2/native'
    assert sha(native/'engine/deim/matcher.py') == protocol['native_matcher_sha256']
    assert sha(native/'engine/deim/deim_criterion.py') == protocol['native_criterion_sha256']
    data = prefix/'DEIM/experiments/native_augmented_supervision_20261010_v1/capture_v1/endpoint12'
    for batch in range(16):
        folder = data/f'batch_{batch:03d}'
        bound = protocol['sources'][f'{a.seed}/{batch}']
        assert sha(folder/'manifest.json') == bound['manifest_sha256']
        for name, digest in bound['files'].items():
            assert sha(folder/name) == digest
    sys.path.insert(0, str(native))
    import torch
    import engine.deim.matcher as mm
    from engine.deim.deim_criterion import DEIMCriterion
    from support_matching import compile_matcher_forward, support_penalty, SupportMatchingBinding
    out.mkdir()
    write(out/'start.json', dict(seed=a.seed, protocol_sha256=sha(root/'protocol.json'),
        model_forwards=0, optimizer_steps=0, pid=__import__('os').getpid(),
        start_ticks=Path('/proc/self/stat').read_text().split()[21]))
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    config = dict(protocol['matcher_config']); config.pop('type')
    config['use_focal_loss'] = True
    native_matcher = mm.HungarianMatcher(**config)
    candidate = mm.HungarianMatcher(**config)
    modified, modified_sha = compile_matcher_forward(mm.HungarianMatcher, protocol['native_matcher_sha256'])
    candidate.forward = types.MethodType(modified, candidate)
    candidate._support_matching_penalty = None
    original_lsap = mm.linear_sum_assignment
    recorded_costs = []

    def capture_cost(cost):
        recorded_costs.append(np.asarray(cost).copy())
        return original_lsap(cost)

    mm.linear_sum_assignment = capture_cost
    rows, pair_records, source_refs = [], [], {}
    comparisons = dict(saved_native_assignment_equal=True, saved_native_GO_equal=True,
        zero_penalty_assignment_equal=True, zero_penalty_cost_exact=True,
        penalty_max_abs_error_vs_numpy64=0., native_replays=0,
        candidate_replays=0, zero_penalty_replays=0, partial_criterion_calls=0,
        partial_criterion_binding_matches_standalone=True)
    stage_names = ['model', 'model.aux_outputs.0', 'model.aux_outputs.1', 'model.pre_outputs', 'model.enc_aux_outputs.0']
    tie_rows = []
    for batch in range(16):
        folder = data/f'batch_{batch:03d}'
        ev = read(folder/'evidence.json')
        assert ev['metadata']['seed'] == a.seed and ev['metadata']['endpoint'] == 12
        _, _, height, width = ev['metadata']['input_shape']
        with np.load(folder/'arrays.npz', allow_pickle=False) as z:
            gt = [z[f'target.{i}.boxes'].copy() for i in range(4)]
            targets = [{k:torch.from_numpy(z[f'target.{i}.{k}'].copy()).to(device) for k in ('boxes','labels')} for i in range(4)]
            stages = [{k:torch.from_numpy(z[f'{s}.{k}'].copy()).to(device) for k in ('pred_boxes','pred_logits')} for s in stage_names]
            refs = stages[3]['pred_boxes']
            up, scale = torch.tensor([.5],device=device), torch.tensor([4.],device=device)
            penalty = support_penalty(refs, targets, up, scale)
            np_all, _ = numpy_penalty(refs.cpu().numpy().reshape(-1,4),np.concatenate(gt))
            err = float(np.abs(penalty.cpu().numpy().astype(np.float64)-np_all).max(initial=0))
            comparisons['penalty_max_abs_error_vs_numpy64'] = max(comparisons['penalty_max_abs_error_vs_numpy64'],err)
            native_indices, candidate_indices, original_costs = [], [], []
            for si, stage in enumerate(stages):
                recorded_costs.clear()
                oi = native_matcher(stage, targets, epoch=4)['indices']
                nc = [c.copy() for c in recorded_costs]
                assert len(nc) == 4
                original_costs.append(nc)
                saved = [(torch.from_numpy(z[f'matcher{si}.batch{i}.{k}'].copy()) for k in ('query_index','GT_row_index')) for i in range(4)]
                saved = [tuple(x) for x in saved]
                comparisons['saved_native_assignment_equal'] &= pairs_equal(oi,saved)
                candidate._support_matching_penalty = torch.zeros_like(penalty)
                recorded_costs.clear()
                identity = candidate(stage, targets, epoch=4)['indices']
                comparisons['zero_penalty_assignment_equal'] &= pairs_equal(oi,identity)
                comparisons['zero_penalty_cost_exact'] &= all(np.array_equal(x,y) for x,y in zip(nc,recorded_costs))
                candidate._support_matching_penalty = penalty
                ci = candidate(stage, targets, epoch=4)['indices']
                native_indices.append(oi); candidate_indices.append(ci)
                comparisons['native_replays'] += 1
                comparisons['zero_penalty_replays'] += 1
                comparisons['candidate_replays'] += 1
            native_go = DEIMCriterion._get_go_indices(None,native_indices[0],native_indices[1:])
            candidate_go = DEIMCriterion._get_go_indices(None,candidate_indices[0],candidate_indices[1:])
            event = next(e for e in ev['events'] if e.get('kind')=='loss_call' and e.get('loss')=='boxes' and e['stage']=='model')
            saved_go = [tuple(torch.from_numpy(z[f"{event['prefix']}.batch{i}.{k}"].copy()) for k in ('query_index','GT_row_index')) for i in range(4)]
            comparisons['saved_native_GO_equal'] &= pairs_equal(native_go,saved_go)
            # The saved fixture lacks distribution logits. Check the actual native
            # criterion wrapper for MAL/L1/GIoU only; no FGL/DN integration claim.
            if batch == 0:
                cm = mm.HungarianMatcher(**config)
                criterion = DEIMCriterion(cm,{'loss_bbox':5,'loss_giou':2,'loss_mal':1,'loss_fgl':.15,'loss_ddf':1.5},
                    ['mal','boxes','local'],alpha=.75,gamma=1.5,num_classes=10)
                binding = SupportMatchingBinding(criterion,protocol['native_matcher_sha256']).install()
                observed_indices=[]
                realforward=cm.forward
                def observe(*args,**kwargs):
                    value=realforward(*args,**kwargs); observed_indices.append(value['indices']);return value
                cm.forward=observe
                output=dict(stages[0],aux_outputs=stages[1:3],pre_outputs=stages[3],enc_aux_outputs=[stages[4]],
                    enc_meta={'class_agnostic':False},up=up,reg_scale=scale)
                with torch.no_grad():
                    losses=criterion(output,targets,epoch=4)
                assert len(observed_indices)==5 and all(torch.isfinite(v).all() for v in losses.values())
                comparisons['partial_criterion_binding_matches_standalone'] &= all(pairs_equal(x,y) for x,y in zip(observed_indices,candidate_indices))
                comparisons['partial_criterion_calls'] += 1
                write(out/'partial_criterion_binding.json',dict(losses={k:float(v) for k,v in losses.items()},
                    matcher_calls=5, missing_FGL_and_DN=True, no_model_forward=True,binding=binding.metadata()))
                binding.close()
            for slot in range(4):
                short = (gt[slot][:,2:].astype(np.float64)*np.array([width,height])).min(1)
                bands = np.select([short<4,short<8,short<16],[0,1,2],default=3)
                pp, reachable = numpy_penalty(refs[slot].cpu().numpy(),gt[slot])
                pred = stages[0]['pred_boxes'][slot].cpu().numpy()
                # Match frequency ties are reported; native tie handling is unchanged.
                for arm, group in [('native',native_indices),('candidate',candidate_indices)]:
                    counts=Counter((int(q),int(g)) for stage in group for q,g in zip(*stage[slot]))
                    byq=defaultdict(list)
                    for (q,g),n in counts.items():byq[q].append((g,n))
                    tied=sum(sum(n==max(nn for gg,nn in vals) for g,n in vals)>1 for vals in byq.values())
                    tie_rows.append(dict(seed=a.seed,batch=batch,slot=slot,arm=arm,top_frequency_tie_queries=tied))
                for scope, pairs in [('MAL',(native_indices[0][slot],candidate_indices[0][slot])),('GO',(native_go[slot],candidate_go[slot]))]:
                    ns=set(zip(*(x.cpu().numpy().tolist() for x in pairs[0])))
                    cs=set(zip(*(x.cpu().numpy().tolist() for x in pairs[1])))
                    for arm, pair in zip(('native','candidate'),pairs):
                        q,g=(x.cpu().numpy() for x in pair)
                        quality=aligned_iou(pred[q],gt[slot][g])
                        for band in range(4):
                            mask=bands[g]==band; qq,gg=q[mask],g[mask]; n=int(mask.sum())
                            rows.append(dict(seed=a.seed,batch=batch,slot=slot,scope=scope,arm=arm,band=band,
                                target_rows=int((bands==band).sum()),pairs=n,supervised_GT_rows=len(np.unique(gg)),
                                outside_pairs=int((~reachable[qq,gg]).sum()),IoU_sum=float(quality[mask].sum()),
                                final_native_cost_sum=float(original_costs[0][slot][qq,gg].sum(dtype=np.float64)),
                                high_IoU_pairs=int((quality[mask]>=.5).sum()),
                                high_IoU_pairs_not_preserved=int(sum(v>=.5 and (int(qi),int(gi)) not in (cs if arm=='native' else ns) for qi,gi,v in zip(qq,gg,quality[mask])))))
                        pair_records.append(dict(seed=a.seed,batch=batch,slot=slot,scope=scope,arm=arm,
                            query_rows=q.tolist(),GT_rows=g.tolist(),source_annotation_ids=z[f'target.{slot}.verified_source_annotation_ids'][g].tolist()))
        source_refs[str(folder)] = sha(folder/'manifest.json')
        print(json.dumps(dict(progress='SAVED_MATCHING_BATCH_COMPLETE',seed=a.seed,batch=batch)),flush=True)
    mm.linear_sum_assignment = original_lsap
    for name,data_rows in [('per_image_groups.csv',rows),('GO_ties.csv',tie_rows)]:
        with (out/name).open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
    write(out/'assignments.json',pair_records)
    totals=[]
    for scope in ('MAL','GO'):
        for arm in ('native','candidate'):
            for band in range(4):
                rr=[r for r in rows if (r['scope'],r['arm'],r['band'])==(scope,arm,band)]
                keys=[k for k in rr[0] if k not in ('seed','batch','slot','scope','arm','band')]
                d=dict(scope=scope,arm=arm,band=band,**{k:sum(r[k] for r in rr) for k in keys})
                d['outside_fraction']=d['outside_pairs']/max(d['pairs'],1)
                d['mean_current_matched_IoU']=d['IoU_sum']/max(d['pairs'],1)
                totals.append(d)
    native_tiny=next(r for r in totals if (r['scope'],r['arm'],r['band'])==('GO','native',0))
    candidate_tiny=next(r for r in totals if (r['scope'],r['arm'],r['band'])==('GO','candidate',0))
    replay_ok=all(comparisons[k] for k in ('saved_native_assignment_equal','saved_native_GO_equal',
        'zero_penalty_assignment_equal','zero_penalty_cost_exact','partial_criterion_binding_matches_standalone'))
    replay_ok &= comparisons['penalty_max_abs_error_vs_numpy64'] < 1e-4
    gate = replay_ok and candidate_tiny['outside_fraction']<native_tiny['outside_fraction'] and candidate_tiny['supervised_GT_rows']>=native_tiny['supervised_GT_rows']
    report=dict(status='COMPLETED_SAVED_MATCHING_PROBE',seed=a.seed, comparisons=comparisons,
        comparator_certified=bool(replay_ok), manipulation_coverage_gate=bool(gate),totals=totals,
        protocol_sha256=sha(root/'protocol.json'),compiled_matcher_source_sha256=modified_sha,source_manifests=source_refs,
        runtime=dict(torch=str(torch.__version__),numpy=np.__version__,CUDA=torch.version.cuda),
        model_forwards=0,BASE_forwards=0,backwards=0,optimizer_steps=0,training_epochs=0,
        actual_training_history=False,AP_or_efficacy_measured=False,novelty_or_root_established=False)
    write(out/'report.json',report)
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(status=report['status'],seed=a.seed,comparator_certified=bool(replay_ok),gate=bool(gate),native_tiny=native_tiny,candidate_tiny=candidate_tiny)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prefix',required=True);p.add_argument('--seed',type=int,choices=(1,2),required=True)
    main(p.parse_args())
