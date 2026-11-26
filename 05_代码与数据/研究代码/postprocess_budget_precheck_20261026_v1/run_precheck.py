"""Frozen cached-only decoder selection controls. No network or model construction."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.coco
import pycocotools.cocoeval
import pycocotools._mask

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
EN = ROOT.parents[1]
WORK = EN/'code/unit_instance_completion_20261021_v1'
CACHE = WORK/'342_codex_stage3_20261022_v1'
S343 = WORK/'343_codex_stage4_3b_20261022_v1'
REF = WORK/'344_codex_closeout_20261022_v1/344_result_独立pycocotools_baseline_v2.json'
AUDIT = EN/'code/problem_definition_audit_20261026_v1/cache_contract_audit.json'
NAMES = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
ARMS = ('A0', 'A1', 'A2', 'A3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    with (ROOT/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def progress(obj):
    print(json.dumps(obj, ensure_ascii=True), flush=True)


def to_edges(b, scale):
    cx,cy,w,h = b.unbind(-1)
    return torch.stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2), -1)*scale


def pair_iou(gt, pred):
    wh = (torch.minimum(gt[:,None,2:],pred[None,:,2:])
          -torch.maximum(gt[:,None,:2],pred[None,:,:2])).clamp(min=0)
    inter = wh.prod(-1)
    ag = (gt[:,2:]-gt[:,:2]).clamp(min=0).prod(-1)
    ap = (pred[:,2:]-pred[:,:2]).clamp(min=0).prod(-1)
    return inter/(ag[:,None]+ap[None,:]-inter).clamp(min=1e-12)


def max_match(a):
    if a.size == 0 or not a.any():
        return 0
    return int((maximum_bipartite_matching(csr_matrix(a),perm_type='column') >= 0).sum())


def choose(prob, legal, arm):
    pool = prob.clone()
    if arm in ('A1','A3'):
        mask = torch.zeros_like(pool, dtype=torch.bool)
        mask.scatter_(1, pool.argmax(1, keepdim=True), True)
        pool[~mask] = -torch.inf
    elif arm == 'A2':
        mask = torch.zeros_like(pool, dtype=torch.bool)
        mask.scatter_(1, pool.topk(2,dim=1).indices, True)
        pool[~mask] = -torch.inf
    if arm == 'A3':
        pool[~legal] = -torch.inf
    score,flat = pool.flatten().topk(300)
    flat = flat[score > .001]
    pre = flat.clone()
    counts = [0]*10
    selected = []
    for i in flat.tolist():
        c = i % 10
        if counts[c] < 100:
            selected.append(i)
            counts[c] += 1
    return torch.tensor(selected,dtype=torch.long),pre


def make_rows(image_id, coords, prob, flat):
    q,c = flat//10,flat%10
    boxes = coords[q]
    assert bool(torch.isfinite(boxes).all()) and bool((boxes[:,2:] > boxes[:,:2]).all())
    wh = boxes[:,2:] - boxes[:,:2]
    xywh = torch.cat((boxes[:,:2], wh),1)
    return [{'image_id':int(image_id),'category_id':int(label),'bbox':box.tolist(),'score':float(s)}
            for label,box,s in zip(c,xywh,prob.flatten()[flat])]


def evaluate(dataset, rows, tiny_ids):
    log = io.StringIO()
    start = time.monotonic()
    with contextlib.redirect_stdout(log):
        gt = COCO()
        gt.dataset = dataset
        gt.createIndex()
        dt = gt.loadRes(rows)
        ev = COCOeval(gt,dt,'bbox')
        assert ev.params.maxDets == [1,10,100]
        ev.params.imgIds = sorted(gt.getImgIds())
        assert len(ev.params.imgIds) == 548
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = {k:float(v) for k,v in zip(NAMES,ev.stats)}
    matched = set()
    tp = fp = 0
    for e in ev.evalImgs:
        if e is None or e['aRng'] != ev.params.areaRng[0] or e['maxDet'] != 100:
            continue
        good = ~np.asarray(e['dtIgnore'][0],dtype=bool)
        gm = np.asarray(e['dtMatches'][0])
        tp += int(((gm > 0)&good).sum())
        fp += int(((gm == 0)&good).sum())
        matched.update(int(x) for x in gm[good & (gm > 0)])
    assert tp == len(matched)
    return {'stats12':stats,'TP50':tp,'FP50':fp,'tiny_TP50':len(matched & tiny_ids),
            'detections':len(rows),'wall_seconds':time.monotonic()-start},matched,log.getvalue()


def main():
    start = time.monotonic()
    ref = json.loads(REF.read_text(encoding='utf-8'))
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    manifest = {x['path']:x['sha256'] for x in audit['input_manifest']}
    ann = Path(ref['per_seed'][0]['annotation_path'])
    dataset = json.loads(ann.read_text(encoding='utf-8'))
    dataset.setdefault('info',{})
    assert len(dataset['annotations']) == 38759
    protected = [ann,REF,AUDIT,ROOT/'protocol.txt',Path(__file__)]
    protected += [S343/f'343_val_seed{s}_baseline_predictions.json' for s in (1,2)]
    protected_hashes = {str(p):sha(p) for p in protected}
    assert protected_hashes[str(ann)] == ref['per_seed'][0]['annotation_sha256']
    seeds = {}
    all_tiny = None
    for seed in (1,2):
        files = sorted((CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt'))
        assert len(files) == 548
        tiny_ids = set()
        rows = {a:[] for a in ARMS}
        invalid = {a:[] for a in ARMS}
        unique = {a:[] for a in ARMS}
        before_caps = {a:0 for a in ARMS}
        ceiling = {'raw_all_pairs_budget300':0,'original_top300':0,'original_export_classcap100':0,
                   'tiny_only_raw_all_pairs':0,'tiny_only_original_top300':0}
        legality_noop = True
        for path in files:
            b = path.read_bytes()
            assert hashlib.sha256(b).hexdigest() == manifest[str(path)]
            d = torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
            image_id = int(d['image_id'])
            raw = d['final300_boxes']
            prob = d['final300_logits'].sigmoid()
            legal = torch.isfinite(raw).all(1) & (raw[:,2:] > 0).all(1)
            coords = to_edges(raw,torch.as_tensor(d['orig_wh'],dtype=raw.dtype).repeat(2))
            legal &= (coords[:,2:] > coords[:,:2]).all(1)
            I = pair_iou(d['gt_boxes_640'].float(),to_edges(raw.float()*640.,1.))
            geom = (I >= .5).numpy()
            gl = d['gt_labels'].numpy()
            gtids = list(d['gt_annotation_ids'])
            tiny = (d['gt_short_side_640_all'] < 4).numpy()
            tiny_ids.update(int(x) for x,t in zip(gtids,tiny) if t)
            raw_n = raw_tiny = 0
            for c in range(10):
                ix = gl == c
                graph = geom[ix] & (prob[:,c].numpy()[None,:] > .001)
                raw_n += max_match(graph)
                raw_tiny += max_match(graph[tiny[ix]])
            ceiling['raw_all_pairs_budget300'] += min(raw_n,300)
            ceiling['tiny_only_raw_all_pairs'] += min(raw_tiny,300)
            selections = {}
            for arm in ARMS:
                flat,pre = choose(prob,legal,arm)
                selections[arm] = (flat,pre)
                before_caps[arm] += len(pre)
                q = flat//10
                unique[arm].append(len(q.unique()))
                bad = q[~legal[q]].unique().tolist()
                if bad:
                    invalid[arm].append({'image_id':image_id,'query_ids':bad,
                                         'raw_cxcywh':[raw[x].tolist() for x in bad]})
                else:
                    rows[arm].extend(make_rows(image_id,coords,prob,flat))
            a0,pre = selections['A0']
            # Applying A3's valid-query domain to A0 must make no difference.
            pp = prob.clone();pp[~legal] = -torch.inf
            v,k = pp.flatten().topk(300)
            legality_noop &= torch.equal(k[v > .001],pre)
            assert torch.equal(pre,d['fullp_flatten_index'])
            for key,flat in (('original_top300',pre),('original_export_classcap100',a0)):
                graph = geom[:,(flat//10).numpy()] & (gl[:,None] == (flat%10).numpy()[None,:])
                ceiling[key] += max_match(graph)
                if key == 'original_top300':
                    ceiling['tiny_only_original_top300'] += max_match(graph[tiny])
        assert len(tiny_ids) == 4952 and legality_noop
        if all_tiny is not None:
            assert tiny_ids == all_tiny
        all_tiny = tiny_ids
        prior_rows = json.loads((S343/f'343_val_seed{seed}_baseline_predictions.json').read_text(encoding='utf-8'))
        assert rows['A0'] == prior_rows, 'A0 output does not reproduce 343'
        progress({'stage':'selection_ready','seed':seed,'invalid':{a:len(v) for a,v in invalid.items()},'ceiling':ceiling})
        result = {'baseline_rows_exact_equal_343':True,'baseline_valid_pool_noop':True,
                  'matching_ceiling_NOT_TP':ceiling,'arms':{}}
        base_matched = None
        for arm in ARMS:
            if invalid[arm]:
                result['arms'][arm] = {'valid':False,'stats12':None,'invalid_output':invalid[arm],
                                        'reason':'Frozen positive-width/height gate failed; no filtering or metrics.'}
                progress({'stage':'invalid','seed':seed,'arm':arm})
                continue
            metrics,matched,log = evaluate(dataset,rows[arm],tiny_ids)
            with (ROOT/f'coco_seed{seed}_{arm}.txt').open('x',encoding='utf-8') as f:
                f.write(log)
            metrics.update(valid=True,mean_distinct_queries=float(np.mean(unique[arm])),
                           detections_before_classcap=before_caps[arm])
            if arm == 'A0':
                diffs = {k:metrics['stats12'][k]-ref['per_seed'][seed-1]['stats12'][k] for k in NAMES}
                assert max(abs(x) for x in diffs.values()) <= 1e-12, diffs
                metrics['minus_344_stats12'] = diffs
                base_matched = matched
            else:
                baseline = result['arms']['A0']
                metrics['delta_stats12'] = {k:metrics['stats12'][k]-baseline['stats12'][k] for k in NAMES}
                metrics['delta_TP50'] = metrics['TP50']-baseline['TP50']
                metrics['delta_FP50'] = metrics['FP50']-baseline['FP50']
                metrics['tiny_TP_gained'] = len((matched-base_matched)&tiny_ids)
                metrics['tiny_TP_lost'] = len((base_matched-matched)&tiny_ids)
                metrics['all_TP_gained'] = len(matched-base_matched)
                metrics['all_TP_lost'] = len(base_matched-matched)
            result['arms'][arm] = metrics
            progress({'stage':'evaluated','seed':seed,'arm':arm,**metrics})
        a2 = result['arms']['A2']
        gates = {'valid':a2['valid']}
        if a2['valid']:
            gates.update(AP_gain_ge_0_1pp=a2['delta_stats12']['AP']>=.001,
                         APs_gain_ge_0_1pp=a2['delta_stats12']['APs']>=.001,
                         tiny_TP_strictly_up=a2['tiny_TP_gained']>a2['tiny_TP_lost'],
                         total_TP_not_down=a2['delta_TP50']>=0,
                         total_FP_not_up=a2['delta_FP50']<=0)
        result['A2_gates'] = gates
        result['A2_pass'] = all(gates.values())
        seeds[str(seed)] = result
        write(f'result_seed{seed}.json',result)
    for p,h in protected_hashes.items():
        assert sha(p) == h, 'Protected input changed: '+p
    for p,h in manifest.items():
        assert sha(p) == h, 'Cache changed: '+p
    output = {'protocol_sha256':sha(ROOT/'protocol.txt'),'per_seed':seeds,
              'primary_A2_pass_both':all(s['A2_pass'] for s in seeds.values()),
              'tiny_annotation_ids_sha256':hashlib.sha256(json.dumps(sorted(all_tiny)).encode()).hexdigest(),
              'protected_input_sha256':protected_hashes,'cache_manifest':str(AUDIT),
              'all_1096_cache_hashes_equal_before_and_after':True,
              'evaluator_sha256':{m.__file__:sha(m.__file__) for m in
                                  (pycocotools.coco,pycocotools.cocoeval,pycocotools._mask)},
              'wall_seconds':time.monotonic()-start,'model_forwards':0,'training_steps':0,
              'GPU':False,'remote':False,'test':False,'old_files_modified':False}
    write('result.json',output)
    progress({'stage':'complete','primary_A2_pass_both':output['primary_A2_pass_both'],
              'wall_seconds':output['wall_seconds']})


if __name__ == '__main__':
    main()
