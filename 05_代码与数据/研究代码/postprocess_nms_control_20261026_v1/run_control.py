"""Predeclared traditional NMS control, using frozen raw predictions only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
import contextlib
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path
import torch
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent/'postprocess_budget_precheck_20261026_v1'
spec = importlib.util.spec_from_file_location('budget_precheck',PREV/'run_precheck.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
torch.set_num_threads(2)


def select(prob, coords, legal=None):
    kept = []
    for c in range(10):
        use = prob[:,c] > .001
        if legal is not None:
            use = use & legal
        q = use.nonzero().flatten()
        k = nms(coords[q],prob[q,c],.5)
        kept.extend((q[k]*10+c).tolist())
    pool = torch.tensor(kept,dtype=torch.long)
    vals,ix = prob.flatten()[pool].topk(min(300,len(pool)))
    flat = pool[ix]
    counts = [0]*10
    out = []
    for i in flat.tolist():
        if counts[i%10] < 100:
            out.append(i)
            counts[i%10] += 1
    return torch.tensor(out,dtype=torch.long)


def main():
    started = time.monotonic()
    reference = json.loads((PREV/'result.json').read_text(encoding='utf-8'))
    audit = json.loads(P.AUDIT.read_text(encoding='utf-8'))
    manifest = {x['path']:x['sha256'] for x in audit['input_manifest']}
    ref344 = json.loads(P.REF.read_text(encoding='utf-8'))
    ann = Path(ref344['per_seed'][0]['annotation_path'])
    dataset = json.loads(ann.read_text(encoding='utf-8'));dataset.setdefault('info',{})
    protected = {str(p):P.sha(p) for p in (ann,PREV/'result.json',PREV/'run_precheck.py',ROOT/'protocol.txt',Path(__file__))}
    result = {}
    for seed in (1,2):
        rows = {'B0':[],'B1':[],'B2':[]}
        invalid = {'B1':[],'B2':[]}
        tiny_ids = set()
        for path in sorted((P.CACHE/f'342_cache_val_v2_seed{seed}').glob('342_*.pt')):
            blob = path.read_bytes()
            assert hashlib.sha256(blob).hexdigest() == manifest[str(path)]
            d = torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
            raw = d['final300_boxes'];prob=d['final300_logits'].sigmoid()
            coords = P.to_edges(raw,torch.as_tensor(d['orig_wh'],dtype=raw.dtype).repeat(2))
            legal = torch.isfinite(coords).all(1)&(coords[:,2:]>coords[:,:2]).all(1)
            flat,pre = P.choose(prob,legal,'A0')
            pp = prob.clone();pp[~legal] = -torch.inf
            v,k = pp.flatten().topk(300)
            assert torch.equal(k[v>.001],pre)
            rows['B0'].extend(P.make_rows(d['image_id'],coords,prob,flat))
            tiny_ids.update(int(g) for g,t in zip(d['gt_annotation_ids'],d['gt_short_side_640_all']<4) if t)
            for arm,mask in (('B1',None),('B2',legal)):
                f = select(prob,coords,mask)
                bad = (f//10)[~legal[f//10]].unique().tolist()
                if bad:
                    invalid[arm].append({'image_id':int(d['image_id']),'query_ids':bad})
                else:
                    rows[arm].extend(P.make_rows(d['image_id'],coords,prob,f))
        assert len(tiny_ids) == 4952
        baseline,base_gt,log = P.evaluate(dataset,rows['B0'],tiny_ids)
        for k in P.NAMES:
            assert baseline['stats12'][k] == reference['per_seed'][str(seed)]['arms']['A0']['stats12'][k]
        for k in ('TP50','FP50','tiny_TP50','detections'):
            assert baseline[k] == reference['per_seed'][str(seed)]['arms']['A0'][k]
        P.progress({'stage':'baseline_reproduced','seed':seed})
        record = {'B0':baseline}
        for arm in ('B1','B2'):
            if invalid[arm]:
                record[arm] = {'valid':False,'stats12':None,'invalid_output':invalid[arm]}
                P.progress({'stage':'invalid','seed':seed,'arm':arm,'invalid':invalid[arm]})
                continue
            if arm == 'B2' and not invalid['B1'] and rows['B2'] == rows['B1']:
                record[arm] = dict(record['B1'],equal_to_B1=True,evaluation_reused=True)
                P.progress({'stage':'identical_reuse','seed':seed,'arm':arm})
                continue
            metrics,matched,log = P.evaluate(dataset,rows[arm],tiny_ids)
            with (ROOT/f'coco_seed{seed}_{arm}.txt').open('x',encoding='utf-8') as f:
                f.write(log)
            metrics.update(valid=True,
                delta_stats12={k:metrics['stats12'][k]-baseline['stats12'][k] for k in P.NAMES},
                delta_TP50=metrics['TP50']-baseline['TP50'],delta_FP50=metrics['FP50']-baseline['FP50'],
                tiny_TP_gained=len((matched-base_gt)&tiny_ids),tiny_TP_lost=len((base_gt-matched)&tiny_ids),
                all_TP_gained=len(matched-base_gt),all_TP_lost=len(base_gt-matched))
            record[arm] = metrics
            P.progress({'stage':'evaluated','seed':seed,'arm':arm,**metrics})
        result[str(seed)] = record
    for path,h in protected.items():
        assert P.sha(path) == h
    for path,h in manifest.items():
        assert P.sha(path) == h
    out = {'per_seed':result,'protected_input_sha256':protected,
           'all_1096_cache_hashes_equal_before_and_after':True,
           'model_forwards':0,'training_steps':0,'wall_seconds':time.monotonic()-started,
           'scope':'Known-method control, not a novel method; no training authorization inferred.'}
    with (ROOT/'result.json').open('x',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    P.progress({'stage':'complete','wall_seconds':out['wall_seconds']})


if __name__ == '__main__':
    main()
