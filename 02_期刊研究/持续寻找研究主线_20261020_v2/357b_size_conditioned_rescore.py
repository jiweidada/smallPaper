"""357b 只读：尺寸条件的分数重标定是否改变 AP/APs（val548，BASE24 raw 双 seed）。
做法：对每个预测框，只用它自身的预测尺寸 s=min(w,h)（640 坐标）分档，score' = score * g(档)。
g 在 seed1 上用"把各档 TP 的分数中位数拉到 [16,32) 档的水平"这一规则一次算出，原样用到 seed2。
另报 oracle 上界：g 在 seed2 自己上算。无模型前向、无训练，只改后处理排序。输出 357b_size_conditioned_rescore.json"""
import json, io, contextlib, copy
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ROOT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文')
PRED = ROOT/'code/unit_instance_completion_20261021_v1/343_codex_stage4_3b_20261022_v1'
ANN = r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations\val.json'
OUT = ROOT/'持续寻找研究主线_20261020_v2/357b_size_conditioned_rescore.json'
NAMES = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(ANN); gt.dataset.setdefault('info', {})
imgs = {i['id']: i for i in gt.dataset['images']}
EDGES = [0, 4, 8, 16, 32, 1e9]
def bin_of(r):
    im = imgs[r['image_id']]; w = r['bbox'][2]*640/im['width']; h = r['bbox'][3]*640/im['height']
    s = min(w, h)
    return next(k for k in range(5) if EDGES[k] <= s < EDGES[k+1])
def evaluate(rows):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(rows); ev = COCOeval(gt, dt, 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    tp = {}
    for e in ev.evalImgs:
        if e and e['aRng'] == [0, 1e10] and e['maxDet'] == 100:
            for did, m in zip(e['dtIds'], e['dtMatches'][0]): tp[did] = m > 0
    return {k: float(v) for k, v in zip(NAMES, ev.stats)}, ev, tp
def fit_gain(rows):
    st, ev, tp = evaluate(rows)
    anns = ev.cocoDt.anns
    med = {}
    for k in range(5):
        s = [a['score'] for did, a in anns.items() if tp.get(did) and bin_of(a) == k]
        med[k] = float(np.median(s)) if len(s) >= 50 else None
    ref = med[3]
    g = {k: (ref/med[k] if med[k] else 1.0) for k in range(5)}
    return g, med, st
def apply(rows, g):
    out = copy.deepcopy(rows)
    for r in out:
        r['score'] = min(1.0, r['score']*g[bin_of(r)])
    return out
R = {}
rows = {s: json.load(open(PRED/f'343_val_seed{s}_baseline_predictions.json', encoding='utf-8')) for s in (1, 2)}
g1, med1, base1 = fit_gain(rows[1])
g2, med2, base2 = fit_gain(rows[2])
R['gain_fit_on_seed1'] = {'tp_score_median_by_predbin': med1, 'g': g1}
R['gain_fit_on_seed2'] = {'tp_score_median_by_predbin': med2, 'g': g2}
for s, base, g_cross, g_self in ((1, base1, g2, g1), (2, base2, g1, g2)):
    cross, _, _ = evaluate(apply(rows[s], g_cross))
    self_, _, _ = evaluate(apply(rows[s], g_self))
    R[f'seed{s}'] = {'baseline': base, 'cross_seed_gain': cross, 'self_gain_oracle': self_,
                     'delta_cross_pp': {k: round(100*(cross[k]-base[k]), 4) for k in NAMES},
                     'delta_self_pp': {k: round(100*(self_[k]-base[k]), 4) for k in NAMES}}
R['_note'] = 'pred bin edges on 640 stretch short side: [0,4),[4,8),[8,16),[16,32),[32,inf); g 让各档 TP 分数中位数等于 [16,32) 档；cross=用另一 seed 拟合的 g'
json.dump(R, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
for s in (1, 2):
    print(s, 'cross', R[f'seed{s}']['delta_cross_pp'], '\n  self', R[f'seed{s}']['delta_self_pp'])
print('g1', g1, '\ng2', g2)
