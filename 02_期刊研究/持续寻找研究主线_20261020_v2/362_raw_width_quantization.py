"""362 只读：VisDrone val 原始 txt 标注中目标宽度的整数分布，以及 tiny（640 拉伸短边<4）对应的原图宽度。
目的：IoU 质量目标在 tiny 上的离散度有多少来自"原图标注只能取整数像素"。无模型。"""
import json, collections, glob, os
from pathlib import Path
ANN = Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val\annotations')
COCO = ANN/'val.json'
OUT = Path(r'C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\362_raw_width_quantization.json')
c = json.load(open(COCO, encoding='utf-8'))
ims = {i['id']: i for i in c['images']}
wc = collections.Counter(); hc = collections.Counter(); nonint = 0; tot = 0
tiny_w = collections.Counter()
for a in c['annotations']:
    im = ims[a['image_id']]; x, y, w, h = a['bbox']; tot += 1
    if abs(w-round(w)) > 1e-3 or abs(h-round(h)) > 1e-3: nonint += 1
    wc[round(w)] += 1; hc[round(h)] += 1
    w640 = w*640/im['width']; h640 = h*640/im['height']
    if min(w640, h640) < 4: tiny_w[round(w)] += 1
raw_score = collections.Counter()
for p in glob.glob(str(ANN/'*.txt')):
    for line in open(p):
        t = line.strip().split(',')
        if len(t) >= 6: raw_score[(t[4], t[5])] += 1
res = {'coco_annotations': tot, 'non_integer_wh': nonint,
       'orig_width_counts_le12': {k: wc[k] for k in range(0, 13)},
       'orig_height_counts_le12': {k: hc[k] for k in range(0, 13)},
       'tiny640_orig_width_counts': dict(sorted(tiny_w.items())),
       'raw_txt_(score,category)_top': raw_score.most_common(15)}
json.dump(res, open(OUT, 'x', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
