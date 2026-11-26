"""Choose comparable geometry pairs from known A15 outcomes before new feature-response metrics."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from a15_sampling_analysis import paired_iou

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'A16_实际特征空间变化审计_v1'
BASE=ROOT/'A15_固定query的采样中心干预_v1'
assert not (RUN/'geometry_design.json').exists()
rows=[];pairs=[];inputs=[];counts=[]
for case in range(12):
    paths=[BASE/'captures'/('case%02d.npz'%case),BASE/'predictions'/('case%02d.npz'%case)]
    inputs.extend({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths)
    with np.load(paths[0],allow_pickle=False) as c,np.load(paths[1],allow_pickle=False) as z:
        mq=z['match_q'];mg=z['match_gt'];gt=z['gt_boxes800'];sizes=z['gt_sizes800']
        final=paired_iou(z['raw_boxes'][mq],gt[mg])
        prior=paired_iou(c['reference'][mq,0],gt[mg])
        priorwh=c['reference'][mq,0,2:]*800
        eligible=[]
        for q,g,s,f,b,wh in zip(mq,mg,sizes[mg],final,prior,priorwh):
            if s>=8:continue
            r={'case':case,'gt':int(g),'query':int(q),'label':int(z['gt_labels'][g]),'size':float(s),
               'native_iou':float(f),'input_reference_iou':float(b),'input_reference_positive':bool(np.all(wh>0)),
               'input_reference_size':float(np.sqrt(np.maximum(wh.prod(),0))),
               'input_reference_iou_bin':min(3,int(b*4)),'good':bool(f>=.75)}
            rows.append(r)
            if r['input_reference_positive']:eligible.append(r)
        local=[]
        for good in [r for r in eligible if r['good']]:
            for bad in [r for r in eligible if not r['good']]:
                if good['label']!=bad['label'] or good['input_reference_iou_bin']!=bad['input_reference_iou_bin']:continue
                if max(good['size'],bad['size'])/min(good['size'],bad['size'])>1.5:continue
                if max(good['input_reference_size'],bad['input_reference_size'])/min(good['input_reference_size'],bad['input_reference_size'])>1.5:continue
                local.append({'case':case,'good_gt':good['gt'],'bad_gt':bad['gt'],
                    'good_query':good['query'],'bad_query':bad['query']})
        pairs.extend(local)
        counts.append({'case':case,'tiny':int(sum(sizes<8)),'pairs':len(local),
            'distinct_good':len({p['good_gt'] for p in local}),'distinct_bad':len({p['bad_gt'] for p in local})})
def write(name,obj):(RUN/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
write('geometry_rows.json',rows);write('geometry_pairs.json',pairs)
summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'KNOWN_GEOMETRY_PAIR_DESIGN_BEFORE_FEATURE_RESPONSE_METRICS',
    'design_scope':'A15geometryalreadyknown;thisisnotindependentdata. Noattention-gradient/retentionvaluehasbeencalculated.',
    'rule':'Sameimage/class/input-referenceIoUquarterbin;GTsqrtsizeratio<=1.5andinput-referenceboxsqrtsizeratio<=1.5;positiveinputreferencewh. GoodfinalIoU>=.75,bad<.75.',
    'cases':counts,'total_tiny_GT':len(rows),'total_pairs':len(pairs),
    'images_with_pairs':sum(c['pairs']>0 for c in counts),
    'distinct_good':len({(p['case'],p['good_gt']) for p in pairs}),
    'distinct_bad':len({(p['case'],p['bad_gt']) for p in pairs}),
    'inputs':inputs,'new_model_forwards':0,'new_fits':0}
write('geometry_design.json',summary)
print(json.dumps({k:summary[k] for k in ['status','total_tiny_GT','total_pairs','images_with_pairs','distinct_good','distinct_bad']}))
