"""Exploratory CPU audit of archived BASE12 pair exposures; no model calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'outputs/saved_fgl_coefficient_analysis_20261010_v2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def xyxy(b):
    return np.concatenate((b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2), axis=-1)

def offsets(ref, edges):
    c, s = ref[:2], ref[2:]
    return np.r_[(c-edges[:2])*4/s-2, (edges[2:]-c)*4/s-2]

def iou(a, b):
    wh = np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter = np.prod(wh)
    union = np.prod(np.maximum(0,a[2:]-a[:2]))+np.prod(np.maximum(0,b[2:]-b[:2]))-inter
    return float(inter/union) if union > 0 else 0.

def ceiling(ref, target):
    c, s = ref[:2], ref[2:]
    lo = np.r_[c-1.5*s, c-.5*s]
    hi = np.r_[c+.5*s, c+1.5*s]
    return iou(np.clip(target, lo, hi), target)

def main():
    spec = {
        'status':'exploratory_existing_array_audit',
        'cohort':'all 4049 existing final ordinary FGL pair exposures, BASE12, augmented input shortside<4, image GT<=300',
        'fixed_support':'up=.5, reg_scale=4: each normalized edge offset in [-4,4]',
        'metrics':['native support maximum IoU at .5/.75','GT-direction edge saturation at 90% of endpoint','same-query aux0/aux1/final predicted reference support ceilings'],
        'equal_width_translation':'replace support center by model predicted center, retain original pre width and height',
        'full_reanchor':'replace center/size by archived model prediction, changes both location and span',
        'final_reference_warning':'final frame is a sensitivity analysis, not an available same-stage inference input',
        'limitations':['pair exposures not independent GT','fixed final identity, not historical assignment','augmented fixtures not full validation','ceilings not AP or deployable predictions','no distribution logits/sampling points saved','descriptive thresholds, no new acceptance gate'],
        'script_sha256':sha(Path(__file__)),
        'new_model_forwards':0,'new_optimizer_updates':0,'new_training_epochs':0,
    }
    (OUT/'06_CPU核查预先口径.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
    # Exact checks on the geometry operator, independent of archived observations.
    r=np.array([.5,.5,.1,.1])
    assert ceiling(r,xyxy(r)) == 1.
    assert ceiling(r,np.array([.8,.8,.9,.9])) == 0.
    assert np.max(np.abs(offsets(r,xyxy(r)))) < 1e-12
    z=np.load(PARENT/'reference_stage_trace_v1/final_under4_FGL_pair_reference_transitions.npz',allow_pickle=False)
    assert len(z['seed']) == 4049
    source_manifest=json.loads((PARENT/'report.json').read_text(encoding='utf-8-sig'))['source_files']
    cache={}; source_checks={}; rows=[]
    for k in range(4049):
        seed,batch,slot,q,g=(int(z[key][k]) for key in ['seed','batch','slot','query_row','GT_row'])
        key=(seed,batch)
        if key not in cache:
            p=ROOT/f'outputs/native_augmented_supervision_20261010_v1/collection_seed{seed}/capture_v1/endpoint12/batch_{batch:03d}/arrays.npz'
            rel=str(p.relative_to(ROOT)); actual=sha(p)
            assert actual == source_manifest[rel], str(p)
            source_checks[rel]=actual
            with np.load(p,allow_pickle=False) as a:
                cache[key]={n:a[n].astype(np.float64) for n in a.files if n.endswith('pred_boxes') or (n.startswith('target.') and n.endswith('.boxes'))}
        a=cache[key]; ref=a['model.pre_outputs.pred_boxes'][slot,q]; final=a['model.pred_boxes'][slot,q]
        target=xyxy(a[f'target.{slot}.boxes'][g]); d=offsets(ref,xyxy(final)); required=offsets(ref,target)
        own_inside=bool(np.max(np.abs(required))<=4)
        assert own_inside == bool(z['in_closed_fixed_FDR_support'][k])
        zero=bool(z['recomputed_FP32_IoU'][k]==0)
        assert abs(iou(xyxy(final),target)-float(z['recomputed_FP32_IoU'][k])) < 1e-4
        row={'seed':seed,'batch':batch,'slot':slot,'q':q,'g':g,'zero':zero,'outside':not own_inside,
             'native_ceiling':ceiling(ref,target),'max_abs_actual_offset':float(np.max(np.abs(d))),
             'outside_edge_saturation90':bool(np.any((np.abs(required)>4)&(np.sign(required)*d>=3.6)))}
        for name,prefix in [('aux0','model.aux_outputs.0.pred_boxes'),('aux1','model.aux_outputs.1.pred_boxes'),('final','model.pred_boxes')]:
            newref=a[prefix][slot,q]
            row[name+'_translation_ceiling']=ceiling(np.r_[newref[:2],ref[2:]],target)
            row[name+'_full_reanchor_ceiling']=ceiling(newref,target)
        rows.append(row)
    result={'spec':spec,'source_arrays_sha256':source_checks,'groups':[]}
    for seed in [1,2]:
        for group in ['all','zero','zero_outside']:
            rs=[r for r in rows if r['seed']==seed and (group=='all' or r['zero']) and (group!='zero_outside' or r['outside'])]
            summary={'seed':seed,'group':group,'pair_exposures':len(rs),'outside':sum(r['outside'] for r in rs),'saturation90_toward_outside_GT_edge':sum(r['outside_edge_saturation90'] for r in rs),'max_abs_actual_offset_quantiles':np.quantile([r['max_abs_actual_offset'] for r in rs],[0,.25,.5,.75,.9,1]).tolist()}
            for t in [.5,.75]:
                native=sum(r['native_ceiling']>=t for r in rs)
                summary[f'native_ceiling_ge_{t}']=native
                for name in ['aux0','aux1','final']:
                    for mode in ['translation','full_reanchor']:
                        f=name+'_'+mode+'_ceiling'
                        summary[f'{f}_ge_{t}']=sum(r[f]>=t for r in rs)
                        summary[f'{f}_newly_ge_{t}']=sum(r['native_ceiling']<t<=r[f] for r in rs)
                        summary[f'{f}_lost_ge_{t}']=sum(r[f]<t<=r['native_ceiling'] for r in rs)
            result['groups'].append(summary)
    (OUT/'07_CPU现有配对_支持域核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    import csv
    with (OUT/'07b_CPU逐配对核查.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'sources_verified':len(source_checks),'groups':result['groups']},ensure_ascii=False))

if __name__=='__main__':
    main()
