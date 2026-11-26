"""Fixed-assignment geometry analysis for the prespecified last-layer intervention."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np

ARMS=['REPLAY_NATIVE','SAMPLE_GT_CENTER','POSITION_GT_CENTER','BOTH_GT_CENTER',
      'NATIVE_PERMUTED_VALUE','SAMPLE_GT_CENTER_PERMUTED_VALUE']

def write(p,v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def xyxy(box):
    return np.concatenate([box[...,:2]-box[...,2:]/2,box[...,:2]+box[...,2:]/2],axis=-1)

def paired_iou(box,gt):
    b=xyxy(np.asarray(box,dtype=np.float64))*800
    g=np.asarray(gt,dtype=np.float64)
    wh=np.maximum(0,np.minimum(b[...,2:],g[...,2:])-np.maximum(b[...,:2],g[...,:2]))
    inter=wh[...,0]*wh[...,1]
    bw=np.maximum(0,b[...,2:]-b[...,:2]);gw=np.maximum(0,g[...,2:]-g[...,:2])
    return inter/np.maximum(bw[...,0]*bw[...,1]+gw[...,0]*gw[...,1]-inter,1e-30)

def boot(values,seed):
    a=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    vals=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'low':float(np.quantile(vals,.025)),
            'high':float(np.quantile(vals,.975)),'images':len(a)}

def analyze(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    images=[];gtrows=[]
    for index in range(12):
        with np.load(root/'predictions'/('case%02d.npz'%index),allow_pickle=False) as z:
            mq=z['match_q'];mg=z['match_gt'];g=z['gt_boxes800'];sz=z['gt_sizes800']
            assert len(np.unique(mq))==len(mq) and len(np.unique(mg))==len(mg)
            tiny=sz<8
            scores=np.zeros((len(ARMS),len(g)),dtype=np.float64)
            for a in range(len(ARMS)):scores[a,mg]=paired_iou(z['arm_boxes'][a,mq],g[mg])
            tiny_scores=scores[:,tiny]
            r75=(tiny_scores>=.75).mean(axis=1)
            miou=tiny_scores.mean(axis=1)
            effects={'sample75':float(r75[1]-r75[0]),
                'content_interaction75':float((r75[1]-r75[0])-(r75[5]-r75[4])),
                'position75':float(r75[2]-r75[0]),'both75':float(r75[3]-r75[0]),
                'sample_mean_iou':float(miou[1]-miou[0])}
            images.append({'case':index,'prefix':str(z['prefix'].item()),'tiny_GT':int(tiny.sum()),
                'tiny_iou75_counts':(tiny_scores>=.75).sum(axis=1).tolist(),
                'tiny_mean_iou':miou.tolist(),'effects':effects})
            for j in np.nonzero(tiny)[0]:
                gtrows.append({'case':index,'gt':int(j),'ious':scores[:,j].tolist()})
    assert sum(r['tiny_GT'] for r in images)==263
    assert len({r['prefix'] for r in images})==12
    stats={key:boot([r['effects'][key] for r in images],p['bootstrap_seed'])
           for key in images[0]['effects']}
    counts=np.sum([r['tiny_iou75_counts'] for r in images],axis=0).tolist()
    assert counts[0]==40,'Fixed A12 native assignment no longer reproduces A13 baseline.'
    passed=(stats['sample75']['mean']>=.05 and stats['sample75']['low']>0 and
            stats['content_interaction75']['mean']>=.03 and stats['content_interaction75']['low']>0)
    decision=('SUPPORTS_BOUNDED_LAST_READ_LOCATION_SUFFICIENCY_NOT_NEW_ALGORITHM' if passed else
              'NO_SUFFICIENT_LAST_LAYER_SAMPLE_CENTER_SIGNAL_NO_METRIC_REFINER_TRAINING')
    summary={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETED_PRESPECIFIED_SAMPLING_INTERVENTION_ANALYSIS',
        'images':12,'prefixes':12,'tiny_GT':263,'arms':ARMS,'tiny_iou75_counts':counts,
        'image_primary_effects':stats,'decision':decision,'gate_passed':bool(passed),
        'new_fits':0,'main_model_updates':0,'mainline_validated':False,
        'scope':['GT-centre diagnostic,notdeployableinferenceorAP.',
            'Last-layeronlywithfixedincomingqueries;permutedvaluedoesnotremoveearlierreal-imageinformation.',
            'Same12previouslyobservedA12official-valcases;notblindtestorindependentgeneralization.',
            'Failurecannotproveencoderinformationloss;successcannotproveautomaticlocalizationornovelty.']}
    write(root/'per_image_analysis.json',images);write(root/'per_GT_analysis.json',gtrows)
    write(root/'analysis_summary.json',summary)
    return summary

if __name__=='__main__':
    import sys
    print(json.dumps(analyze(Path(sys.argv[1])),ensure_ascii=False))
