"""Compare stored S1 prediction records by image and within-image rank; no inference."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import gc
import json
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
HIST=Path(r'C:\Users\jiwei\Desktop\DEIM\outputs\deim_hgnetv2_s_MSCA_EnhancedStem_SiLU_320_eval\pred.json')
CPU=PAPER/'本地CPU核验_20261103/results/S1/predictions.json'
OUTPUT=HERE/'s1_saved_prediction_rank_comparison.json'

def sha(path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    with path.open(encoding='utf-8-sig') as f:
        records=json.load(f)
    assert isinstance(records,list)
    n=len(records)
    image=np.empty(n,dtype=np.int32)
    category=np.empty(n,dtype=np.int16)
    bbox=np.empty((n,4),dtype=np.float64)
    score=np.empty(n,dtype=np.float64)
    for i,row in enumerate(records):
        assert len(row['bbox'])==4
        image[i]=row['image_id']
        category[i]=row['category_id']
        bbox[i]=row['bbox']
        score[i]=row['score']
    del records
    gc.collect()
    assert np.isfinite(bbox).all() and np.isfinite(score).all()
    return image,category,bbox,score

def percentiles(a):
    return {str(q):float(np.percentile(a,q)) for q in [0,50,90,99,99.9,100]}

def main():
    assert not OUTPUT.exists()
    h=load(HIST)
    c=load(CPU)
    hist_counts=Counter(h[0].tolist())
    cpu_counts=Counter(c[0].tolist())
    result={'status':'EXISTING_PREDICTIONS_COMPARED_BY_SAVED_ORDER',
            'historical':{'path':str(HIST),'sha256':sha(HIST),'records':len(h[0]),
                          'images':len(hist_counts),'per_image_counts':dict(Counter(hist_counts.values()))},
            'cpu':{'path':str(CPU),'sha256':sha(CPU),'records':len(c[0]),
                   'images':len(cpu_counts),'per_image_counts':dict(Counter(cpu_counts.values()))},
            'same_image_count_map':hist_counts==cpu_counts,
            'models_run':0,'forwards':0,'evaluator_runs':0,'optimizer_updates':0,
            'limits':['Comparison is by saved within-image order only; swaps and ties need not represent corresponding detections.',
                      'Similarity of predictions does not prove identical checkpoint, code, historical image bytes, or evaluation protocol.',
                      'The historical pred.json and eval.pth have no retained cryptographic binding to one another or a checkpoint.',
                      'No matching optimization, new prediction, AP calculation, or inference was performed.']}
    if len(h[0])==len(c[0]) and np.array_equal(h[0],c[0]):
        same_category=h[1]==c[1]
        score_diff=np.abs(h[3]-c[3])
        bbox_diff=np.max(np.abs(h[2]-c[2]),axis=1)
        result['rank_aligned']={'records':len(h[0]),'same_category_count':int(np.count_nonzero(same_category)),
            'different_category_count':int(np.count_nonzero(~same_category)),
            'score_abs_difference_percentiles_all':percentiles(score_diff),
            'max_bbox_coordinate_abs_difference_percentiles_all':percentiles(bbox_diff),
            'score_abs_difference_percentiles_same_category':percentiles(score_diff[same_category]),
            'max_bbox_coordinate_abs_difference_percentiles_same_category':percentiles(bbox_diff[same_category])}
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
