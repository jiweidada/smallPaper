"""Run the actual deployment scorer on the author's full native FRCNN reference."""
import collections
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat
from uavdt_protocol import NativeScorer, voc_ap

def run(data):
    data=Path(data);a=json.loads((data/'annotations/test.json').read_text());s=NativeScorer(a)
    by_sequence=collections.defaultdict(dict)
    for image in a['images']:by_sequence[image['uav_sequence']][image['uav_frame']]=image['id']
    for sequence,frames in sorted(by_sequence.items()):
        rows=np.loadtxt(data/'protocol_reference'/(sequence+'.txt'),delimiter=',',ndmin=2)
        grouped=collections.defaultdict(list)
        for row in rows:grouped[int(row[0])].append(row[2:7])
        for frame,image_id in sorted(frames.items()):s.update(image_id,grouped[frame])
    named,recall,precision=s.finish()
    official=loadmat(data/'protocol_reference/det_FRCNN_overall.mat',simplify_cells=True)['attribute']
    rr=np.asarray(official['rec']).reshape(-1);pp=np.asarray(official['prec']).reshape(-1)
    assert len(rr)==len(recall)==417036 and s.count==375884
    rec_diff=float(np.max(np.abs(rr-recall)));prec_diff=float(np.max(np.abs(pp-precision)))
    assert rec_diff<1e-12 and prec_diff<1e-12
    return {'author_reference_full_curve_passed':True,'detections':len(recall),'GT':s.count,'AP70_reference_only':named['AP70'],
            'recall_max_absolute_difference':rec_diff,'precision_max_absolute_difference':prec_diff}

if __name__=='__main__':print(json.dumps(run(sys.argv[1])),flush=True)
