"""Finite sanity check of the evaluator's threshold-matching set property.

No detector, evaluator class, private data or training is invoked.
The general proof is separate; finite enumeration does not establish it alone.
"""
import itertools,json,time,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
LEVELS=(.2,.5,.75,.9)
def greedy(M,t):
    used=set();prefix=[]
    for row in M:
        best=t;owner=None
        for g,x in enumerate(row):
            if g in used or x<best:continue
            best=x;owner=g
        if owner is not None:used.add(owner)
        prefix.append(frozenset(used))
    return prefix
def main():
    start=time.perf_counter();counter=None;n=0;prefix_checks=0
    for flat in itertools.product(LEVELS,repeat=9):
        M=[flat[i:i+3] for i in (0,3,6)]
        low=greedy(M,.5);high=greedy(M,.75);n+=1
        for k,(a,b) in enumerate(zip(low,high)):
            prefix_checks+=1
            if not b.issubset(a):
                counter={'matrix':M,'detection_prefix':k+1,'low':sorted(a),'high':sorted(b)};break
        if counter is not None:break
    result={'matrix_shape':[3,3],'levels':LEVELS,'thresholds':[.5,.75],'enumerated_matrices':n,'prefix_comparisons':prefix_checks,
        'counterexample':counter,'conditions':'fixed detection order, exclusive GT, all GT regular, fixed max-IoU/tie preferences',
        'meaning':'finite consistency check only; induction proof required for all sizes',
        'elapsed_seconds':time.perf_counter()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with open(OUT/'monotonicity_finite_check.json','x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
