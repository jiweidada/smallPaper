from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
code=ROOT/'英文小论文/code/native_selection_analysis12_20261011_v4/selection_core.py'
s=importlib.util.spec_from_file_location('bound_grid_checks',code);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
path=ROOT/'outputs/native_selection_anchor_audit12_20261011_v2/seed2/original_stem.npz'
with np.load(path,allow_pickle=False) as z:anchor=z['EMA_anchors'].copy()
cases={'actual_EMA_exact_buffer_preserved':m.verify_grid_centers(dict(dense_anchor_logits=anchor),anchor)}
def reject(name,fn):
    try:fn()
    except AssertionError:cases[name]=True
    else:raise AssertionError(name)
reject('missing_checkpoint_reference',lambda:m.verify_grid_centers(dict(dense_anchor_logits=anchor)))
valid=np.where(np.isfinite(anchor).all(1))[0][0];bad=anchor.copy();bad[valid,0]+=1e-6
reject('tiny_observed_change_rejected_exactly',lambda:m.verify_grid_centers(dict(dense_anchor_logits=bad),anchor))
reject('wrong_float_dtype',lambda:m.verify_grid_centers(dict(dense_anchor_logits=anchor.astype(np.float16)),anchor.astype(np.float16)))
bad=anchor.copy();bad[valid,0]=0
reject('matched_but_wrong_nominal_cell',lambda:m.verify_grid_centers(dict(dense_anchor_logits=bad),bad))
out=ROOT/'outputs/native_selection_analysis12_20261011_v4_CPU/checkpoint_buffer_fixtures.json'
with out.open('x',encoding='utf8') as f:json.dump(dict(status='PASS_CHECKPOINT_BUFFER_EXACTNESS_AND_FEATURE_CELL_REJECTION_CASES',cases=cases,source_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),reference_export_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0),f,indent=2)
print(json.dumps(cases))
