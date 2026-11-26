from pathlib import Path
import json
import numpy as np
from semantic_math import auc
D=Path(__file__).resolve().parent
groups=np.array(['A','A','A','A','B','B'])
labels=np.array([1,0,1,0,1,0]);scores=np.array([.9,.1,.8,.2,0.,.7])
order=['A','B'];draw=[0,0,1]
indices={g:np.flatnonzero(groups==g) for g in order}
sample=np.concatenate([indices[order[i]] for i in draw])
np.testing.assert_array_equal(sample,[0,1,2,3,0,1,2,3,4,5])
assert abs(float(auc(labels[sample],scores[sample]))-.8)<1e-15
assert json.loads(json.dumps(dict(gate=np.bool_(True),n=np.int64(2)),default=lambda x:x.item()))=={'gate':True,'n':2}
result=dict(status='PASS_GROUP_LITERAL_DUPLICATION_HAND_AUC_AND_SCALAR_JSON',
    bootstrap_duplicates_entire_group=True,expected_auc=.8,new_author_model_calls=0,new_data_probe_fits=0)
with (D/'group_bootstrap_fixture_result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
