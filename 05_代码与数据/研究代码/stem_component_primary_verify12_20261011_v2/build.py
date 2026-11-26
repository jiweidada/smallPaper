"""Keep primary verifier checks intact; only make the original hardcoded arm explicit."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

root=Path.cwd();here=root/'英文小论文/code/stem_component_primary_verify12_20261011_v2'
old=root/'英文小论文/code/stem_component_primary_analysis12_20261011_v1/verify_primary.py'
text=old.read_text(encoding='utf8')
changes=[("def verify(seed):\n    arm='highfreq_rgb';folder=", "def verify(seed, arm):\n    assert arm in ('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')\n    folder="),
 ("p.add_argument('--seed',type=int,choices=(1,2),required=True);a=p.parse_args();verify(a.seed)",
  "p.add_argument('--seed',type=int,choices=(1,2),required=True);p.add_argument('--arm',choices=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample'),required=True);a=p.parse_args();verify(a.seed,a.arm)")]
original=text
for a,b in changes:
    assert text.count(a)==1;text=text.replace(a,b)
ast.parse(text)
with (here/'verify_primary.py').open('x',encoding='utf8') as f:f.write(text)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(source_original_sha256=sha(old),source_new_sha256=sha(here/'verify_primary.py'),
    changes=[dict(old=a,new=b) for a,b in changes],all_verification_checks_unchanged=True,
    schema='component_primary_verifier_explicit_arm_v2',saved_EMA_replay_claim_allowed=False,
    scope='Actual native primary archive only; cached baseline validation and all548 stage arrays required. No Torch/model/evaluator imports or calls.')
with (here/'adaptation.json').open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
with (here/'adaptation.diff').open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(here/'verify_primary.py'))))
print(json.dumps(manifest))
