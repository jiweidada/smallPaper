"""Make an isolated author-head copy with standard pooling and per-query time."""
from pathlib import Path
import hashlib
import json

P=Path(__file__).resolve().parent
src=P/'DiffusionDet_primary_source_v1/diffusiondet/head.py'
raw=src.read_text('utf-8')
changes=[
('from detectron2.modeling.poolers import ROIPooler\nfrom detectron2.structures import Boxes',
 'from torch_detector_parts_v1 import ROIPooler, Boxes'),
('        time = self.time_mlp(t)',
 '        time = (self.time_mlp(t.reshape(-1)).reshape(*t.shape, -1)\n                if t.ndim == 2 else self.time_mlp(t))'),
('        scale_shift = self.block_time_mlp(time_emb)\n        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)',
 '        if time_emb.ndim == 3:\n            scale_shift = self.block_time_mlp(time_emb.reshape(-1, time_emb.shape[-1]))\n        else:\n            scale_shift = self.block_time_mlp(time_emb)\n            scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)')]
for old,new in changes:
    assert raw.count(old)==1,old
    raw=raw.replace(old,new)
target=P/'head_query_time_A60_v1.py'
with target.open('x',encoding='utf-8') as f:f.write(raw)
with (P/'head_source_derivation_v1.json').open('x',encoding='utf-8') as f:json.dump(dict(
    author_repo='ShoufaChen/DiffusionDet',pin='1efb36d4b0b721f2d509bc72656235a94d73ab02',
    author_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    replacements=[dict(before=a,after=b) for a,b in changes],
    unchanged_parameter_names_shapes=True,numerical_equivalence_verified=False,
    limits='Standard R50/FPN/ROIAlign and selected author DynamicHead; controlled local implementation, not complete FlowDet or CEOSOD reproduction.'),f,indent=2)
print('Prepared isolated head source; original author files preserved; no model constructed.')
