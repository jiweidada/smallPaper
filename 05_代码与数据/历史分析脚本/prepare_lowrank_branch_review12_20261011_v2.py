"""Preserve the v1 CPU rank-check failure; fix factor-product precision only."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent.parent
old=R/'英文小论文/code/p2_lowrank_branch_review12_20261011_v1'
new=R/'英文小论文/code/p2_lowrank_branch_review12_20261011_v2';new.mkdir(exist_ok=False)
(new/'branch.py').write_bytes((old/'branch.py').read_bytes())
s=(old/'validate_cpu.py').read_text()
before='assert torch.linalg.matrix_rank(branch.effective_weight().double()).item()<=16'
after="factor_product64=branch.restore.weight.view(256,16).double()@branch.mix.weight.view(16,64).double()\nassert torch.linalg.matrix_rank(factor_product64).item()<=16\nassert torch.linalg.matrix_rank(branch.effective_weight()).item()<=16"
assert s.count(before)==1;s=s.replace(before,after)
s=s.replace("rank_upper_bound=16,equivalent_rank_limited_stride2_conv_verified=True", "rank_upper_bound=16,rank_factor_product64=int(torch.linalg.matrix_rank(factor_product64)),rank_rounded_float32_product=int(torch.linalg.matrix_rank(branch.effective_weight())),float32_product_roundoff_max=float((branch.effective_weight().double()-factor_product64).abs().max()),equivalent_rank_limited_stride2_conv_verified=True")
ast.parse(s);(new/'validate_cpu.py').write_text(s,encoding='utf8')
spec=json.loads((old/'protocol.json').read_bytes());spec['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in new.glob('*.py')}
spec['preserved_v1_checker_sha256']=hashlib.sha256((old/'validate_cpu.py').read_bytes()).hexdigest()
spec['repair_scope']='V1 multiplied factors in float32 then changed to float64, whose rank threshold interpreted float32 roundoff as signal. V2 checks product from float64 factors, and separately float32 product with its own dtype threshold. Branch is byte-identical; no arbitrary rank tolerance or training change.'
(new/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
print(json.dumps(spec))
