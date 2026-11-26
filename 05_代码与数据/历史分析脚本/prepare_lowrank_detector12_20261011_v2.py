"""Repair source copying before any model execution; retain v1 artifact."""
from pathlib import Path
R=Path(__file__).resolve().parent.parent
old=R/'outputs/prepare_lowrank_detector12_20261011_v1.py'
s=old.read_text(encoding='utf8')
s=s.replace('p2_lowrank12_20261011_v1','p2_lowrank12_20261011_v2')
s=s.replace("    s=p.read_text(encoding='utf8')","    original_text=p.read_text(encoding='utf8');s=original_text")
s=s.replace("    (new/p.name).write_text(s,encoding='utf8')","    if s==original_text:(new/p.name).write_bytes(p.read_bytes())\n    else:(new/p.name).write_text(s,encoding='utf8')")
needle="assert sha(new/'geometry.py')==sha(old/'geometry.py') and sha(new/'stage_trace.py')==sha(old/'stage_trace.py')"
replacement=needle+"\nassert all(sha(new/n)==h for n,h in json.loads((new/'source_binding.json').read_bytes())['legacy_constructor_files'].items())\nassert sha(new/'original_stem.py')==spec['original_stem_source_sha256']\nfor n in ('geometry.py','residual.py','original_stem.py','original_stem_model.py','original_stem_source.json','original_backbone_source.txt','training_audit.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','user_p2.py','user_attention.py','reuse_baseline.py'):assert sha(new/n)==sha(old/n),n"
assert s.count(needle)==1;s=s.replace(needle,replacement)
exec(compile(s,str(Path(__file__)), 'exec'))
