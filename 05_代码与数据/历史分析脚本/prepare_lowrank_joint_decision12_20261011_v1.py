"""Adapt the existing both-seed fixed12 decision without changing its gates."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'英文小论文/code/original_stem12_monitor_20261011_v1/decide_fixed12.py'
new=R/'英文小论文/code/p2_lowrank12_decision_20261011_v1'
assert not new.exists();new.mkdir()
s=old.read_text(encoding='utf8')
s=s.replace("from verify_terminal_local import PASS, PROTOCOL, local_protocol, read, sha, write", "ROOT=Path(__file__).resolve().parents[3]\nsys.path.insert(0,str(ROOT/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1'))\nfrom verify_terminal_local import PASS, PROTOCOL, local_protocol, read, sha, write")
s=s.replace("r['identity']['arm'] == 'original_stem'", "r['identity']['arm'] == 'p2_lowrank16'")
s=s.replace("schema='original_stem_fixed12_two_seed_decision_v1'", "schema='p2_lowrank12_fixed12_two_seed_decision_v1'")
s=s.replace('RETAIN_ORIGINAL_STEM_FOR_MECHANISM_FOLLOWUP','RETAIN_LOW_RANK_P2_FOR_MECHANISM_FOLLOWUP')
s=s.replace('REJECT_ORIGINAL_STEM_NO_EXTENSION_OR_RETRY','REJECT_LOW_RANK_P2_NO_EXTENSION_OR_RETRY')
s=s.replace('bounded original-Stem configuration','bounded rank16 P2 configuration')
s=s.replace("assert len(reports) == 2 and {r['identity']['seed'] for r in reports} == {1, 2}",
    "assert len(reports) == 2 and all(type(r['identity']['seed']) is int for r in reports) and {r['identity']['seed'] for r in reports} == {1, 2}")
s=s.replace("    p = local_protocol()", "    spec=read(Path(__file__).parent/'protocol.json')\n    assert sha(__file__)==spec['decision_sha256'] and sha(PROTOCOL)==spec['local_reader_protocol_sha256']\n    p = local_protocol()")
ast.parse(s);(new/'decide_fixed12.py').write_text(s,encoding='utf8')
p=dict(schema='p2_lowrank12_joint_decision_source_binding_v1',decision_sha256=sha(new/'decide_fixed12.py'),
    source_decision_sha256=sha(old),local_reader_protocol_sha256=sha(R/'英文小论文/code/p2_lowrank12_local_reader_20261011_v1/local_reader_protocol.json'),
    requirement='Both specified seeds at native fixed12 EMA, APs strictly above same-seed BASE12 and AP non-decreasing; all downloaded data and saved-EMA arrays must be verified first.',
    new_actual_final_reports=0,new_model_forwards=0,new_training_epochs=0)
(new/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf8')
print(json.dumps(dict(status='PREPARED_BOTH_SEED_FIXED12_DECISION_GATE_UNCHANGED',decision_sha256=p['decision_sha256'],new_actual_final_reports=0)))
