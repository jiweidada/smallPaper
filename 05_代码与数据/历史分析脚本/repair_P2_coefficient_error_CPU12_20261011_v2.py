"""Repair path coercion in a new CPU-only version; do not change proposed matrices."""
from pathlib import Path
import hashlib,json,ast,zipfile
R=Path(__file__).resolve().parent.parent
old=R/'英文小论文/code/p2_coefficient_error_CPU12_20261011_v1';new=R/'英文小论文/code/p2_coefficient_error_CPU12_20261011_v2'
out=R/'outputs/p2_coefficient_error_control12_20261011_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
failed=read(out/'remote_CPU_v1_engineering_failure_inputs.json')
for v in failed.values():
    b=v['payload'].encode();assert len(b)==v['transport']['bytes'] and hashlib.sha256(b).hexdigest()==v['transport']['sha256'];j=json.loads(b)
    assert j['exit_code']==1 and j['report'] is None and "AttributeError: 'str' object has no attribute 'read_bytes'" in j['worker_log_tail']
before={p.name:sha(p) for p in old.iterdir()}
s=(old/'check_cpu.py').read_text(encoding='utf8');target="sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()";assert s.count(target)==1
s=s.replace(target,"sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()")
ast.parse(s);new.mkdir(exist_ok=False);(new/'check_cpu.py').write_text(s,encoding='utf8')
spec=read(old/'protocol.json');spec['CPU_check_sha256']=sha(new/'check_cpu.py');spec['schema']='P2_coefficient_error_control_real_CPU_v2'
(new/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
assert sha(new/'check_cpu.py')==hashlib.sha256(Path(new/'check_cpu.py').read_bytes()).hexdigest()
archives=[]
for seed in (1,2):
    assert sha(out/f'seed{seed}_CPU/report.json')==spec['inputs'][str(seed)]['report_sha256']
    assert sha(out/f'seed{seed}_CPU/weights.npz')==spec['inputs'][str(seed)]['weights_sha256']
    archive=out/f'seed{seed}_CPU_deployment_v2.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in new.iterdir():z.write(p,'code/'+p.name)
        for p in (out/f'seed{seed}_CPU').iterdir():z.write(p,f'seed{seed}_input/'+p.name)
    archives.append(dict(seed=seed,archive_sha256=sha(archive),bytes=archive.stat().st_size))
assert before=={p.name:sha(p) for p in old.iterdir()}
review=dict(status='REPAIRED_CPU_ONLY_PATH_COERCION_NEW_VERSION_IDENTICAL_WEIGHT_PROPOSALS',source_sha256=sha(new/'check_cpu.py'),protocol_sha256=sha(new/'protocol.json'),
    archives=archives,v1_failure_preserved=True,proposal_matrices_and_norm_error_constraints_unchanged=True,old_v1_source_unchanged=True,
    failed_training_retried=False,model_replay_tolerance_changed=False,new_training_epochs=0,new_model_forwards=0,producer_sha256=sha(__file__))
(out/'remote_CPU_v2_source_review.json').write_text(json.dumps(review,indent=2),encoding='utf8');print(json.dumps(review))
