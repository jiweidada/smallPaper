"""New launch protocol; model and fixed12 efficacy criterion unchanged, no old failure erased."""
from pathlib import Path
import ast,hashlib,json,zipfile
R=Path(__file__).resolve().parent.parent
old=R/'英文小论文/code/p2_lowrank12_20261011_v2';new=R/'英文小论文/code/p2_lowrank12_20261011_v3'
out=R/'outputs/p2_lowrank12_20261011_v3';new.mkdir(exist_ok=False);out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
before={p.name:sha(p) for p in old.iterdir()}
for p in old.iterdir():(new/p.name).write_bytes(p.read_bytes())
p=new/'prepare_training.py';s=p.read_text(encoding='utf8');assert s.count('p2_lowrank12_20261011_v2')==1;p.write_text(s.replace('p2_lowrank12_20261011_v2','p2_lowrank12_20261011_v3'),encoding='utf8')
p=new/'dispatch.py';s=p.read_text(encoding='utf8');start=s.index('def reviewed_two_seed_gate():');end=s.index('\n\ndef authorize_training',start)
gate='''def reviewed_two_seed_gate():
    gate=read(HERE.parent/'training_decision.json')
    assert gate['schema']=='p2_lowrank12_joint_engineering_release_v2' and gate['training_protocol_sha256']==sha(HERE/'training_protocol.json')
    assert gate['reviewed_model_source_sha256']==sha(HERE/'user_model.py') and gate['reviewed_branch_source_sha256']==sha(HERE/'lowrank_branch.py')
    assert gate['status']=='RELEASE_DISTINCT_FIXED12_AFTER_BOTH_HOST_ENGINEERING_REVIEW' and gate['seeds']==[1,2]
    assert gate['BASE_training_epochs_added']==gate['BASE_model_forwards_added']==0
    assert not gate['training_efficacy_proven'] and not gate['novelty_proven']
    assert gate['A_original_P2_exact_replay_status']=='CANDIDATE_RELOAD_MISMATCH_NO_RETRY'
    assert not gate['two_seed_P2_runtime_diagnostic_complete']
    for seed in ('1','2'):
        r=gate['engineering'][seed]
        assert r['CPU_status']=='PASS_STRICT_SHARED_INITIALIZATION_AND_ISOLATED_NEW_RNG'
        assert r['shared_native_states_exact']==794 and r['new_states']==8 and r['parameters']==10238173
        assert r['dataset_images_verified']==7019
        assert r['AMP_status']=='PASS_LOW_RANK_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'
        assert r['candidate_full_detector_forwards']==2 and r['optimizer_updates']==r['training_epochs']==r['BASE_forwards']==0
        assert r['implementation']==implementation_hashes()
        assert len(r['CPU_report_sha256'])==len(r['AMP_report_sha256'])==64
    return gate
'''
s=s[:start]+gate+s[end:];s=s.replace('from runtime import HERE,read,write,sha,verify_bundle','from runtime import HERE,read,write,sha,verify_bundle,implementation_hashes')
s=s.replace('waiting_positive_joint_runtime_review_and_actual_GPU_idle','waiting_both_host_engineering_review_and_actual_GPU_idle');p.write_text(s,encoding='utf8')
p=new/'training_protocol.json';spec=json.loads(p.read_bytes())
spec['joint_release']='Both new-version common794/new8 CPU initialization,7019 data hashes and full AMP loss/backward preflight must pass before either training. Prior saved-P2 runtime success is diagnostic evidence, not a training prerequisite. Its A exact replay failure remains negative; no tolerance changed or old namespace retried.'
spec['launch_revision']=dict(reason='A old P2 exact replay differs in both order and numerical values despite strict801-state load. New rank16 training starts from common pretrained initialization and uses existing BASE12, so it is an independent falsifiable test, not a weight-replay comparison.',previous_training_started=False,previous_runtime_gate_satisfied=False,two_seed_runtime_diagnostic_complete=False,original_failure_preserved=True,model_and_initialization_unchanged=True,efficacy_criterion_unchanged=True)
p.write_text(json.dumps(spec,indent=2),encoding='utf8')
p=new/'closeout_protocol.json';spec=json.loads(p.read_bytes());spec['files']={n:sha(new/n) for n in spec['files']};p.write_text(json.dumps(spec,indent=2),encoding='utf8')
for p in new.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
changed={n for n in before if sha(new/n)!=before[n]};assert changed=={'prepare_training.py','dispatch.py','training_protocol.json','closeout_protocol.json'}
assert {p.name:sha(p) for p in old.iterdir()}==before
manifest={p.name:sha(p) for p in new.iterdir()};assert len(manifest)==29
(out/'source_files.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
with zipfile.ZipFile(out/'reviewed_source.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(new.iterdir()):z.writestr(p.name,p.read_bytes())
r=dict(status='PASS_NEW_LAUNCH_PROTOCOL_MODEL_INITIALIZATION_AND_EFFICACY_CRITERION_UNCHANGED',changed_files=sorted(changed),files=manifest,original_v2_source_preserved=True,old_exact_failure_preserved=True,old_training_epochs=0,training_started=False,BASE_forwards=0,zip_sha256=sha(out/'reviewed_source.zip'),builder_sha256=sha(__file__))
(out/'source_review.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r))
