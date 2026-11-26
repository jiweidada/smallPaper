"""Freeze corrected observation runtime for the never-trained seed2 only."""
from pathlib import Path
import ast,datetime,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'英文小论文/code/small_object_encoder_decoder12_20261010_v2'
NEW=OLD.with_name('small_object_encoder_decoder12_20261010_v3')
OUT=ROOT/'outputs/small_object_encoder_decoder12_20261010_v3'
REPAIR=Path(__file__).with_name('training_audit.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    NEW.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
    names=['runtime.py','experiment.py','baseline_eval.py','supervise.py','training_audit.py','results.py','stage_trace.py','small_object_model.py','geometry_memory.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','prepare_remote.py']
    for n in names:(NEW/n).write_bytes((OLD/n).read_bytes())
    tested=REPAIR.read_text(encoding='utf8');body=tested[tested.index('import hashlib'):]
    (NEW/'training_audit.py').write_text('"""Native input parity, passive first8 observations and epoch1 learning checks.\n\nUses the tested audit repair. This file does not authorize restarting a stopped\nconfiguration; the frozen protocol limits this deployment to fresh seed2.\n"""\n'+body,encoding='utf8')
    assert (NEW/'training_audit.py').read_text().split('import hashlib',1)[1]==tested.split('import hashlib',1)[1]
    p=NEW/'prepare_remote.py';s=p.read_text();needle="prefix=Path(a.prefix);experiment=HERE.parent;plan_path=HERE/f'configs/fine_seed{a.seed}.json';plan=read(plan_path)";assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    assert a.seed==2 and read(HERE/'protocol.json')['execution_allowlist']=={'fine':[2],'degraded':[]},'Only never-trained seed2 is authorized in this deployment'\n    for version in (1,2):\n        prior=prefix/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}/seed2_fine'\n        assert not (prior/'candidate_start.json').exists() and not (prior/'run').exists(),'Seed2 has prior candidate training; do not reset its budget'")
    needle="write(bundle/'image_manifest.json',images)";assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    assert sha(bundle/'image_manifest.json')=='405381d365b72f6a1b700a5c8820056fa129fab8e7f558765f5168266f6b53f5','Images changed from original paired identity'")
    p.write_text(s,encoding='utf8')
    p=NEW/'experiment.py';s=p.read_text();needle='receipt,plan,paths=verify_bundle(bundle)';assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    assert plan['seed']==2 and plan['variant']=='fine','This deployment permits only fresh fine seed2; seed1 budget decision pending'");p.write_text(s,encoding='utf8')
    p=NEW/'supervise.py';s=p.read_text();needle="receipt,plan,paths=verify_bundle(a.bundle);root=paths['screen_root'];protocol=read(Path(__file__).with_name('protocol.json'))";assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    assert plan['seed'] in protocol['execution_allowlist'][plan['variant']],'Configuration not authorized in this deployment'");p.write_text(s,encoding='utf8')
    protocol=read(OLD/'protocol.json');protocol.update(schema='small_object_encoder_decoder12_20261010_v3',status='FROZEN_FRESH_SEED2_ONLY_BEFORE_TRAINING',frozen_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor_protocol_sha256=sha(OLD/'protocol.json'),execution_allowlist={'fine':[2],'degraded':[]},training_integrity='First100 pre-forward input/target/RNG hashes exact. First8 real optimizer updates passively record norms/exact motion/raw parameter tensors; exact movement of each added group checked after first complete epoch. Native EMA count equals forward attempts, including AMP-skipped optimizer steps; actual optimizer steps separate. Same fixed12 endpoint and native recipe.',fresh_seed2_authority={'source':'Existing user authorization allows each configuration at most12 total epochs. Seed2 candidate never started in either prior runtime.','prior_seed2_candidate_attempts':0,'prior_seed2_candidate_updates':0,'seed1_failed_attempts_preserved':10,'seed1_failed_updates_preserved':8,'seed1_restart_budget_exception_approved':False,'seed1_action':'Keep stopped; user budget attribution pending.','not_a_new_seed_search':True},audit_correction={'tested_repair_sha256':sha(REPAIR),'gradient_validation_sha256':sha(ROOT/'outputs/encoder_decoder_gradient_diagnosis_20261010_v1/validation_report.json'),'repair_mock_validation_sha256':sha(ROOT/'outputs/encoder_decoder_audit_repair_20261010_v2/collection/encoder_decoder_audit_repair_20261010_v2/mock_results/completion.json'),'changes':['Remove premature8-step group-motion failure; require actual motion by first complete epoch.','Record FP64 norms,exact nonzero/moved counts and raw step8 diagnostic parameters.','Audit native EMA updates per attempt; no change to native EMA behavior.'],'actual_full_epoch_training_integration_verified':False,'model_optimizer_data_loss_schedule_changes':False})
    write(NEW/'protocol.json',protocol);write(OUT/'protocol.json',protocol)
    (NEW/'configs').mkdir()
    for p in (OLD/'configs').glob('*.json'):
        plan=read(p);plan.update(schema=protocol['schema'],screen_protocol_sha256=sha(NEW/'protocol.json'));write(NEW/'configs'/p.name,plan)
    unchanged=['geometry_memory.py','small_object_model.py','runtime.py','baseline_eval.py','results.py','stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json']
    assert all(sha(NEW/n)==sha(OLD/n) for n in unchanged)
    for p in NEW.glob('*.py'):ast.parse(p.read_text(),filename=p.name)
    files={p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()};write(NEW/'deployment_manifest.json',files)
    z=OUT/'deploy_bundle.zip'
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as f:
        for n in list(files)+['deployment_manifest.json']:f.write(NEW/n,n)
    meta={'status':'FROZEN_NOT_LAUNCHED','files':len(files)+1,'archive_sha256':sha(z),'archive_bytes':z.stat().st_size,'protocol_sha256':sha(NEW/'protocol.json'),'model_sources_unchanged':True,'unchanged_files':unchanged,'allowed_fresh_seed':2,'seed2_prior_training_updates':0,'seed1_budget_exception_assumed':False}
    write(OUT/'deployment_local.json',meta);print(json.dumps(meta))
if __name__=='__main__':main()
