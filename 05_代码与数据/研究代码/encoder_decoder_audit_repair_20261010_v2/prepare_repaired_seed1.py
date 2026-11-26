"""Freeze the user-authorized seed1 restart and reuse its verified BASE12."""
from pathlib import Path
import ast, datetime, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / '英文小论文/code/small_object_encoder_decoder12_20261010_v3'
NEW = OLD.with_name('small_object_encoder_decoder12_20261010_v4')
OUT = ROOT / 'outputs/small_object_encoder_decoder12_20261010_v4'
ARCHIVED = ROOT / 'outputs/small_object_encoder_decoder12_20261010_v2/collection/terminal_seed1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


def write(p, value):
    with Path(p).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def replace_once(p, old, new):
    text = p.read_text(encoding='utf8')
    assert text.count(old) == 1, (p.name, old)
    p.write_text(text.replace(old, new), encoding='utf8')


def main():
    state = read(ROOT / 'outputs/sod_roi12_20261009/execution_state.json')
    authorization = state['authorization']['repaired_seed1_restart']
    assert authorization['authorized'] and authorization['user_words'] == '代码 改好就上机跑代码'
    assert authorization['formal_max_epochs_per_seed'] == 12
    NEW.mkdir(exist_ok=False); OUT.mkdir(exist_ok=False)
    for p in OLD.iterdir():
        if p.is_file() and p.name not in ('protocol.json', 'deployment_manifest.json'):
            (NEW / p.name).write_bytes(p.read_bytes())
    (NEW / 'reuse_baseline.py').write_bytes(Path(__file__).with_name('reuse_baseline.py').read_bytes())
    replace_once(NEW / 'runtime.py', "IMPLEMENTATION=('runtime.py',", "IMPLEMENTATION=('reuse_baseline.py','runtime.py',")
    replace_once(NEW / 'training_audit.py',
        'configuration; the frozen protocol limits this deployment to fresh seed2.',
        'configuration; the frozen protocol records the user-authorized seed1 restart.')
    replace_once(NEW / 'experiment.py',
        "assert plan['seed']==2 and plan['variant']=='fine','This deployment permits only fresh fine seed2; seed1 budget decision pending'",
        "assert plan['seed']==1 and plan['variant']=='fine','Only the user-authorized corrected seed1 is allowed'")
    replace_once(NEW / 'supervise.py',
        'One local host, one BASE12 evaluation and one fixed12 candidate. No queue.',
        'One verified cached BASE12 and one fixed12 candidate. No baseline rerun or queue.')
    replace_once(NEW / 'supervise.py',
        "phase(root,'baseline',[sys.executable,'-B',str(Path(__file__).with_name('baseline_eval.py')),'--bundle',a.bundle],1800)",
        "from reuse_baseline import attach_cached_baseline\n        attach_cached_baseline(receipt,plan,paths)")
    original_guard = """assert a.seed==2 and read(HERE/'protocol.json')['execution_allowlist']=={'fine':[2],'degraded':[]},'Only never-trained seed2 is authorized in this deployment'
    for version in (1,2):
        prior=prefix/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}/seed2_fine'
        assert not (prior/'candidate_start.json').exists() and not (prior/'run').exists(),'Seed2 has prior candidate training; do not reset its budget'"""
    new_guard = """protocol=read(HERE/'protocol.json')
    assert a.seed==1 and protocol['execution_allowlist']=={'fine':[1],'degraded':[]}
    assert protocol['repaired_seed1_authority']['authorized']
    prior=prefix/'DEIM/experiments/small_object_encoder_decoder12_20261010_v2/seed1_fine'
    for name,expected_sha in protocol['failed_engineering_attempt']['pinned_files'].items():
        assert sha(prior/name)==expected_sha,'Original failed attempt changed'
    rows=read(prior/'run/first8_optimizer_steps.json')
    assert len(rows)==8 and rows[-1]['actual_optimizer_step']==8 and rows[-1]['attempt']==10
    assert len((prior/'run/first100_training_inputs.jsonl').read_text().splitlines())==10
    assert not (prior/'run/last.pth').exists() and not (prior/'run/training_completion.json').exists()
    assert read(prior/'candidate_completion.json')['exit_code']==1
    for name,pid_key,tick_key in (('candidate_start.json','child_pid','child_start_ticks'),('controller_start.json','pid','start_ticks')):
        old_process=read(prior/name);proc=Path(f\"/proc/{old_process[pid_key]}/stat\")
        if proc.exists():assert proc.read_text().split()[21]!=str(old_process[tick_key]),'Prior process still alive'
    for version in (1,3):
        other=prefix/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}/seed1_fine'
        assert not (other/'candidate_start.json').exists() and not (other/'run').exists(),'Unexpected other seed1 attempt'"""
    replace_once(NEW / 'prepare_remote.py', original_guard, new_guard)
    proof_anchor = "print(json.dumps({'status':'PREPARED','bundle':str(bundle),'image_files':7019,'optimizer_steps':0,'epochs_added':0}),flush=True)"
    replace_once(NEW / 'prepare_remote.py', proof_anchor,
        "from reuse_baseline import verify_cached_baseline\n    _,proof=verify_cached_baseline(read(bundle/'receipt.json'),plan,paths)\n    write(bundle/'baseline_cache_check.json',proof)\n    " + proof_anchor)
    protocol = read(OLD / 'protocol.json')
    protocol.pop('fresh_seed2_authority')
    source = '/25085400120/DEIM/experiments/small_object_encoder_decoder12_20261010_v2/seed1_fine/baseline_eval'
    cache = ARCHIVED / 'seed1_fine/baseline_eval'
    pinned = {n: sha(cache / n) for n in ('completion.json', 'historical_comparison.json',
        'baseline.yml', 'endpoint_epoch12/manifest.json', 'stages_epoch12/manifest.json')}
    failed = {n: sha(ARCHIVED / 'seed1_fine' / n) for n in ('controller_failure.json',
        'candidate_completion.json', 'run/first8_optimizer_steps.json', 'run/first100_training_inputs.jsonl')}
    protocol.update(schema='small_object_encoder_decoder12_20261010_v4',
        status='FROZEN_AUTHORIZED_REPAIRED_SEED1_CACHED_BASELINE',
        frozen_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        predecessor_protocol_sha256=sha(OLD / 'protocol.json'),
        execution_allowlist={'fine': [1], 'degraded': []},
        repaired_seed1_authority=authorization,
        failed_engineering_attempt={'attempts': 10, 'actual_optimizer_updates': 8,
            'completed_epochs': 0, 'preserved': True, 'pinned_files': failed,
            'accounting': 'Retained failed engineering attempt; latest direct run instruction interpreted as authorization for one repaired fresh formal12. No further automatic repeat.'},
        baseline_reuse={'source': source, 'pinned_files': pinned,
            'reason': 'Both BASE12 seeds already verified. User explicitly questioned unnecessary baseline reruns; only hash/config/result identity checks now.',
            'local_evidence_validation_sha256': sha(ROOT / 'outputs/small_object_encoder_decoder12_20261010_v2/terminal_evidence_validation.json'),
            'new_baseline_forwards': 0, 'new_baseline_epochs': 0},
        baseline_reproduction_gate='Reuse original exact historical12-stat PASS with pinned data/checkpoint/evaluator/config/evidence; no additional GPU evaluation.',
        stage1='Fine geometry field seed1 repaired formal12; seed2 v3 already training independently. Retain prior seed1 failed10 attempts/8 updates separately.',
        failure='No auto retries/resume/rollback/recipe change/extensions. Preserve all previous and new logs/locks/weights. Candidate timeout64800seconds.')
    write(NEW / 'protocol.json', protocol); write(OUT / 'protocol.json', protocol)
    (NEW / 'configs').mkdir()
    for p in (OLD / 'configs').glob('*.json'):
        plan=read(p); plan.update(schema=protocol['schema'], screen_protocol_sha256=sha(NEW / 'protocol.json'))
        write(NEW / 'configs' / p.name, plan)
    unchanged = ['geometry_memory.py','small_object_model.py','baseline_eval.py','results.py',
                 'stage_trace.py','evaluator_binding.py','evaluator_sources.json','source_binding.json']
    assert all(sha(NEW / n) == sha(OLD / n) for n in unchanged)
    assert (NEW / 'training_audit.py').read_text().split('import hashlib',1)[1] == (OLD / 'training_audit.py').read_text().split('import hashlib',1)[1]
    for p in NEW.glob('*.py'): ast.parse(p.read_text(), filename=p.name)
    files={p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()}
    write(NEW / 'deployment_manifest.json', files)
    archive=OUT / 'deploy_bundle.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in list(files)+['deployment_manifest.json']:z.write(NEW/n,n)
    meta={'status':'FROZEN_NOT_DEPLOYED','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
          'protocol_sha256':sha(NEW/'protocol.json'),'files':len(files)+1,'unchanged_files':unchanged,
          'training_audit_body_unchanged':True,'new_baseline_forwards':0,'formal_epoch_limit':12,
          'prior_failed_attempts_preserved':10,'prior_failed_updates_preserved':8}
    write(OUT/'deployment_local.json',meta);print(json.dumps(meta))


if __name__=='__main__':main()
