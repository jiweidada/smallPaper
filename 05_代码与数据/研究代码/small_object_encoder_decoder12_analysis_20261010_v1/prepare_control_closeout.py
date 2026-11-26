"""Freeze isolated degraded-control closeout helpers; never runs a model."""
from pathlib import Path
import ast, datetime, hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[3]
OLD = Path(__file__).resolve().parent
NEW = OLD.with_name('small_object_encoder_decoder12_control_analysis_20261010_v1')
OUT = ROOT/'outputs/small_object_encoder_decoder12_20261010_v5/closeout_tools_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf8'))


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def write(p, value):
    with p.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    assert not NEW.exists() and not OUT.exists()
    NEW.mkdir(); OUT.mkdir()
    originals = {}
    for name in ('results.py', 'stage_trace.py', 'evaluator_binding.py', 'evaluator_sources.json',
                 'geometry_core.py', 'input_scale_breakdown.py', 'verify_gradient_movement.py'):
        (NEW/name).write_bytes((OLD/name).read_bytes()); originals[name] = sha(OLD/name)
    protocol = read(OLD/'protocol.json')
    protocol.update(status='CONTROL_DESCRIPTIVE_ANALYSIS_FROZEN_DURING_EPOCH1',
        frozen_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary_gate_unchanged='Both fine native fixed12 gates passed. FineAPs>degradedAPs for each seed is the unchanged prospective mechanism gate; no new endpoint or tuning.',
        candidate_arm='degraded', runtime_version=5)
    write(NEW/'protocol.json', protocol)
    analyzer = replace((OLD/'analyze_endpoint.py').read_text(encoding='utf8'),
        'def analyze(baseline,annotation_path,output,candidate=None):',
        "def analyze(baseline,annotation_path,output,candidate=None,candidate_arm='degraded'):")
    analyzer = replace(analyzer, "right,rh,ra,rd,rp=inspect_run(candidate,'fine',annotation)",
        "assert candidate_arm in ('fine','degraded')\n        right,rh,ra,rd,rp=inspect_run(candidate,candidate_arm,annotation)")
    (NEW/'analyze_endpoint.py').write_text(analyzer, encoding='utf8')

    preflight = '''def verify_tools():
    here=Path(__file__).resolve().parent
    protocol=json.loads((here/'protocol.json').read_text())
    assert protocol['runtime_version']==5 and protocol['candidate_variant']=='degraded'
    for name,h in protocol['files'].items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest()==h,name


def assert_trainers_stopped(screen):
    for name,pk,tk in [('controller_start.json','pid','start_ticks'),('candidate_start.json','child_pid','child_start_ticks')]:
        record=json.loads((screen/name).read_text());proc=Path(f"/proc/{record[pk]}/stat")
        if proc.exists():
            fields=proc.read_text().split()
            assert fields[21]!=str(record[tk]) or fields[2]=='Z','Owned trainer or supervisor is still live'


'''
    reload = (OLD/'verify_candidate_reload.py').read_text(encoding='utf8')
    reload = replace(reload, 'def worker(bundle):', preflight+'def worker(bundle):\n    verify_tools()')
    reload = replace(reload, 'receipt,plan,paths=verify_bundle(bundle)',
        "receipt,plan,paths=verify_bundle(bundle)\n    assert plan['variant']=='degraded' and receipt['identity']['arm']=='degraded'\n    assert_trainers_stopped(paths['screen_root'])")
    reload = replace(reload, 'def supervise(bundle):', 'def supervise(bundle):\n    verify_tools()')
    reload = replace(reload, "assert json.loads((screen/'controller_completion.json').read_text())['status']=='COMPLETE_FIXED12'",
        "assert json.loads((screen/'controller_completion.json').read_text())['status']=='COMPLETE_FIXED12'\n    assert_trainers_stopped(screen)\n    assert receipt['identity']['arm']=='degraded'")
    reload = replace(reload, "'seed':plan['seed'],'checkpoint_sha256'", "'seed':plan['seed'],'variant':'degraded','checkpoint_sha256'")
    (NEW/'verify_candidate_reload.py').write_text(reload, encoding='utf8')

    collector = (OLD/'collect_terminal.py').read_text(encoding='utf8')
    collector = replace(collector, 'def collect(seed,check_only=False):',
        preflight+'def collect(seed,check_only=False):\n    verify_tools()')
    collector = replace(collector, "version={1:4,2:3}[seed]", 'version=5')
    collector = replace(collector, "screen=root/f'seed{seed}_fine'", "screen=root/f'seed{seed}_degraded'")
    collector = replace(collector, "plan['variant']=='fine'", "plan['variant']=='degraded'")
    collector = replace(collector, "'candidate.log','baseline.log','controller_live.json','one_shot_launch.lock'):",
        "'candidate.log','baseline.log','controller_live.json','one_shot_launch.lock','dual_seed_fine_promotion.json'):")
    collector = replace(collector, "tree('code',code);tree('prepared',bundle)",
        "tree('code',code);tree('prepared',bundle)\n    reuse=read(screen/'baseline_reuse.json')\n    assert reuse['new_model_forwards']==reuse['new_optimizer_steps']==reuse['new_epochs']==0\n    for n in ('preparation_start.json','preparation_completion.json','preparation.log',f'supervisor_launch_seed{seed}.json'):\n        add('deployment/'+n,root/n)")
    collector = replace(collector, "'status':state,'seed':seed,'runtime_version':version,",
        "'status':state,'seed':seed,'runtime_version':version,'candidate_variant':'degraded',")
    collector = replace(collector, "'sha256':sha(archive),'files':len(files)+1,'seed':seed}",
        "'sha256':sha(archive),'files':len(files)+1,'seed':seed,'runtime_version':5,'candidate_variant':'degraded'}")
    (NEW/'collect_terminal.py').write_text(collector, encoding='utf8')

    verifier = (OLD/'verify_terminal_collection.py').read_text(encoding='utf8')
    verifier = replace(verifier, 'metadata=read(metadata_path);manifest=extract_checked(archive,metadata,destination)',
        "metadata=read(metadata_path)\n    assert metadata.get('candidate_variant')=='degraded' and metadata.get('runtime_version')==5,'Wrong experiment arm/version'\n    manifest=extract_checked(archive,metadata,destination)")
    verifier = replace(verifier, "version={1:4,2:3}[seed];assert manifest['runtime_version']==version",
        "version=5;assert manifest['runtime_version']==version and manifest['candidate_variant']=='degraded'")
    verifier = replace(verifier, "assert manifest['identity']==identity",
        "assert manifest['identity']==identity\n    assert identity['arm']=='degraded' and identity['seed']==seed")
    verifier = replace(verifier, "assert read(run/'runtime_start.json')['identity']==identity",
        "runtime=read(run/'runtime_start.json');assert runtime['identity']==identity\n    init=runtime['initialization']\n    assert init['variant']=='degraded' and init['parameters']==13806625 and init['extra_parameters']==3574628\n    assert init['native_state_exact'] and init['independent_geometry_layer_copies_exact']\n    assert runtime['optimizer_coverage']['all_parameters_exactly_once']\n    reuse=read(destination/'screen/baseline_reuse.json')\n    assert reuse['new_model_forwards']==reuse['new_optimizer_steps']==reuse['new_epochs']==0\n    from control_proofs import verify_promotion_packet\n    promotion=verify_promotion_packet(destination)")
    verifier = replace(verifier,
        "bound_tools=read(ROOT/'outputs/small_object_encoder_decoder12_analysis_20261010_v1/closeout_tools_v1/metadata.json')['files']",
        "local_metadata=ROOT/'outputs/small_object_encoder_decoder12_20261010_v5/closeout_tools_v1/metadata.json'\n    assert read(destination/'tools/metadata.json')==read(local_metadata)\n    bound_tools=read(local_metadata)['files']")
    verifier = replace(verifier, "assert proof['checkpoint_sha256']==done['last_checkpoint_sha256']",
        "assert proof['variant']=='degraded' and proof['seed']==seed\n    assert proof['checkpoint_sha256']==done['last_checkpoint_sha256']")
    verifier = replace(verifier, "gradient_driven_movement=movement,", "gradient_driven_movement=movement,dual_seed_fine_promotion=promotion,variant='degraded',")
    verifier = replace(verifier,
        "scope='One seed exploratory fixed12 result. Both seeds required before capacity-matched controls; no significance,stability,novelty or causal-root claim.'",
        "scope='One seed fixed12 capacity control. The prospective mechanism gate compares fineAPs>degradedAPs in both seeds. No significance,stability,novelty or causal-root claim.'")
    (NEW/'verify_terminal_collection.py').write_text(verifier, encoding='utf8')

    for p in NEW.glob('*.py'): ast.parse(p.read_text(encoding='utf8'), filename=str(p))
    helper_files={n:sha(NEW/n) for n in ('verify_candidate_reload.py','collect_terminal.py')}
    helper_protocol={'scope':'Post-fixed12 degraded-control saved-EMA verification and terminal collection only.',
        'created_local_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runtime_version':5,'candidate_variant':'degraded','files':helper_files,
        'new_training_updates':0,'new_baseline_forwards':0,'candidate_reload_forwards_after12':137,
        'no_automatic_retry':True,'no_automatic_next_configuration':True,
        'required_before_reload':'Owned original trainer/supervisor stopped exact12; expected GPU idle.',
        'required_before_archive':'Owned trainer/supervisor/reload stopped; preserve failed terminal evidence.'}
    write(OUT/'protocol.json',helper_protocol)
    for name in helper_files: (OUT/name).write_bytes((NEW/name).read_bytes())
    files={n:sha(OUT/n) for n in (*helper_files,'protocol.json')}
    metadata={'schema':'degraded_control_closeout_tools_v1','files':files,
        'original_fine_helper_sha256':{n:sha(OLD/n) for n in helper_files},
        'model_or_optimizer_calls_during_preparation':0,'runtime_version':5,'variant':'degraded'}
    write(OUT/'metadata.json',metadata)
    archive=OUT/'tools.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in (*files,'metadata.json'): z.write(OUT/n,n)
    write(OUT/'local_freeze.json',{'status':'FROZEN_NOT_DEPLOYED','archive_sha256':sha(archive),
        'archive_bytes':archive.stat().st_size,'files':len(files)+1,
        'unchanged_analysis_dependencies':originals,'new_baseline_forwards':0,
        'new_candidate_forwards':0,'new_optimizer_updates':0,
        'python_AST_files':len(list(NEW.glob('*.py')))})
    print(json.dumps({'status':'CONTROL_CLOSEOUT_FROZEN','archive':str(archive),
        'bytes':archive.stat().st_size,'sha256':sha(archive)},ensure_ascii=False))


if __name__=='__main__': main()
