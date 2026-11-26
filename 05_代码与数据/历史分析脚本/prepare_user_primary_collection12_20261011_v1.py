"""Adapt previously verified Warm primary collection to the three frozen user configurations."""
from pathlib import Path
import ast, base64, difflib, hashlib, json

ROOT = Path.cwd()
old = ROOT/'英文小论文/code/warm_stem_primary_collection12_20261011_v1'
dest = ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def change(text, before, after):
    assert before in text, before
    return text.replace(before, after)

assert sha(old/'collect.py') == '465c037b57f91239190c185282ded254658ad0cc0c725eead816e2a4d2ba7085'
assert sha(old/'verify_primary_v2.py') == 'e6fa14be35a5f2ac36be0485ea6c22bec669c4911d2b236c0cd1888ef9498351'
user = ROOT/'英文小论文/code/user_backbone12_20261011_v1'
expected = read(ROOT/'outputs/user_backbone12_20261011_v1/local_code_manifest_v2.json')
assert len(expected) == 28 and {p.name: sha(p) for p in user.iterdir() if p.is_file()} == expected
captured, trainers = {}, {}
for seed, value in read(ROOT/'outputs/user_primary_collection_identity12_20261011_v1_inputs.json').items():
    raw = value['payload'].encode()
    assert len(raw) == value['transport']['bytes'] and hashlib.sha256(raw).hexdigest() == value['transport']['sha256']
    current = json.loads(raw)
    assert current['seed'] == int(seed)
    captured[seed], trainers[seed] = current, {}
    for arm, job in current['jobs'].items():
        assert job['identity_seed'] == int(seed) and job['identity_arm'] == arm and job['identity_schema'] == 'user_backbone12_training_v1'
        trainers[seed][arm] = job['trainer']

collector = (old/'collect.py').read_text(encoding='utf8')
for before, after in [
    ("import argparse,datetime,hashlib,json,sys,zipfile", "import argparse,datetime,hashlib,json,sys,zipfile\nfrom guards import ARMS,identity_plan,positive_terminal"),
    ("assert arm=='warm_stem'", "assert arm in ARMS"),
    ('warm_stem12_20261011_v1', 'user_backbone12_20261011_v1'),
    ("protocol['warm_files_sha256']", "protocol['user_files_sha256']"),
    ("assert plan['seed']==seed and plan['arm']==arm", "assert plan['seed']==seed and plan['arm']==arm\n    identity_plan(receipt['identity'],plan,read(bundle/'cpu_initialization.json')['initialization'],arm,seed)"),
    ("assert done['entered_epochs']==list(range(12)) and done['identity']==identity", "assert done['entered_epochs']==list(range(12)) and done['identity']==identity\n    positive_terminal(identity,done,read(screen/'controller_completion.json'),read(screen/'candidate_completion.json'))"),
    ('warm_stem_primary_collection12_20261011_v1', 'user_backbone_primary_collection12_20261011_v1'),
    ('exact12_warm_primary_artifact_collection_v1', 'exact12_user_backbone_primary_artifact_collection_v1'),
    ("choices=('warm_stem',)", "choices=ARMS"),
    ("f'seed{a.seed}_collect", "f'seed{a.seed}_{a.arm}_collect"),
    ("seed=a.seed,source_sha256=sha(__file__)", "seed=a.seed,arm=a.arm,source_sha256=sha(__file__)"),
    ('STOPPED_WARM_PRIMARY12_PACKAGE_COLLECTED', 'STOPPED_USER_PRIMARY12_PACKAGE_COLLECTED'),
]:
    collector = change(collector, before, after)
(dest/'collect.py').write_text(collector, encoding='utf8')
reader = (old/'verify_primary_v2.py').read_text(encoding='utf8')
for before, after in [
    ("import argparse,hashlib,importlib.util,json,stat,sys,zipfile", "import argparse,hashlib,importlib.util,json,stat,sys,zipfile\nfrom guards import ARMS,identity_plan,positive_terminal"),
    ("assert arm=='warm_stem'", "assert arm in ARMS"),
    ('warm_stem_primary_collection12_20261011_v1', 'user_backbone_primary_collection12_20261011_v1'),
    ('STOPPED_WARM_PRIMARY12_PACKAGE_COLLECTED', 'STOPPED_USER_PRIMARY12_PACKAGE_COLLECTED'),
    ("read(collector/'protocol.json')['warm_files_sha256']", "read(collector/'protocol.json')['user_files_sha256']"),
    ("exact12_warm_primary_artifact_collection_v1", "exact12_user_backbone_primary_artifact_collection_v1"),
    ("plan['schema']=='warm_stem12_training_v1' and plan['arm']=='warm_stem'", "plan['schema']=='user_backbone12_training_v1' and plan['arm']==arm"),
    ("assert identity['native_parameter_count']==10231997 and identity['added_parameter_count']==11815", "assert sha(destination/'bundle/cpu_initialization.json')==identity['cpu_initialization_sha256']\n    initialization=read(destination/'bundle/cpu_initialization.json')['initialization']\n    identity_plan(identity,plan,initialization,arm,seed)\n    positive_terminal(identity,done,read(destination/'screen/controller_completion.json'),read(destination/'screen/candidate_completion.json'))"),
    ('warm_stem12_20261011_v1/runtime.py', 'user_backbone12_20261011_v1/runtime.py'),
    ('PASS_CANDIDATE_FULL_DETECTOR_AMP_LOSS_BACKWARD_NO_UPDATES', 'PASS_USER_CONFIG_FULL_AMP_LOSS_BACKWARD_NO_UPDATES'),
    ('PASS_WARM_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS', 'PASS_USER_PRIMARY_EXACT12_ALL_FILES_AND548_STAGE_ARRAYS'),
    ("choices=('warm_stem',)", "choices=ARMS"),
]:
    reader = change(reader, before, after)
reader = change(reader, "assert executed['receipt']['seed']==seed and not executed['receipt']['automatic_retry']", "assert executed['receipt']['seed']==seed and executed['receipt']['arm']==arm and not executed['receipt']['automatic_retry']")
reader = change(reader, "assert start['identity']==identity and start['optimizer_coverage']['all_parameters_exactly_once']", "assert start['identity']==identity and start['optimizer_coverage']['all_parameters_exactly_once']\n    assert start['optimizer_coverage']['all_native_parameter_groups_unchanged']\n    assert start['optimizer_coverage']['added_parameters']==identity['added_parameter_count']\n    assert start['initialization']==initialization\n    assert start['native_augmentation_and_supervision_weights_unchanged']\n    assert start['evaluator_runtime']['binding']==identity['evaluator_binding']")
reader = change(reader, "assert full['optimizer_updates']==full['training_epochs']==0 and full['candidate_detector_forwards']==2", "assert full['optimizer_updates']==full['training_epochs']==0 and full['candidate_detector_forwards']==2\n    assert full['initialization']==initialization and full['optimizer_coverage']['all_parameters_exactly_once']\n    assert [case['side'] for case in full['cases']]==[640,832]\n    assert all(case['finite_gradients'] for case in full['cases'])\n    if arm=='p2_skip':assert all(all(case['branch'][k] for k in ('zero_restore_maintained','restore_gradient_active','upstream_bridge_gradients_initially_zero')) for case in full['cases'])")
reader = change(reader, "collector=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'", "collector=ROOT/'英文小论文/code/user_backbone_primary_collection12_20261011_v1'\n    protocol=read(collector/'protocol.json')\n    assert sha(__file__)==protocol['local_verifier_sha256']\n    assert sha(collector/'guards.py')==protocol['remote_sources']['guards.py']")
(dest/'verify_primary.py').write_text(reader, encoding='utf8')
diff = ''.join(difflib.unified_diff((old/'collect.py').read_text().splitlines(keepends=True), collector.splitlines(keepends=True), fromfile='verified_Warm_collect.py', tofile='user_collect.py'))
diff += ''.join(difflib.unified_diff((old/'verify_primary_v2.py').read_text().splitlines(keepends=True), reader.splitlines(keepends=True), fromfile='verified_Warm_reader_v2.py', tofile='user_reader.py'))
(dest/'adaptation.diff').write_text(diff, encoding='utf8')
for name in ('collect.py', 'wait_stopped.py', 'guards.py', 'verify_primary.py'):
    ast.parse((dest/name).read_text(encoding='utf8'))
protocol = dict(schema='user_backbone_read_only_primary_collection12_v1',
    collector_sha256=sha(dest/'collect.py'), local_verifier_sha256=sha(dest/'verify_primary.py'),
    user_files_sha256=expected, existing_trainers=trainers, max_wait_seconds=86400,
    execution_allowlist={a: [1, 2] for a in ('backbone_silu','p2_skip','backbone_attention')},
    remote_sources={n: sha(dest/n) for n in ('collect.py','wait_stopped.py','guards.py')},
    adaptation_source_sha256={'Warm_collect.py': sha(old/'collect.py'), 'Warm_verify_primary_v2.py': sha(old/'verify_primary_v2.py')},
    exact12_required=True, both_actual_trainer_and_controller_stopped_required=True,
    independent_saved_EMA_required_for_formal_endpoint=True, training_commands=0, new_model_forwards=0,
    new_evaluator_calls=0, new_training_epochs=0, existing_process_signals=0, automatic_retry=False,
    scope='Read-only archival of existing three user configurations only, not new model/BASE evaluation or training. Pending formal saved-EMA replay remains a separate gate.')
write(dest/'protocol.json', protocol)
files={n:dict(base64=base64.b64encode((dest/n).read_bytes()).decode(),bytes=(dest/n).stat().st_size,sha256=sha(dest/n)) for n in ('collect.py','wait_stopped.py','guards.py','protocol.json')}
write(ROOT/'outputs/user_primary_collection12_20261011_v1_deploy_files.json', files)
print(json.dumps(dict(status='USER_PRIMARY_COLLECTION_AND_STRICT_READER_SOURCE_BOUND', files={n:v['sha256'] for n,v in files.items()}, local_reader_sha256=sha(dest/'verify_primary.py'))))
