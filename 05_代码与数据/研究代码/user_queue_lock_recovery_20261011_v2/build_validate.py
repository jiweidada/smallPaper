from pathlib import Path
import ast
import copy
import hashlib
import json
import difflib
import zipfile
from lock_support import assert_recoverable

ROOT = Path.cwd()
HERE = ROOT / '英文小论文/code/user_queue_lock_recovery_20261011_v2'
USER = ROOT / '英文小论文/code/user_backbone12_20261011_v1'
OUT = ROOT / 'outputs/user_queue_lock_recovery_20261011_v2'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = (USER / 'dispatch.py').read_text(encoding='utf8')
assert sha(USER / 'dispatch.py') == 'deaec7e84fc08e0e7feda3b743a83ac59786156d1740c53db04b3507d5ddbd8e'
changes = [
    ("    queue.mkdir(exist_ok=False)", "    guard = guard_recovery(root, seed)"),
    ("    write(queue/'start.json',owner)", "    write(queue/'recovery_start_v2.json',dict(owner,guard=guard,original_dispatcher=read(root/f'queue_dispatch_seed{seed}.prelock_v1.json')))\n    publish_replacement(root, seed, owner)"),
    ("    lock=(root/f'launch_seed{seed}.lock').open('a')", "    lock=LocalLaunchLock(root, seed)"),
    ("                    fcntl.flock(lock,fcntl.LOCK_EX)", "                    lock.acquire()"),
    ("                        fcntl.flock(lock,fcntl.LOCK_UN)", "                        lock.release()"),
]
patched = source
for old, new in changes:
    assert patched.count(old) == 1, old
    patched = patched.replace(old, new)
preamble = '''from pathlib import Path as RecoveryPath
import sys as recovery_sys
recovery_sys.path.insert(0, str(RecoveryPath(__file__).resolve().parents[1] / 'code'))
from lock_support import guard_recovery, publish_replacement, LocalLaunchLock
'''
generated = preamble + patched
ast.parse(generated)
(HERE / 'dispatch.py').write_text(generated, encoding='utf8')
valid = dict(frozen28_source_exact=True, old_dispatcher_stopped=True, old_children=[],
    predecessor_warm_trainer_live=True, actual_configurations_started=0, actual_completed_epochs=0,
    queue_failure=False, existing_recovery=False, local_tmp_filesystem='overlay')
assert_recoverable(valid)
rejections = []
for key, value in [('frozen28_source_exact', False), ('old_dispatcher_stopped', False),
    ('old_children', ['123']), ('predecessor_warm_trainer_live', False),
    ('actual_configurations_started', 1), ('actual_completed_epochs', 1),
    ('queue_failure', True), ('existing_recovery', True), ('local_tmp_filesystem', 'nfs')]:
    item = copy.deepcopy(valid); item[key] = value
    try:
        assert_recoverable(item)
    except AssertionError:
        rejections.append(key)
    else:
        raise AssertionError('Missing rejection: ' + key)
OUT.mkdir(exist_ok=False)
report = dict(status='PASS_ONE_POSITIVE_AND_NINE_NEGATIVE_ZERO_START_HANDOVER_FIXTURES',
    rejected=rejections, original_dispatch_sha256=sha(USER / 'dispatch.py'),
    replacement_dispatch_sha256=sha(HERE / 'dispatch.py'), lock_support_sha256=sha(HERE / 'lock_support.py'),
    source_changes=[dict(old=a,new=b) for a,b in changes],
    frozen_model_training28_sources_changed=False, training_retry=False,
    max_epochs_each=12, BASE_forwards=0, model_forwards=0, training_epochs=0)
(OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
(OUT / 'dispatcher.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), generated.splitlines(True),
    fromfile='frozen_code/dispatch.py', tofile='recovery_v2/dispatch.py')), encoding='utf8')
manifest = dict(frozen28_files={p.name:sha(p) for p in USER.iterdir() if p.is_file()},
    recovery_files={n:sha(HERE/n) for n in ('dispatch.py','lock_support.py')},
    validation_sha256=sha(OUT/'validation.json'), CPU_validation=report['status'],
    scope='Scheduling-only recovery before any of the three configurations has started. Preserve immutable code, recipe, initial states, counters, original queue and live old trainers. Replace the public coordinator handle so all waiting diagnostics follow the actual replacement.')
assert len(manifest['frozen28_files']) == 28
(HERE / 'recovery_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
with zipfile.ZipFile(OUT / 'recovery_v2.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for n in ('dispatch.py','lock_support.py','recovery_manifest.json'):
        z.write(HERE/n,n)
(OUT / 'archive.json').write_text(json.dumps(dict(bytes=(OUT/'recovery_v2.zip').stat().st_size,
    sha256=sha(OUT/'recovery_v2.zip'), files=3),indent=2),encoding='utf8')
print(json.dumps(report))
