"""Repair the local reader's preflight evidence location, not the model/replay gates."""
from pathlib import Path
import hashlib
import json
import datetime

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / '英文小论文/code/p2_lowrank12_local_reader_20261011_v1'
NEW = ROOT / '英文小论文/code/p2_lowrank12_local_reader_20261011_v2'
EVIDENCE = ROOT / 'outputs/p2_lowrank12_local_reader_20261011_v2'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


assert not NEW.exists() and not EVIDENCE.exists()
source = (OLD / 'verify_terminal_local.py').read_bytes()
assert sha(source) == '1065525f27f2cf3c7909ce3fa20e510b9c6da334279dd9553708fb564c2c776c'
assert sha((OLD / 'local_reader_protocol.json').read_bytes()) == '2e9adbd5067764e9ab31c5f581073053d4b332a145b6eb5a3212e74e080809c7'
inputs = ROOT / 'outputs/turn7_lowrank_local_reader_actual_inputs.json'
data = json.loads(inputs.read_text(encoding='utf-8'))
NEW.mkdir()
EVIDENCE.mkdir()
protocol = json.loads((OLD / 'local_reader_protocol.json').read_text(encoding='utf-8'))
protocol['schema'] = 'p2_lowrank12_local_reader_v2'
protocol['preflight_sidecars'] = {}
for seed in (1, 2):
    captured = data[f'turn7_lowrank_local_reader_actual_runtime_inputs_seed{seed}']
    raw = captured['payload']
    assert isinstance(raw, str)
    assert sha(raw.encode()) == captured['transport']['sha256']
    assert len(raw.encode()) == captured['transport']['bytes']
    amp = json.loads(raw)['files']['AMP']
    options = [json.dumps(amp['report'], ensure_ascii=ascii_mode, indent=2).encode('utf-8') + suffix for ascii_mode in (False, True) for suffix in (b'', b'\n')]
    matches = [b for b in options if sha(b) == amp['sha256'] and len(b) == amp['bytes']]
    assert len(matches) >= 1, 'Captured AMP report cannot be reconstructed byte-exactly.'
    path = EVIDENCE / f'seed{seed}_preflight_report.json'
    path.write_bytes(matches[0])
    protocol['preflight_sidecars'][str(seed)] = {'relative_path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': amp['sha256'], 'bytes': amp['bytes'], 'source_capture_relative_path': str(inputs.relative_to(ROOT)).replace('\\', '/'), 'source_capture_sha256': sha(inputs.read_bytes())}

code = source.decode('utf-8')
code = code.replace("p['schema'] == 'p2_lowrank12_local_reader_v1'", "p['schema'] == 'p2_lowrank12_local_reader_v2'")
old = "    preflight=read(screen/'detector_preflight_v1/report.json')"
new = """    preflight_binding = protocol['preflight_sidecars'][str(identity['seed'])]
    preflight_path = (ROOT / preflight_binding['relative_path']).resolve()
    assert preflight_path.is_relative_to(ROOT)
    assert sha(preflight_path) == preflight_binding['sha256']
    assert preflight_path.stat().st_size == preflight_binding['bytes']
    capture_path = (ROOT / preflight_binding['source_capture_relative_path']).resolve()
    assert capture_path.is_relative_to(ROOT) and sha(capture_path) == preflight_binding['source_capture_sha256']
    capture = read(capture_path)['turn7_lowrank_local_reader_actual_runtime_inputs_seed'+str(identity['seed'])]
    assert hashlib.sha256(capture['payload'].encode()).hexdigest() == capture['transport']['sha256']
    captured_amp = json.loads(capture['payload'])['files']['AMP']
    assert captured_amp['sha256'] == preflight_binding['sha256'] and captured_amp['report'] == read(preflight_path)
    preflight=read(preflight_path)"""
assert code.count(old) == 1
code = code.replace(old, new)
old2 = "        verification_script_sha256=sha(__file__), local_reader_protocol_sha256=sha(PROTOCOL),"
new2 = "        verification_script_sha256=sha(__file__), local_reader_protocol_sha256=sha(PROTOCOL),\n        preflight_evidence_path=str(preflight_path), preflight_evidence_sha256=sha(preflight_path),"
assert code.count(old2) == 1
code = code.replace(old2, new2)
assert 'all(np.array_equal(a[k], b[k]) for k in a.files)' in code
assert code.count('all(np.array_equal(a[k], b[k]) for k in a.files)') == source.decode().count('all(np.array_equal(a[k], b[k]) for k in a.files)')
(NEW / 'verify_terminal_local.py').write_text(code, encoding='utf-8')
protocol['reader_files_sha256'] = {'verify_terminal_local.py': sha((NEW / 'verify_terminal_local.py').read_bytes())}
protocol['previous_reader_protocol_sha256'] = sha((OLD / 'local_reader_protocol.json').read_bytes())
protocol['revision'] = 'The terminal collector included top-level screen files but omitted detector_preflight_v1/report.json. Bind the already captured real AMP report byte-exactly as an external sidecar. Native PR, exact548 replay, zero tolerance,12 epochs and frozen remote code are unchanged.'
protocol['notes'].append('This is a pretrained-Stem P2 experiment. It is not evidence of Stem improvement relative to a random-Stem baseline.')
dump(NEW / 'local_reader_protocol.json', protocol)
dump(EVIDENCE / 'source_review.json', {'status': 'PREPARED_EXTERNAL_PREFLIGHT_EVIDENCE_BINDING_WITH_UNCHANGED_ENDPOINT_AND_REPLAY_CHECKS', 'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'previous_reader_sha256': sha(source), 'new_reader_sha256': protocol['reader_files_sha256']['verify_terminal_local.py'], 'preflight_sidecars': protocol['preflight_sidecars'], 'remote_code_or_archives_changed': False, 'new_model_forwards': 0, 'training_epochs': 0})
print(json.dumps({'status': 'PREPARED_V2', 'reader': str(NEW), 'sidecars': protocol['preflight_sidecars']}, ensure_ascii=False))
