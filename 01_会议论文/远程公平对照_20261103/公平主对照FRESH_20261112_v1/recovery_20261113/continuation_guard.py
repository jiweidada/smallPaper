"""Verify a one-time continuation against its staged evidence and authorization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def remaining_seconds(plan):
    return (datetime.fromisoformat(plan['deadline_remote_utc']) - datetime.now(timezone.utc)).total_seconds()


def verify_continuation(here, protocol_sha, authorization_sha):
    from backup_protocol import EpochGate, accepted_bytes, ack_bytes, parse_manifest

    authorization_path = here / 'continuation_authorization.json'
    require(sha(authorization_path) == authorization_sha, 'Continuation authorization differs')
    auth = read(authorization_path)
    plan_path = here / 'continuation_plan.json'
    require(auth['authorized'] is True and auth['plan_sha256'] == sha(plan_path),
            'No matching continuation authorization')
    plan = read(plan_path)
    expected = {'BASE_FRESH': (165, 1617), 'FULL_FRESH': (140, 0)}
    rounds, extra_attempts = expected[plan['arm']]
    require(plan['original_protocol_sha256'] == protocol_sha == sha(here / 'execution_protocol.json'),
            'Original protocol changed')
    require(here.name == plan['arm'] and here == Path(plan['continuation_root'])
            and here != Path(plan['original_root']), 'Unexpected continuation location')
    require(plan['accepted_rounds'] == rounds and plan['target_rounds'] == 320
            and plan['retained_attempts'] == rounds * 1617
            and plan['remaining_attempts'] == (320 - rounds) * 1617
            and plan['remaining_full_validations'] == 320 - rounds
            and plan['replayed_attempts_upper_bound'] == auth['replayed_attempts_upper_bound'] == extra_attempts,
            'Unexpected continuation budget')
    require(remaining_seconds(plan) > 0, 'Original hard deadline elapsed')
    for name, expected_sha in plan['continuation_code'].items():
        require(sha(here / name) == expected_sha, 'Continuation source differs: ' + name)
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['boot_id'],
            'Machine boot identity changed')
    gate = EpochGate(here / 'run')
    gate.require_probe()
    stage = gate.stage(rounds - 1)
    payload = (stage / 'manifest.json').read_bytes()
    manifest = parse_manifest(payload, here.name, rounds - 1)
    require(sha(stage / 'manifest.json') == plan['checkpoint_manifest_sha256']
            and (stage / 'ack.json').read_bytes() == ack_bytes(payload, manifest)
            and (stage / 'accepted.json').read_bytes() == accepted_bytes(payload, manifest)
            and manifest['rolling_sha256'] == plan['checkpoint_sha256']
            and manifest['best'] == plan['best'] and manifest['stage_one'] == plan['stage_one']
            and manifest['evaluated_state_is_next_state'], 'Accepted continuation boundary differs')
    require(not gate.stage(rounds).exists() and not (here / 'run/receipt.json').exists(),
            'Continuation already started or another round exists')
    for item in manifest['files']:
        path = item['source']
        if path.startswith('run/') and path != 'run/receipt.json':
            require(sha(here / path) == item['sha256'], 'Retained output changed: ' + path)
    return plan
