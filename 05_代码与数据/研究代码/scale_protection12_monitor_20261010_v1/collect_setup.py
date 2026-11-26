"""Snapshot immutable completed setup evidence, not mutable training results."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--seed', type=int, choices=(1, 2), required=True)
    a = p.parse_args()
    root = Path(a.root); seed = a.seed
    screen = root / f'seed{seed}_target_protect'
    read = lambda q: json.loads(q.read_text())
    receipt = read(screen/'bundle/receipt.json'); identity = receipt['identity']
    assert identity['arm'] == 'target_protect' and identity['seed'] == seed
    package = read(root/'training_deployment_manifest_v2.json')
    assert sha_bytes((root/'training_package_v2.zip').read_bytes()) == 'f16681c2f862cfb19ea9cd49a8d663331fb56debca4e24808f050b30cda0e5b2'
    for name, item in package['files'].items():
        b = (root/name).read_bytes()
        assert len(b) == item['bytes'] and sha_bytes(b) == item['sha256'], name
    assert sha_bytes((screen/'bundle/plan.json').read_bytes()) == identity['plan_sha256']
    assert sha_bytes((screen/'bundle/resolved.yml').read_bytes()) == receipt['resolved_sha256']
    images = root/f'image_verification_seed{seed}.json'
    assert sha_bytes(images.read_bytes()) == identity['image_verification_sha256']
    assert read(images)['files_verified'] == 7019
    assert sha_bytes((root/f'image_manifest_seed{seed}.json').read_bytes()) == identity['image_manifest_sha256']
    for name, digest in identity['implementation'].items():
        assert sha_bytes((root/'code_v2'/name).read_bytes()) == digest, name
    native100 = Path(receipt['paths']['baseline_first100'])
    assert sha_bytes(native100.read_bytes()) == identity['baseline_first100_sha256']
    candidate100 = screen/'run/first100_training_inputs.jsonl'
    reference_rows = [json.loads(x) for x in native100.read_text().splitlines()]
    candidate_rows = [json.loads(x) for x in candidate100.read_text().splitlines()]
    assert len(candidate_rows) == len(reference_rows) == 100 and candidate_rows == reference_rows
    assert read(screen/'run/first100_parity.json')['all100_rows_exact'] is True
    start = read(screen/'run/runtime_start.json')
    assert start['identity'] == identity and start['initialization']['parameters'] == 10231997
    assert start['initialization']['extra_parameters'] == start['optimizer_coverage']['added_parameters'] == 0
    assert start['optimizer_coverage']['all_native_parameter_groups_unchanged']
    reuse = read(screen/'baseline_reuse.json')
    assert reuse['new_model_forwards'] == reuse['new_optimizer_steps'] == reuse['new_epochs'] == 0
    for tag in ('', '_v2'):
        x = read(root/f'prepare_seed{seed}{tag}_completion.json')
        assert x['exit_code'] == (1 if not tag else 0)
    # Only files no longer written after first100, plus immutable setup files.
    rel = ['collect_setup.py','protocol.json','training_deployment_manifest.json','training_deployment_manifest_v2.json',
           f'image_manifest_seed{seed}.json', f'image_verification_seed{seed}.json']
    rel += [f'prepare_seed{seed}{tag}{suffix}' for tag in ('', '_v2')
            for suffix in ('.log', '_completion.json')]
    rel += [str(q.relative_to(root)) for q in (root/'code_v2').iterdir() if q.is_file()]
    rel += [f'seed{seed}_target_protect/{name}' for name in
            ('bundle/receipt.json','bundle/plan.json','bundle/resolved.yml','launch_receipt.json',
             'controller_start.json','candidate_start.json','baseline_reuse.json',
             'run/runtime_start.json','run/first100_parity.json','run/first100_training_inputs.jsonl')]
    contents = {n:(root/n).read_bytes() for n in rel}
    contents['reference/baseline_first100_training_inputs.jsonl'] = native100.read_bytes()
    observed = dict(schema='scale_protection12_setup_packet_v1', seed=seed,
        observed_remote_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='FIRST100_AND_NATIVE_SETUP_VERIFIED_NOT_TERMINAL',
        identity=identity, first100_all_rows_exact=True, current_images_verified=7019,
        preparation_failure_before_model_or_training=True, new_baseline_forwards=0,
        new_baseline_optimizer_steps=0, new_baseline_epochs=0,
        candidate_training_in_progress=True, endpoint_AP_not_available=True,
        scope='Immutable setup and first100 input/pre-forward RNG proof only; not efficacy or full-trajectory equality.')
    candidate = read(screen/'candidate_start.json')
    for key, tickkey in (('child_pid','child_start_ticks'),):
        proc = Path(f"/proc/{candidate[key]}/stat")
        assert proc.exists() and proc.read_text().split()[21] == candidate[tickkey]
    contents['setup_evidence.json'] = json.dumps(observed,ensure_ascii=False,indent=2).encode()
    manifest = dict(files={n:dict(bytes=len(b),sha256=sha_bytes(b)) for n,b in contents.items()})
    archive = root/f'setup_seed{seed}_v2.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in contents.items(): z.writestr(n,b)
        z.writestr('setup_collection_manifest.json',json.dumps(manifest,indent=2).encode())
    print(json.dumps(dict(seed=seed,files=len(contents),archive_bytes=archive.stat().st_size,
                         archive_sha256=sha_bytes(archive.read_bytes()))),flush=True)


if __name__ == '__main__':
    main()
