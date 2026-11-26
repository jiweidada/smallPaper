"""Launch exactly one already authorized random control with frozen formal code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def live(record, pid_key, ticks_key):
    p = Path('/proc') / str(record[pid_key]) / 'stat'
    if not p.exists():
        return False
    fields = p.read_text().split(') ', 1)[1].split()
    return fields[19] == str(record[ticks_key]) and fields[0] not in ('Z', 'X')


def main():
    p = argparse.ArgumentParser()
    for n in ('prefix', 'python', 'promotion-sha'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--seed', type=int, choices=(1, 2), required=True)
    args = p.parse_args()
    root = Path(args.prefix) / 'DEIM/experiments/scale_protection12_20261010_v1'
    promotion_path = root / 'dual_seed_native_promotion.json'
    assert sha(promotion_path) == args.promotion_sha
    promotion = read(promotion_path)
    assert promotion['both_target_protect_fixed12_native_gates_passed'] is True
    assert promotion['maximum_total_epochs_per_configuration'] == 12
    assert promotion['experiment_protocol_sha256'] == sha(root / 'protocol.json')
    assert promotion['training_protocol_sha256'] == sha(root / 'code_v2/training_protocol.json')
    target = root / f'seed{args.seed}_target_protect'
    proof = promotion['seeds'][str(args.seed)]
    assert proof['native_gate_passed'] and proof['fixed_epoch'] == 12
    assert proof['delta_AP_metrics_pp']['APs'] > 0 and proof['delta_AP_metrics_pp']['AP'] >= 0
    for name, key in (
        ('run/training_completion.json', 'training_completion_sha256'),
        ('run/fixed12_comparison.json', 'comparison_sha256'),
        ('run/endpoint_epoch12/summary.json', 'endpoint_summary_sha256'),
        ('candidate_reload_v1/verification.json', 'reload_verification_sha256'),
        ('run/last.pth', 'candidate_checkpoint_sha256')):
        assert sha(target / name) == proof[key], name
    assert read(target / 'bundle/receipt.json')['identity'] == proof['identity']
    assert read(target / 'controller_completion.json')['status'] == 'COMPLETE_FIXED12'
    assert read(target / 'candidate_completion.json')['exit_code'] == 0
    assert read(target / 'candidate_reload_v1/completion.json')['exit_code'] == 0
    assert not live(read(target / 'candidate_start.json'), 'child_pid', 'child_start_ticks')
    assert not live(read(target / 'controller_start.json'), 'pid', 'start_ticks')
    screen = root / f'seed{args.seed}_random_protect'
    assert not screen.exists(), 'Existing configuration cannot be rerun'
    protocol = read(root / 'code_v2/training_protocol.json')
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == protocol['GPU_uuid'][str(args.seed)]
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    audit = root / f'random_control_launch_seed{args.seed}_v1'
    audit.mkdir(exist_ok=False)
    write(audit / 'start.json', dict(seed=args.seed, source_sha256=sha(__file__),
        promotion_sha256=args.promotion_sha, started_remote_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        baseline_training_epochs_added=0, baseline_model_forwards_added=0, maximum_total_epochs=12,
        automatic_retry=False, acceptance_gates_changed=False))
    try:
        prepare = [args.python, '-B', str(root / 'code_v2/prepare_training.py'),
                   '--prefix', args.prefix, '--seed', str(args.seed), '--arm', 'random_protect']
        with (audit / 'prepare.log').open('xb') as log:
            done = subprocess.run(prepare, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, cwd=root)
        write(audit / 'prepare_completion.json', dict(exit_code=done.returncode, command=prepare, automatic_retry=False))
        assert done.returncode == 0, 'Preparation failed; retain evidence and do not launch or retry'
        bundle = screen / 'bundle'
        receipt = read(bundle / 'receipt.json')
        assert receipt['identity']['seed'] == args.seed and receipt['identity']['arm'] == 'random_protect'
        assert receipt['identity']['implementation'] == proof['identity']['implementation']
        assert receipt['identity']['native_parameter_count'] == 10231997
        assert receipt['identity']['added_parameter_count'] == 0
        command = [args.python, '-B', str(root / 'code_v2/supervise.py'), '--bundle', str(bundle)]
        with (screen / 'controller.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, cwd=screen,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                         MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'))
        fields = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()
        assert fields[0] not in ('Z', 'X')
        record = dict(supervisor_pid=child.pid, supervisor_start_ticks=fields[19], command=command,
            bundle_receipt_sha256=sha(bundle / 'receipt.json'), promotion_sha256=args.promotion_sha,
            max_epochs_total=12, automatic_retry=False, baseline_epochs_added=0, baseline_forwards_added=0,
            formal_source_changed=False)
        write(screen / 'launch_receipt.json', record)
        write(audit / 'completion.json', dict(status='ONE_RANDOM_CONTROL_SUPERVISOR_LAUNCHED', **record))
        assert 'torch' not in sys.modules
        print(json.dumps(dict(status='ONE_RANDOM_CONTROL_SUPERVISOR_LAUNCHED', seed=args.seed, arm='random_protect',
            **record)), flush=True)
    except BaseException:
        write(audit / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
