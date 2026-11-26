"""Wait on each existing user experiment's exact process handles, collect its primary artifacts once."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, runpy, sys, time, traceback
from guards import ARMS, identity_plan, owners, positive_terminal

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, o):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(o, f, indent=2, allow_nan=False)

def state(record, pid='pid', ticks='start_ticks'):
    try:
        fields = Path('/proc/'+str(record[pid])+'/stat').read_text().split()
        return dict(record, same_live=fields[21] == str(record[ticks]) and fields[2] != 'Z', state=fields[2])
    except (FileNotFoundError, ProcessLookupError):
        return dict(record, same_live=False)

def main(prefix, seed, arm):
    assert seed in (1, 2) and prefix == {1: '/25085400120', 2: '/24085403014'}[seed] and arm in ARMS
    protocol = read(HERE/'protocol.json')
    assert {n: sha(HERE/n) for n in protocol['remote_sources']} == protocol['remote_sources']
    experiment = Path(prefix)/'DEIM/experiments/user_backbone12_20261011_v1'
    assert {p.name: sha(p) for p in (experiment/'code').iterdir() if p.is_file()} == protocol['user_files_sha256']
    screen = experiment/f'seed{seed}_{arm}'
    trainer, controller = read(screen/'candidate_start.json'), read(screen/'controller_start.json')
    frozen = protocol['existing_trainers'][str(seed)][arm]
    owners(trainer, controller, frozen)
    receipt_bundle, plan = read(screen/'bundle/receipt.json'), read(screen/'bundle/plan.json')
    identity = receipt_bundle['identity']
    assert identity['implementation'] == protocol['user_files_sha256']
    initialization = read(screen/'bundle/cpu_initialization.json')['initialization']
    identity_plan(identity, plan, initialization, arm, seed)
    root, tag = HERE.parent, f'seed{seed}_{arm}'
    assert not any((root/(tag+n)).exists() for n in ('_waiter_start.json', '_waiter_completion.json', '_waiter_failure.json', '_collect_start.json'))
    receipt = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        source_sha256=sha(__file__), protocol_sha256=sha(HERE/'protocol.json'), seed=seed, arm=arm,
        identity=identity, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_training_handle=state(trainer, 'child_pid', 'child_start_ticks'), actual_controller_handle=state(controller),
        automatic_retry=False, training_commands=0, new_model_forwards=0, new_evaluator_calls=0)
    write(root/(tag+'_waiter_start.json'), receipt)
    started = time.monotonic()
    try:
        while state(trainer, 'child_pid', 'child_start_ticks')['same_live'] or state(controller)['same_live']:
            temp = root/(tag+'_waiter_live.tmp')
            temp.write_text(json.dumps(dict(phase='waiting_exact_existing_user_trainer_and_controller',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                actual_trainer=state(trainer, 'child_pid', 'child_start_ticks'), actual_controller=state(controller))))
            temp.replace(root/(tag+'_waiter_live.json'))
            if time.monotonic()-started >= protocol['max_wait_seconds']:
                raise TimeoutError('Observation timed out: no signal, restart or automatic collection retry')
            time.sleep(30)
        positive_terminal(identity, read(Path(receipt_bundle['paths']['output'])/'training_completion.json'),
                          read(screen/'controller_completion.json'), read(screen/'candidate_completion.json'))
        sys.argv = [str(HERE/'collect.py'), '--prefix', prefix, '--seed', str(seed), '--arm', arm]
        runpy.run_path(str(HERE/'collect.py'), run_name='__main__')
        completion = read(root/(tag+'_collect_completion.json'))
        assert completion['exit_code'] == 0 and completion['receipt']['pid'] == receipt['pid']
        assert completion['receipt']['start_ticks'] == receipt['start_ticks']
        assert 'torch' not in sys.modules and not any(n.startswith('pycocoeval') for n in sys.modules)
        write(root/(tag+'_waiter_completion.json'), dict(status='USER_PRIMARY12_COLLECTED_ONCE_AFTER_ACTUAL_HANDLES_STOPPED',
            receipt=receipt, collection_completion_sha256=sha(root/(tag+'_collect_completion.json')),
            metadata=completion['metadata'], exit_code=0, new_model_forwards=0, new_evaluator_calls=0,
            new_training_epochs=0, existing_process_signals=0, automatic_retry=False,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except BaseException:
        write(root/(tag+'_waiter_failure.json'), dict(status='READ_ONLY_WAIT_OR_COLLECTION_FAILED_NO_RETRY',
            receipt=receipt, traceback=traceback.format_exc(), new_model_forwards=0, new_training_epochs=0, existing_process_signals=0))
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', required=True)
    p.add_argument('--seed', type=int, choices=(1, 2), required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    args = p.parse_args()
    main(args.prefix, args.seed, args.arm)
