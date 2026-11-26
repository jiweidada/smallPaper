"""Serial local CPU execution of exactly the authorized eight fixed runs."""
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
C = importlib.import_module('348_common')


def preflight(plan):
    started = time.perf_counter()
    seals = [C.verify_seal(r['path']) for r in plan['old_seals']]
    index = C.read(plan['active_cache_manifest']['path'])
    for entry in index['entries']:
        C.check(Path(entry['path']).stat().st_size == entry['bytes'], 'cache bytes')
        C.verify_record(entry['path'], entry['file_sha256'])
    C.write(C.HERE / '348_开训前封存与缓存核验.json',
            {'status': '已验证', 'utc': C.now(), 'old_seals': seals,
             'active_cache_files': len(index['entries']),
             'active_cache_bytes': sum(e['bytes'] for e in index['entries']),
             'head_forwards': 0, 'optimizer_steps': 0,
             'seconds': time.perf_counter() - started})
    C.emit({'event': 'preflight_passed', 'cache_files': len(index['entries']),
            'sealed_files': sum(r['verified_files'] for r in seals)})


def last_progress(path):
    if not path.exists() or not path.stat().st_size:
        return None
    with open(path, 'rb') as f:
        f.seek(max(0, path.stat().st_size - 65536))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def latest_checkpoint(logical):
    valid = []
    for path in logical.glob('attempt_*/347_checkpoint_step*_receipt.json'):
        receipt = C.read(path)
        C.verify_record(receipt['checkpoint_path'], receipt['checkpoint_sha256'])
        C.check(receipt['binding']['mode'] == 'formal', 'formal resume only')
        valid.append((receipt['step'], str(path), {'path': str(path), 'sha256': C.sha(path)}))
    C.check(valid, 'No complete checkpoint; refusing a step0 restart')
    return max(valid)[2]


def verify_completed(run_dir, identity):
    # This verifier never calls a model or loss, and does no evaluation.
    result = C.read(run_dir / '347_run_result.json')
    rows = C.read(run_dir / '347_steps.json')
    binding = C.read(run_dir / '347_run_binding.json')
    config = binding['inherited_recipe']
    arm, seed = config['arm'], config['global_seed']
    C.check(identity == config['run_id'], 'run identity')
    C.check(result['full_budget_completed'] is True and result['step'] == 1218, 'completed budget')
    C.check((result['epoch'], result['batch_index'], result['position_in_epoch']) == (6, 0, 0), 'final cursor')
    C.check(len(rows) == 1218 and binding['max_steps'] == 1218, '1218 numeric records')
    order = C.read(result['binding']['shuffle_path'])['epochs']
    curve = C.read(config['lr_non_kappa']['curve'])
    exposures = 0
    last_batch_sizes = []
    for step, row in enumerate(rows):
        epoch, b = divmod(step, 203)
        ids = order[epoch][16 * b:16 * (b + 1)]
        C.check((row['step'], row['epoch'], row['batch_index'], row['position_in_epoch']) ==
                (step, epoch, b, 16 * b), 'step cursor')
        C.check(row['image_ids'] == ids, 'materialized order mismatch')
        exposures += len(ids)
        if b == 202:
            last_batch_sizes.append(len(ids))
        expected_lr = [curve['non_kappa'][step]] * (3 if arm == 'G' else 2)
        if arm == 'G':
            expected_lr[2] = curve['G_kappa'][step]
        C.check(row['param_group_lr'] == expected_lr, 'sealed LR curve mismatch')
        for key in ('loss_det', 'loss_mass', 'loss_assoc', 'loss_total', 'grad_norm_before_clip'):
            C.check(math.isfinite(row[key]), 'nonfinite numeric log')
    C.check(exposures == 19410 and last_batch_sizes == [3] * 6, 'epoch exposure count')
    first2 = C.read(C.S345 / '345_B_CPU_acceptance_runs' /
                    f'345_{identity}_formal_order_first2' / '345_steps.json')
    C.check(rows[:2] == first2, '345 first2 numeric mismatch')
    reference10 = None
    if identity == 'G_seed1':
        reference10 = C.read(C.S347 / '347_subject_G_seed1_10' / '347_steps.json')
        C.check(rows[:10] == reference10, '347 G_seed1 first10 numeric mismatch')
    receipt_reference = result['last_checkpoint']
    C.verify_record(receipt_reference['path'], receipt_reference['sha256'])
    receipt = C.read(receipt_reference['path'])
    C.verify_record(receipt['checkpoint_path'], receipt['checkpoint_sha256'])
    C.check(receipt['step'] == 1218 and receipt['binding'] == result['binding'], 'final checkpoint receipt')
    # Import CPU dependencies only after the worker exits; never concurrent training.
    sys.path.insert(0, str(C.S347))
    F = importlib.import_module('train347')
    checkpoint = F.torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    C.check(checkpoint['step_logs'] == rows and checkpoint['step'] == 1218, 'checkpoint numeric logs')
    C.check(F.state_sha(checkpoint['model']) == result['final_model_state_sha256'] ==
            checkpoint['model_state_sha256'], 'final model tensor SHA')
    C.check(all(v.device.type == 'cpu' and (not v.is_floating_point() or v.dtype == F.torch.float32)
                for v in checkpoint['model'].values()), 'final CPU FP32')
    C.check(result['counts_this_process']['global_forwards'] == 0, 'global forward count')
    return {'status': '已验证', 'run_id': identity, 'run_dir': str(run_dir),
            'full_budget_completed': True, 'step': 1218, 'epochs': 6,
            'crop_exposures': exposures, 'last_batch_sizes': last_batch_sizes,
            'all_losses_and_grad_norm_finite': True, 'formal_order_1218_steps': True,
            'sealed_lr_curve_1218_steps': True, 'first2_equal_345': True,
            'G_seed1_first10_equal_347': True if reference10 is not None else '不适用',
            'final_checkpoint': receipt_reference, 'checkpoint_path': receipt['checkpoint_path'],
            'final_checkpoint_sha256': receipt['checkpoint_sha256'],
            'final_model_state_sha256': result['final_model_state_sha256'],
            'counts_final_process': result['counts_this_process'], 'verified_utc': C.now()}


def run_one(plan, identity):
    logical = C.HERE / '348_runs' / identity
    verification_path = logical / '348_run_verified.json'
    run_seal = logical / '348_run_封存SHA256.json'
    if verification_path.exists():
        C.check(run_seal.exists(), 'verification exists without completed run seal')
        C.verify_seal(run_seal)
        return C.read(verification_path)
    logical.mkdir(parents=True, exist_ok=True)
    arm, seed_text = identity.split('_seed')
    prior = sorted(logical.glob('348_attempt_*_job.json'))
    resume = None if not prior else latest_checkpoint(logical)
    while True:
        C.check(not (C.HERE / '348_pause_after_checkpoint.request').exists(), 'explicit pause requested')
        index = len(list(logical.glob('348_attempt_*_job.json')))
        prefix = f'348_attempt_{index:03d}'
        run_dir = logical / f'attempt_{index:03d}'
        job_path = logical / (prefix + '_job.json')
        job = {'arm': arm, 'seed': int(seed_text), 'device': 'cpu', 'max_steps': 1218,
               'logical_dir': str(logical), 'run_dir': str(run_dir), 'resume': resume,
               'progress_path': str(logical / (prefix + '_progress.jsonl')),
               'outcome_path': str(logical / (prefix + '_outcome.json')), 'created_utc': C.now()}
        C.write(job_path, job)
        C.emit({'event': 'run_start', 'run_id': identity, 'attempt': index, 'resume': resume})
        with open(logical / (prefix + '_stdout.txt'), 'xb') as stdout, \
                open(logical / (prefix + '_stderr.txt'), 'xb') as stderr:
            child = subprocess.Popen([sys.executable, '-B', str(C.HERE / '348_worker.py'), str(job_path)],
                                     cwd=str(C.WORK), env=C.cpu_env(), stdout=stdout, stderr=stderr)
            C.write(logical / (prefix + '_process.json'),
                    {'pid': child.pid, 'started_utc': C.now(), 'job_path': str(job_path)})
            last_announcement = 0.0
            while child.poll() is None:
                if time.monotonic() - last_announcement >= 30:
                    progress = last_progress(Path(job['progress_path']))
                    C.emit({'event': 'progress', 'run_id': identity, 'pid': child.pid,
                            'completed_steps': None if progress is None else progress.get('completed_steps'),
                            'last_event_utc': None if progress is None else progress['utc']})
                    last_announcement = time.monotonic()
                time.sleep(5)
        C.write(logical / (prefix + '_exit.json'), {'returncode': child.returncode, 'utc': C.now()})
        outcome_path = Path(job['outcome_path'])
        outcome = C.read(outcome_path) if outcome_path.exists() else None
        if child.returncode == 0 and outcome and outcome['status'] == 'completed':
            verified = verify_completed(run_dir, identity)
            C.write(verification_path, verified)
            seal = C.seal_tree(logical, run_seal.name)
            C.emit({'event': 'run_sealed', 'run_id': identity, 'step': 1218, 'seal': seal})
            return verified
        if outcome is not None:
            raise RuntimeError('训练中止；保留断点，状态=' + outcome['status'] + ': ' + str(outcome.get('error')))
        # An externally killed process has no outcome. Resume its nearest intact
        # 347 checkpoint once; a repeated interruption requires investigation.
        C.check(index == 0, 'repeated external interruption; checkpoint preserved')
        resume = latest_checkpoint(logical)
        C.emit({'event': 'external_interruption_resume', 'run_id': identity, 'receipt': resume})


def finish(plan, runs, start_wall):
    comparisons = []
    for seed in (1, 2):
        group = [r for r in runs if r['run_id'].endswith(f'_seed{seed}')]
        batches = []
        for r in group:
            rows = C.read(Path(r['run_dir']) / '347_steps.json')
            batches.append([(v['image_ids'], v['batch_tensor_sha256']) for v in rows])
        C.check(len(group) == 4 and all(b == batches[0] for b in batches), 'four-arm batch inequality')
        comparisons.append({'seed': seed, 'steps_per_arm': 1218, 'four_arms_batch_ids_and_SHA_identical': True})
    ending_seals = [C.verify_seal(r['path']) for r in plan['old_seals']]
    # Cache tensors were SHA checked on every training load; also rescan the
    # previously approved development+val inventory without running evaluation.
    cache_index = C.read(plan['active_cache_manifest']['path'])
    for e in cache_index['entries']:
        C.verify_record(e['path'], e['file_sha256'])
    document = {'status': '八个本机CPU固定预算run已完成并封存；待主控核对',
                'started_utc': plan['authorized_utc'], 'completed_utc': C.now(),
                'runs': runs, 'same_seed_four_arm_batches': comparisons,
                'completed_runs': len(runs), 'effective_optimizer_steps': sum(r['step'] for r in runs),
                'crop_exposures': sum(r['crop_exposures'] for r in runs),
                'device': 'cpu', 'max_parallel_training_processes': 1,
                'old_seals_end': ending_seals, 'active_cache_SHA_verified_end': len(cache_index['entries']),
                'wall_seconds_this_controller_process': time.perf_counter() - start_wall,
                'no_best_selection': True, 'extra_epochs': 0, 'recipe_changes': 0,
                'val548_evaluation': '未执行', 'G_delta': '未执行',
                'bootstrap': '未执行', 'twelve_item_decision': '未执行',
                'remote_SSH': '未执行', 'GPU': '未执行'}
    C.write(C.HERE / '348_训练_result.json', document)
    lines = ['# 348 本机 CPU 八 run 训练交接', '',
             '八个 run 已按封存的 train347.py 完成。每个 1218 步、6 轮，固定末轮权重；未选 best。', '',
             '| run | 步数 | 完整预算 | 末模型 SHA256 |', '|---|---:|---|---|']
    lines += [f"| {r['run_id']} | {r['step']} | true | {r['final_model_state_sha256']} |" for r in runs]
    lines += ['', '合计 9744 个有效优化器步骤、155280 次窗口曝光。同 seed 四臂的 1218 组图 ID 与 batch SHA 全部一致；损失和梯度范数全程有限。',
              '', '每个 run 的目录、末断点路径、SHA、验证记录见 348_训练_result.json 和 348_runs 下的独立封存。',
              '347/346/345 封存及 7566 条活动缓存在结束时再次核验。训练配方、几何、标签、阈值与窗均未修改。',
              '', '已停止，待主控独立核对。val548 评估、G Δ、bootstrap、12 项判定均未执行；未访问远程服务器，未使用 GPU。',
              '这些训练完成记录不作科学增益判断。', '']
    C.text(C.HERE / '348_一页中文汇总.md', '\n'.join(lines))
    seal = C.seal_tree(C.HERE, '348_交付封存SHA256.json')
    C.emit({'event': 'all_eight_completed_and_sealed', 'completed_runs': len(runs),
            'effective_optimizer_steps': document['effective_optimizer_steps'], 'seal': seal})


def main():
    start_wall = time.perf_counter()
    plan = C.plan()
    with C.lock(C.HERE / '348_controller.lock'):
        preflight_path = C.HERE / '348_开训前封存与缓存核验.json'
        if not preflight_path.exists():
            preflight(plan)
        runs = [run_one(plan, identity) for identity in plan['run_order']]
        finish(plan, runs, start_wall)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        path = C.HERE / ('348_实现失败_' + str(time.time_ns()) + '.json')
        C.write(path, {'status': '实现失败', 'utc': C.now(), 'type': type(exc).__name__,
                       'error': str(exc), 'traceback': traceback.format_exc(),
                       'training_checkpoints_preserved': True})
        raise
