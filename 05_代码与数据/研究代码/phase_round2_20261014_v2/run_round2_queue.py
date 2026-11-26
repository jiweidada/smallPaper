"""One bounded, append-only six-job queue per account. No automatic retries."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

ADDONS = pathlib.Path(__file__).resolve().parent
ACCEPTANCE_STAGE = ADDONS.parent
STAGE = ACCEPTANCE_STAGE/'execution_v2'
PHASE = ACCEPTANCE_STAGE.parent.parent
CODE = PHASE/'code'
PLAN = json.loads((ADDONS/'queue_plan.json').read_text())
STATE = STAGE/'queue_state.json'
ENV = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
STATE_DATA = {'account': PLAN['account'], 'queue_pid': os.getpid(), 'start': time.time(),
              'planned_jobs': PLAN['jobs'], 'completed_jobs': [], 'state': 'starting'}


def state(name, **fields):
    STATE_DATA.update(state=name, observed_at=time.time(), **fields)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(STATE_DATA, indent=2))
    os.replace(temporary, STATE)
    with (STAGE/'queue_events.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'state': name, **fields})+'\n')
    print(json.dumps({'time': time.time(), 'state': name, **fields}), flush=True)


def gpu_pids():
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [line.strip() for line in query.stdout.splitlines() if line.strip().isdigit()]


def run(command, log, label, cwd=CODE):
    log = pathlib.Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb') as output:
        process = subprocess.Popen(command, cwd=cwd, env=ENV, stdout=output, stderr=subprocess.STDOUT)
        state(label, child_pid=process.pid, command=command, log=str(log))
        while process.poll() is None:
            # Ongoing bounded job monitoring; no restart, configuration changes, or notifications.
            STATE_DATA['heartbeat'] = time.time()
            temporary = STATE.with_suffix('.tmp')
            temporary.write_text(json.dumps(STATE_DATA, indent=2)); os.replace(temporary, STATE)
            time.sleep(30)
        returncode = process.returncode
    if returncode:
        raise RuntimeError(label+' failed, returncode='+str(returncode)+'; see '+str(log))


def wait_for_existing():
    while True:
        jobs = [json.loads((PHASE/'jobs'/(name+'.json')).read_text()) for name in PLAN['existing_jobs']]
        failed = [job for job in jobs if job.get('state') == 'failed' or job.get('returncode', 0) != 0]
        if failed:
            raise RuntimeError('Existing experiment failed; refusing to continue its queue')
        if all(job.get('state') == 'complete' and job.get('returncode') == 0 for job in jobs) and not gpu_pids():
            return
        state('waiting_existing_experiments', old_jobs={job['job']: job['state'] for job in jobs})
        time.sleep(60)


def verify_training(job_id):
    output = PHASE/'outputs'/job_id
    rows = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
    rows = [row for row in rows if 'test_coco_eval_bbox' in row]
    if sorted(row['epoch'] for row in rows) != list(range(96)):
        raise RuntimeError('Expected exactly one complete validation row for each of 96 epochs')
    checkpoint = output/'best_stg2.pth'
    if not checkpoint.is_file():
        raise RuntimeError('Missing required best_stg2 EMA checkpoint')
    finite_metrics = all(len(row['test_coco_eval_bbox']) == 12 and all(__import__('math').isfinite(x) for x in row['test_coco_eval_bbox']) for row in rows)
    if not finite_metrics:
        raise RuntimeError('Training evaluation contains non-finite or incomplete metrics')
    with (output/'round2_train_acceptance.json').open('x') as f:
        json.dump({'job': job_id, 'epochs': 96, 'checkpoint': str(checkpoint),
                   'all_twelve_metrics_finite': finite_metrics}, f, indent=2)
    return checkpoint


def export_model(name, config, checkpoint, soep=False, cli_test=False):
    root = PHASE/'round2_results'/name
    root.mkdir(parents=True, exist_ok=False)
    for split, overrides in [('val', []), ('testdev', ['--images', PLAN['testdev_images'], '--annotations', PLAN['testdev_annotations']])]:
        if cli_test:
            entry = CODE/('train_round2.py' if soep else 'train.py')
            command = [sys.executable, '-u', str(entry), '-c', str(config), '-r', str(checkpoint),
                       '--test-only', '--seed', '0', '--output-dir', str(root/(split+'_test_only')),
                       '-u', 'score_threshold=0.0', 'PostProcessor.num_top_queries=300',
                       'val_dataloader.num_workers=2', 'val_dataloader.total_batch_size=4']
            if split == 'testdev':
                command += ['val_dataloader.dataset.img_folder='+PLAN['testdev_images'],
                            'val_dataloader.dataset.ann_file='+PLAN['testdev_annotations']]
            run(command, root/(split+'_test_only.log'), name+'_'+split+'_test_only')
        command = [sys.executable, '-u', str(ADDONS/'dump_preds.py'), '--config', str(config),
                   '--checkpoint', str(checkpoint), '--out', str(root/split), '--project-eval'] + overrides
        if soep:
            command += ['--register-soep']
        run(command, root/(split+'_export.log'), name+'_'+split+'_independent_export')
    return root


def analyze_pair(reference, candidate, label):
    run([sys.executable, str(ADDONS/'analyze_preds.py'), '--annotations', PLAN['val_annotations'],
         '--reference', str(reference/'val/predictions.json'), '--candidate', str(candidate/'val/predictions.json'),
         '--out', str(PHASE/'round2_analysis'/label)], PHASE/'round2_analysis'/(label+'.log'), label+'_size_localization')


def main():
    with (STAGE/'queue_execution.lock').open('x') as f:
        f.write(str(os.getpid()))
    if hashlib.sha256((PHASE/'artifacts/common_init_seed0.pth').read_bytes()).hexdigest() != PLAN['common_init_sha256']:
        raise RuntimeError('Common initialization identity changed')
    if not PLAN['effective_recipe_audit_accepted']:
        raise RuntimeError('Recipe audit not accepted')
    wait_for_existing()
    activation = ACCEPTANCE_STAGE/'activation_receipt.json'
    if not activation.exists():
        run([sys.executable, str(ADDONS/'activate_remote.py')], STAGE/'queue_activation.log', 'activating_after_existing_training')
    if json.loads(activation.read_text())['status'] != 'activated_gpu_verified':
        raise RuntimeError('Activation or five-arm GPU preflight failed')
    # SOEP-lite is fully specified and technically reviewed before the queue reaches it.
    run([sys.executable, str(ADDONS/'test_prediction_analysis.py')], STAGE/'analysis_tests.log', 'cpu_analysis_acceptance')
    run([sys.executable, str(ADDONS/'test_soep_lite.py')], STAGE/'soep_tests.log', 'soep_structure_acceptance')
    if not (ADDONS/'soep_cost_audit.json').exists():
        run([sys.executable, str(ADDONS/'benchmark_soep_lite.py')], STAGE/'soep_cost_audit.log', 'soep_cost_acceptance')
    cost = json.loads((ADDONS/'soep_cost_audit.json').read_text())
    assert len(cost['results']) == 3 and cost['results'][-1]['branch_parameters'] == 15424
    exports = {}
    existing_done = False
    for spec in PLAN['jobs']:
        arm, seed = spec.split(':'); job_id = arm+'_seed'+seed
        state('preparing_job', active_job=job_id)
        if gpu_pids():
            raise RuntimeError('Unexpected GPU process appeared before a new training job')
        config = CODE/'configs/experiments/phase'/(arm+'.yml')
        output = PHASE/'outputs'/job_id
        job_state = PHASE/'jobs'/(job_id+'.json')
        if output.exists() or job_state.exists() or (PHASE/'logs'/(job_id+'.log')).exists():
            raise RuntimeError('Refusing to overwrite existing job artifacts: '+job_id)
        with job_state.open('x') as f:
            json.dump({'job': job_id, 'state': 'running', 'start': time.time(), 'round2_queue_pid': os.getpid()}, f)
        soep = arm == 'soep_lite_full96'
        entry = CODE/('train_round2.py' if soep else 'train.py')
        command = [sys.executable, '-u', str(entry), '-c', str(config), '--seed', seed, '--use-amp',
                   '-t', str(PHASE/'artifacts/common_init_seed0.pth'), '--output-dir', str(output)]
        try:
            run(command, PHASE/'logs'/(job_id+'.log'), 'training_'+job_id)
            checkpoint = verify_training(job_id)
            current = json.loads(job_state.read_text()); current.update(state='trained_pending_independent_eval', returncode=0, train_end=time.time())
            job_state.write_text(json.dumps(current))
            exports[job_id] = export_model(job_id, config, checkpoint, soep=soep, cli_test=True)
            if not existing_done and int(seed) in [0, 1]:
                for model in PLAN['existing_models']:
                    exports[model['name']] = export_model(model['name'], CODE/model['config'], pathlib.Path(model['checkpoint']), cli_test=True)
                for other in ['baseline', 'lowpass', 'singlephase']:
                    analyze_pair(exports['existing_'+other], exports['existing_c'], 'existing_C_minus_'+other+'_seed'+str(PLAN['primary_seed']))
                existing_done = True
                if PLAN['account'] == 'A':
                    for label in ['baseline', 'c', 'lowpass']:
                        model = next(row for row in PLAN['existing_models'] if row['name'] == 'existing_'+label)
                        run([sys.executable, '-u', str(ADDONS/'run_translation_suite.py'), '--config', str(CODE/model['config']),
                             '--checkpoint', model['checkpoint'], '--annotations', PLAN['val_annotations'],
                             '--zero-predictions', str(exports['existing_'+label]/'val/predictions.json'),
                             '--out', str(PHASE/'round2_translation'/label)], PHASE/'round2_translation'/(label+'.log'), 'translation_'+label)
            if arm == 'c_mean_full96':
                reference = exports['c_full96_seed'+seed] if int(seed) == 2 else exports['existing_c']
                analyze_pair(exports[job_id], reference, 'C_minus_Cmean_seed'+seed)
            elif arm == 'rank16_mean_full96':
                reference = exports['rank16_full96_seed'+seed]
                analyze_pair(exports[job_id], reference, 'rank16_minus_rank16mean_seed'+seed)
            elif soep:
                analyze_pair(exports[job_id], exports['existing_c'], 'C_minus_SOEPlite_seed'+seed)
            current.update(state='complete', end=time.time(), independent_evaluation_complete=True)
            job_state.write_text(json.dumps(current))
            STATE_DATA['completed_jobs'].append(job_id)
            state('job_complete', active_job=job_id)
        except BaseException as exc:
            current = json.loads(job_state.read_text()); current.update(state='failed', error=type(exc).__name__+': '+str(exc), end=time.time())
            job_state.write_text(json.dumps(current))
            raise
    state('complete', active_job=None)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        state('failed', error=type(exc).__name__+': '+str(exc))
        raise
