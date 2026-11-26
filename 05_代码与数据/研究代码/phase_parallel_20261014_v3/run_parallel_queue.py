"""Two training slots, including live adopted/legacy jobs; no training retries.

All independent evaluation and analysis runs after training, with an idle GPU.
The old A parent is held while its child runs, preserving the child's exit code.
"""
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time


def identity(pid):
    path = pathlib.Path('/proc') / str(pid)
    try:
        stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'ppid': int(stat[1]), 'state': stat[0], 'start_ticks': int(stat[19]),
                'exit_status': int(stat[49]),
                'argv': [x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x]}
    except FileNotFoundError:
        return None


def same_process(expected):
    actual = identity(expected['pid'])
    if actual is None or actual['start_ticks'] != expected['start_ticks']:
        return None
    return actual


def signal_exact(expected, value):
    actual = same_process(expected)
    if actual is None:
        raise RuntimeError('Process identity changed: '+str(expected['pid']))
    os.kill(actual['pid'], value)


def slots_available(active_count, external_count):
    return max(0, 2-active_count-external_count)


def adopted_returncode(expected):
    actual = same_process(expected)
    if actual is None:
        raise RuntimeError('Adopted trainer disappeared without an observable exit status')
    return os.waitstatus_to_exitcode(actual['exit_status']) if actual['state'] == 'Z' else None


def main_training_processes(phase):
    found = {}
    for path in pathlib.Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = identity(int(path.name))
        except (OSError, ValueError, IndexError):
            continue
        if row and row['state'] != 'Z' and '--test-only' not in row['argv']:
            names = {pathlib.PurePosixPath(arg).name for arg in row['argv']}
            if names & {'train.py', 'train_round2.py'} and any(str(phase/'outputs') in arg for arg in row['argv']):
                found[row['pid']] = row
    return [row for row in found.values() if row['ppid'] not in found]


class Queue:
    def __init__(self, stage):
        self.stage = pathlib.Path(stage).resolve()
        self.phase = self.stage.parent.parent.parent
        self.code = self.stage/'code'
        self.tools = self.stage/'tools'
        self.plan = json.loads((self.stage/'queue_plan.json').read_text())
        self.state_path = self.stage/'queue_state.json'
        self.env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        self.data = {'account': self.plan['account'], 'queue_pid': os.getpid(), 'started': time.time(),
                     'planned_jobs': self.plan['jobs'], 'training_completed_jobs': [], 'completed_jobs': [],
                     'maximum_concurrent_trainings': 2, 'active_jobs': {}, 'state': 'ready_for_handoff',
                     'postprocessing_policy': 'after_all_trainings_on_idle_gpu'}
        self.active = {}
        self.failed = False
        self.handoff = None
        self.exports = {}

    def state(self, name=None, **fields):
        if name:
            self.data['state'] = name
        self.data.update(fields, heartbeat=time.time(), observed_at=time.time())
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, indent=2))
        os.replace(temporary, self.state_path)
        if name:
            with (self.stage/'queue_events.jsonl').open('a') as output:
                output.write(json.dumps({'time': time.time(), 'state': name, **fields})+'\n')
            print(json.dumps({'state': name, **fields}), flush=True)

    def write_job(self, job, **fields):
        path = self.phase/'jobs'/(job+'.json')
        data = json.loads(path.read_text()) if path.exists() else {'job': job}
        data.update(fields)
        temporary = path.with_suffix('.parallel_tmp')
        temporary.write_text(json.dumps(data, indent=2))
        os.replace(temporary, path)

    def external(self):
        rows = {}
        for job in self.plan['existing_jobs']:
            path = self.phase/'jobs'/(job+'.json')
            data = json.loads(path.read_text())
            rows[job] = data
            if data.get('state') == 'failed' or data.get('returncode', 0) != 0:
                self.failed = True
        self.data['old_jobs'] = {job: row.get('state') for job, row in rows.items()}
        return [job for job, row in rows.items() if row.get('state') not in ['complete', 'failed']]

    def verify(self, job):
        output = self.phase/'outputs'/job
        rows = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
        rows = [row for row in rows if 'test_coco_eval_bbox' in row]
        if sorted(row['epoch'] for row in rows) != list(range(96)):
            raise RuntimeError('Training did not produce exactly epochs 0..95: '+job)
        if not all(len(row['test_coco_eval_bbox']) == 12 and all(math.isfinite(x) for x in row['test_coco_eval_bbox']) for row in rows):
            raise RuntimeError('Incomplete/nonfinite metrics: '+job)
        if not (output/'best_stg2.pth').is_file():
            raise RuntimeError('Missing best_stg2: '+job)
        with (output/'round2_train_acceptance.json').open('x') as handle:
            json.dump({'job': job, 'epochs': 96, 'checkpoint': str(output/'best_stg2.pth'),
                       'all_twelve_metrics_finite': True, 'queue_version': 'parallel_v3'}, handle, indent=2)

    def start(self, spec):
        arm, seed = spec.split(':')
        job = arm+'_seed'+seed
        output = self.phase/'outputs'/job
        log = self.phase/'logs'/(job+'.log')
        path = self.phase/'jobs'/(job+'.json')
        if output.exists() or log.exists() or path.exists():
            raise RuntimeError('Refusing duplicate training/artifact overwrite: '+job)
        if len(main_training_processes(self.phase)) >= 2:
            return False
        query = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
        if int(query.stdout.strip().splitlines()[0]) < 10240:
            return False
        command = [sys.executable, '-u', str(self.code/('train_round2.py' if arm == 'soep_lite_full96' else 'train.py')),
                   '-c', str(self.code/'configs/experiments/phase'/(arm+'.yml')), '--seed', seed, '--use-amp',
                   '-t', str(self.phase/'artifacts/common_init_seed0.pth'), '--output-dir', str(output)]
        with path.open('x') as handle:
            json.dump({'job': job, 'state': 'running', 'start': time.time(), 'parallel_queue_pid': os.getpid()}, handle)
        with log.open('xb') as handle:
            process = subprocess.Popen(command, cwd=self.code, env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT)
        actual = identity(process.pid)
        if actual is None:
            raise RuntimeError('Trainer exited before identity capture: '+job)
        self.active[job] = {'process': process, 'identity': actual}
        self.data['active_jobs'][job] = {'pid': process.pid, 'start_ticks': actual['start_ticks'],
                                        'command': command, 'log': str(log), 'adopted': False}
        self.state('parallel_training', newly_started_job=job)
        return True

    def finish_training(self, job, returncode):
        info = self.active[job]
        if info.get('adopted'):
            with (self.stage/'adopted_training_exit.json').open('x') as handle:
                json.dump({'trainer': info['identity'], 'returncode': returncode,
                           'observed_zombie': identity(info['identity']['pid']), 'time': time.time()}, handle, indent=2)
            parent = self.handoff['old_controller']
            signal_exact(parent, signal.SIGTERM)
            signal_exact(parent, signal.SIGCONT)
        try:
            if returncode:
                raise RuntimeError('Training exited '+str(returncode))
            self.verify(job)
            self.write_job(job, state='trained_pending_independent_eval', returncode=0, train_end=time.time())
            self.data['training_completed_jobs'].append(job)
        except BaseException as exc:
            self.failed = True
            self.write_job(job, state='failed', returncode=returncode,
                           error=type(exc).__name__+': '+str(exc), end=time.time())
            self.data.setdefault('errors', []).append({'job': job, 'error': str(exc)})
        del self.active[job]
        self.data['active_jobs'].pop(job)
        self.state('parallel_training' if not self.failed else 'draining_after_failure', training_ended_job=job)

    def run_phase(self, command, log, label, cwd=None):
        log = pathlib.Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('xb') as output:
            process = subprocess.Popen(command, cwd=cwd or self.code, env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
            self.state(label, child_pid=process.pid, command=command, log=str(log))
            while process.poll() is None:
                self.state()
                time.sleep(10)
        if process.returncode:
            raise RuntimeError(label+' exited '+str(process.returncode)+'; '+str(log))

    def postprocess(self):
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
        if any(line.strip().isdigit() for line in query.stdout.splitlines()):
            raise RuntimeError('GPU not idle before exclusive postprocessing')
        spec = importlib.util.spec_from_file_location('round2_serial_helpers', self.tools/'run_round2_queue.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.CODE = self.code
        helper.PHASE = self.phase
        helper.PLAN = self.plan
        helper.ADDONS = self.tools
        helper.run = self.run_phase
        if self.plan['account'] == 'B':
            self.run_phase([sys.executable, str(self.tools/'benchmark_soep_lite.py')],
                           self.stage/'soep_cost_audit.log', 'exclusive_soep_cost_audit')
        for model in self.plan['existing_models']:
            self.exports[model['name']] = helper.export_model(model['name'], self.code/model['config'], pathlib.Path(model['checkpoint']), cli_test=True)
        for other in ['baseline', 'lowpass', 'singlephase']:
            helper.analyze_pair(self.exports['existing_'+other], self.exports['existing_c'],
                                'existing_C_minus_'+other+'_seed'+str(self.plan['primary_seed']))
        for item in self.plan['jobs']:
            arm, seed = item.split(':'); job = arm+'_seed'+seed
            try:
                self.exports[job] = helper.export_model(job, self.code/'configs/experiments/phase'/(arm+'.yml'),
                    self.phase/'outputs'/job/'best_stg2.pth', soep=arm == 'soep_lite_full96', cli_test=True)
                if arm == 'c_mean_full96':
                    reference = self.exports['c_full96_seed2'] if seed == '2' else self.exports['existing_c']
                    helper.analyze_pair(self.exports[job], reference, 'C_minus_Cmean_seed'+seed)
                elif arm == 'rank16_mean_full96':
                    helper.analyze_pair(self.exports[job], self.exports['rank16_full96_seed'+seed], 'rank16_minus_rank16mean_seed'+seed)
                elif arm == 'soep_lite_full96':
                    helper.analyze_pair(self.exports[job], self.exports['existing_c'], 'C_minus_SOEPlite_seed'+seed)
                self.write_job(job, state='complete', end=time.time(), independent_evaluation_complete=True)
                self.data['completed_jobs'].append(job)
                self.state('job_complete', completed_job=job)
            except BaseException as exc:
                self.write_job(job, state='failed', error=type(exc).__name__+': '+str(exc), end=time.time())
                raise
        if self.plan['account'] == 'A':
            for label in ['baseline', 'c', 'lowpass']:
                model = next(row for row in self.plan['existing_models'] if row['name'] == 'existing_'+label)
                self.run_phase([sys.executable, '-u', str(self.tools/'run_translation_suite.py'), '--config', str(self.code/model['config']),
                    '--checkpoint', model['checkpoint'], '--annotations', self.plan['val_annotations'],
                    '--zero-predictions', str(self.exports['existing_'+label]/'val/predictions.json'),
                    '--out', str(self.phase/'round2_translation'/label)],
                    self.phase/'round2_translation'/(label+'.log'), 'translation_'+label)

    def main(self):
        with (self.stage/'controller.lock').open('x') as handle:
            handle.write(str(os.getpid()))
        assert hashlib.sha256((self.phase/'artifacts/common_init_seed0.pth').read_bytes()).hexdigest() == self.plan['common_init_sha256']
        self.state('ready_for_handoff')
        deadline = time.time()+300
        while not (self.stage/'handoff_committed.json').exists():
            if time.time() > deadline:
                raise RuntimeError('No handoff committed; no training started')
            time.sleep(1)
        self.handoff = json.loads((self.stage/'handoff_committed.json').read_text())
        adopted = self.handoff.get('adopted_training')
        if adopted:
            job = adopted['job']
            assert same_process(adopted['identity'])
            assert same_process(self.handoff['old_controller'])['state'] in ['T', 't']
            self.active[job] = {'identity': adopted['identity'], 'adopted': True}
            self.data['active_jobs'][job] = {'pid': adopted['identity']['pid'], 'start_ticks': adopted['identity']['start_ticks'],
                                           'adopted': True, 'log': str(self.phase/'logs'/(job+'.log'))}
        remaining = [item for item in self.plan['jobs'] if not adopted or item.split(':')[0]+'_seed'+item.split(':')[1] != adopted['job']]
        self.state('parallel_training')
        while self.active or remaining or self.external():
            external = self.external()
            for job, info in list(self.active.items()):
                code = adopted_returncode(info['identity']) if info.get('adopted') else info['process'].poll()
                if code is not None:
                    self.finish_training(job, code)
            if not self.failed:
                while remaining and slots_available(len(self.active), len(external)):
                    try:
                        started = self.start(remaining[0])
                    except BaseException as exc:
                        self.failed = True
                        self.data.setdefault('errors', []).append({'spec': remaining[0], 'error': str(exc)})
                        self.state('draining_after_failure')
                        break
                    if not started:
                        break
                    remaining.pop(0)
            elif not self.active and not external:
                self.state('failed', not_started_jobs=remaining)
                return
            self.state()
            time.sleep(10)
        self.state('all_training_complete_pending_independent_eval')
        self.postprocess()
        self.state('complete')


if __name__ == '__main__':
    queue = Queue(pathlib.Path(__file__).resolve().parent)
    try:
        queue.main()
    except BaseException as exc:
        queue.state('failed', error=type(exc).__name__+': '+str(exc))
        raise
