"""Continue saved epoch boundaries, skipping all completed training jobs."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

STAGE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('original_parallel_v3', STAGE.parent/'execution_v3/run_parallel_queue.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class ResumeQueue(original.Queue):
    def __init__(self, stage):
        super().__init__(stage)
        self.code = self.stage.parent/'execution_v3/code'
        self.tools = self.stage.parent/'execution_v3/tools'
        self.recovery = json.loads((self.stage/'resume_plan.json').read_text())
        self.resume_by_job = {r['job']: r for r in self.recovery['resume_jobs']}
        self.data.update(version='resume_parallel_v4', code_directory=str(self.code),
                         resumed_jobs=[], training_completed_jobs=list(self.recovery['completed_jobs']),
                         previous_execution='execution_v3', rng_policy='Original per-epoch reseeding unchanged; no bitwise guarantee.')

    def start(self, item):
        arm, seed = item.split(':')
        job = arm+'_seed'+seed
        if job in self.data['training_completed_jobs']:
            raise RuntimeError('Refusing to retrain completed job: '+job)
        if job not in self.resume_by_job:
            return super().start(item)
        if len(original.main_training_processes(self.phase)) >= 2:
            return False
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        if int(free.strip().splitlines()[0]) < 10240:
            return False
        row = self.resume_by_job[job]
        checkpoint = pathlib.Path(row['backup_checkpoint'])
        assert checkpoint.is_file() and row['backup_sha256'] == row['sha256']
        output = self.phase/'outputs'/job
        log = self.phase/'logs'/(job+'.log')
        records = [json.loads(line) for line in (output/'log.txt').read_text().splitlines() if line.strip()]
        epochs = [r['epoch'] for r in records if 'test_coco_eval_bbox' in r]
        assert epochs == row['logged_epochs']
        command = [sys.executable, '-u', str(self.code/('train_round2.py' if arm == 'soep_lite_full96' else 'train.py')),
                   '-c', str(self.code/'configs/experiments/phase'/(arm+'.yml')),
                   '--seed', seed, '--use-amp', '--resume', str(checkpoint), '--output-dir', str(output)]
        assert '-t' not in command and '--tuning' not in command
        with log.open('ab') as handle:
            handle.write(('\nRESUME_V4 '+json.dumps({'time': time.time(), 'checkpoint': str(checkpoint),
                                                    'next_round_one_based': row['next_round_one_based']})+'\n').encode())
            handle.flush()
            process = subprocess.Popen(command, cwd=self.code, env=self.env, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT)
        identity = original.identity(process.pid)
        if identity is None:
            raise RuntimeError('Resumed trainer exited immediately: '+job)
        self.active[job] = {'process': process, 'identity': identity}
        self.data['active_jobs'][job] = {'pid': process.pid, 'start_ticks': identity['start_ticks'], 'command': command,
                                       'log': str(log), 'adopted': False, 'resumed': True,
                                       'resume_after_round': row['last_epoch_zero_based']+1}
        self.write_job(job, state='running', returncode=None, resume_start=time.time(),
                       parallel_queue_pid=os.getpid(), resumed_from_checkpoint=str(checkpoint),
                       resumed_checkpoint_sha256=row['sha256'], resume_after_round=row['last_epoch_zero_based']+1)
        self.data['resumed_jobs'].append(job)
        self.state('parallel_training', newly_started_job=job)
        return True

    def main(self):
        with (self.stage/'controller.lock').open('x') as handle:
            handle.write(str(os.getpid()))
        assert not original.main_training_processes(self.phase), 'Refusing duplicate live training'
        assert self.recovery['status'] == 'ready_to_resume'
        remaining = [item for item in self.plan['jobs'] if item.replace(':', '_seed') not in self.data['training_completed_jobs']]
        self.state('parallel_training', remaining_specs=remaining)
        while self.active or remaining or self.external():
            external = self.external()
            for job, info in list(self.active.items()):
                code = info['process'].poll()
                if code is not None:
                    self.finish_training(job, code)
            if not self.failed:
                while remaining and original.slots_available(len(self.active), len(external)):
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
            self.state(remaining_specs=remaining)
            time.sleep(10)
        self.state('all_training_complete_pending_independent_eval')
        self.postprocess()
        self.state('complete')


if __name__ == '__main__':
    queue = ResumeQueue(STAGE)
    try:
        queue.main()
    except BaseException as exc:
        queue.state('failed', error=type(exc).__name__+': '+str(exc))
        raise
