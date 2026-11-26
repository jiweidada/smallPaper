"""Reviewed controller draft; user approved repair on 2026-10-16.

Not deployed: both SSH endpoints reset their handshakes during repair preflight.
Deployment must first preserve the shared v4 outputs, stop only the verified v4
controllers/trainers, then copy each immutable input to that job's output folder.

ResumeQueue v4 can only be used with checkpoint parents equal to job output dirs:
BaseSolver ignores --output-dir when resume is set. This draft changes scheduling
metadata only; production v3 model, solver, configuration and recipe remain intact.
"""
import importlib.util
import hashlib
import json
import pathlib

STAGE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recovery_v4', STAGE.parent/'execution_v4/resume_parallel_queue_v4.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class FixedResumeQueue(original.ResumeQueue):
    def __init__(self, stage):
        super().__init__(stage)
        self.data.update(version='resume_parallel_v5', previous_execution='execution_v4')
        for job, row in self.resume_by_job.items():
            expected = self.phase/'outputs'/job
            checkpoint = pathlib.Path(row['backup_checkpoint'])
            if checkpoint.parent != expected:
                raise RuntimeError('Resume checkpoint parent must be the unique task output directory: '+job)
        directories = [str(pathlib.Path(r['backup_checkpoint']).parent) for r in self.resume_by_job.values()]
        if len(directories) != len(set(directories)):
            raise RuntimeError('Refusing shared output directories')

    def start(self, item):
        arm, seed = item.split(':')
        job = arm+'_seed'+seed
        if job in self.resume_by_job:
            row = self.resume_by_job[job]
            checkpoint = pathlib.Path(row['backup_checkpoint'])
            if checkpoint.parent != self.phase/'outputs'/job:
                raise RuntimeError('Resume output path mismatch: '+job)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Immutable resume checkpoint changed: '+job)
        return super().start(item)


if __name__ == '__main__':
    queue = FixedResumeQueue(STAGE)
    try:
        queue.main()
    except BaseException as exc:
        queue.state('failed', error=type(exc).__name__+': '+str(exc))
        raise
