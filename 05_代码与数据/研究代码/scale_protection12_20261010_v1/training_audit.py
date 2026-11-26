"""Passive native first100 identity and actual update/EMA counters."""
import hashlib
import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            v = x.detach().as_subclass(torch.Tensor).cpu().contiguous()
            h.update(str((str(v.dtype), tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                add(v)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


class TrainingAudit:
    def __init__(self, model, optimizer, output, reference, reference_sha):
        self.model, self.output = model, Path(output)
        self.attempts = self.steps = 0
        self.epoch = None
        self.epoch_start_attempts = 0
        assert hashlib.sha256(Path(reference).read_bytes()).hexdigest() == reference_sha
        self.reference = [json.loads(x) for x in Path(reference).read_text().splitlines()]
        assert len(self.reference) == 100
        self.stream = (self.output/'first100_training_inputs.jsonl').open('x')
        self.handles = [model.register_forward_pre_hook(self.forward, with_kwargs=True),
                        optimizer.register_step_post_hook(self.after_step)]

    def enter_epoch(self, epoch):
        assert 0 <= epoch <= 11
        self.epoch = epoch
        self.epoch_start_attempts = self.attempts

    def forward(self, module, args, kwargs):
        if not module.training:
            return
        self.attempts += 1
        epoch_attempt = self.attempts-self.epoch_start_attempts
        if self.epoch in (0, 4, 8, 11) and epoch_attempt in (1, 101, 501, 1001, 1501):
            samples = args[0]
            targets = args[1] if len(args) > 1 else kwargs['targets']
            assert samples.ndim == 4 and len(targets) == len(samples)
            scale = np.array([samples.shape[-1], samples.shape[-2]], dtype=np.float64)
            sizes = [t['boxes'].detach().as_subclass(torch.Tensor).cpu().double().numpy()[:, 2:]*scale for t in targets]
            side = np.concatenate([v.min(1) for v in sizes]) if sizes else np.empty(0)
            hist = np.histogram(side, bins=[0., 2., 4., 8., 16., 32., np.inf])[0]
            assert np.isfinite(side).all() and (side > 0).all() and int(hist.sum()) == len(side)
            with (self.output/'sampled_training_target_sizes.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch_zero_based=self.epoch, epoch_attempt=epoch_attempt,
                    samples_shape=list(samples.shape), target_exposures=len(side), target_histogram=hist.tolist(),
                    bands=['under2','2_to_under4','4_to_under8','8_to_under16','16_to_under32','ge32'],
                    scope='Actual post-collate target exposures; MixUp/DN repetitions and source annotation identities not inferred.'))+'\n')
        if self.attempts <= 100:
            row = dict(attempt=self.attempts, inputs_targets_sha256=digest((args, kwargs)),
                rng_sha256=digest((torch.get_rng_state(), torch.cuda.get_rng_state(),
                    pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))))
            self.stream.write(json.dumps(row)+'\n'); self.stream.flush()
            if row != self.reference[self.attempts-1]:
                (self.output/'input_parity_failure.json').write_text(json.dumps(
                    dict(actual=row, reference=self.reference[self.attempts-1], automatic_retry=False), indent=2))
                raise RuntimeError('Native first100 inputs/RNG differ before scale protection is active')
            if self.attempts == 100:
                (self.output/'first100_parity.json').write_text(json.dumps(dict(
                    all100_rows_exact=True, implementation='same historical native model constructor; no added parameters',
                    scope='Epoch0 inputs/pre-forward RNG only; no assertion that augmented later learning trajectories are equal.'), indent=2))

    def after_step(self, optimizer, args, kwargs):
        self.steps += 1
        if self.steps % 100 == 0:
            self.live()

    def live(self):
        (self.output/'training_audit_live.json').write_text(json.dumps(dict(
            attempts=self.attempts, actual_optimizer_steps=self.steps,
            CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved())))

    def epoch_complete(self, epoch, stats, ema):
        assert 0 <= epoch <= 11 and ema is not None and int(ema.updates) == self.attempts
        row = dict(epoch_zero_based=epoch, attempts_total=self.attempts,
            actual_optimizer_steps_total=self.steps, attempts_without_optimizer_step=self.attempts-self.steps,
            EMA_updates=int(ema.updates), CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(), no_new_model_parameters=True,
            scale_protection_active=epoch >= 4, no_step_definition='attempts minus optimizer posthooks; cause not inferred')
        with (self.output/'training_audit_epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        self.live()

    def close(self):
        for h in self.handles:
            h.remove()
        self.stream.close()
