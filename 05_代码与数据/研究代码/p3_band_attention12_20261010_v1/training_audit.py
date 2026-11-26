"""Passive audit: exact first100 input/RNG hashes, actual steps and gate gradients."""
import hashlib
import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch


def digest(value):
    # Identical encoding to the archived native BASE/P3 audit for direct comparison.
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            v = x.detach().as_subclass(torch.Tensor).cpu().contiguous()
            h.update(str((str(v.dtype), tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                h.update(str(key).encode()); add(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x:
                add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def norms(module, gradients=False):
    if module is None:
        return {}
    result = {}
    for name, parameter in module.named_parameters():
        value = parameter.grad if gradients else parameter
        result[name] = None if value is None else dict(
            norm=float(value.detach().float().norm()), finite=bool(torch.isfinite(value).all()))
    return result


class TrainingAudit:
    def __init__(self, model, optimizer, output):
        self.model = model; self.output = Path(output)
        self.attention = model.band_attention
        self.attempts = 0; self.steps = 0; self.rows = []; self.handles = []
        self.stream = (self.output/'first100_training_inputs.jsonl').open('x', encoding='utf-8')
        self.handles.append(model.register_forward_pre_hook(self.forward, with_kwargs=True))
        self.handles.append(optimizer.register_step_pre_hook(self.before_step))
        self.handles.append(optimizer.register_step_post_hook(self.after_step))

    def forward(self, module, args, kwargs):
        if not module.training:
            return
        self.attempts += 1
        if self.attempts <= 100:
            row = dict(attempt=self.attempts, inputs_targets_sha256=digest((args, kwargs)),
                rng_sha256=digest((torch.get_rng_state(), torch.cuda.get_rng_state(),
                    pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))))
            self.stream.write(json.dumps(row)+'\n'); self.stream.flush()

    def before_step(self, optimizer, args, kwargs):
        if self.steps < 8:
            self.step_gradients = norms(self.attention, gradients=True)

    def after_step(self, optimizer, args, kwargs):
        self.steps += 1
        if self.steps <= 8:
            self.rows.append(dict(actual_optimizer_step=self.steps, attempt=self.attempts,
                attention_gradients_after_native_unscale_and_clip=self.step_gradients,
                attention_parameters=norms(self.attention)))
            (self.output/'first8_optimizer_steps.json').write_text(json.dumps(self.rows, indent=2), encoding='utf-8')
        if self.steps % 100 == 0:
            (self.output/'training_audit_live.json').write_text(json.dumps(dict(
                attempts=self.attempts, actual_optimizer_steps=self.steps)), encoding='utf-8')

    def epoch_complete(self, epoch, log_stats, ema):
        row = dict(epoch_zero_based=epoch, attempts_total=self.attempts,
            actual_optimizer_steps_total=self.steps, attempts_without_optimizer_step=self.attempts-self.steps,
            EMA_updates=int(ema.updates) if ema is not None else None,
            attention_parameters=norms(self.attention),
            no_step_definition='Forward attempts minus actual optimizer post-hooks; not a diagnosed overflow count')
        with (self.output/'training_audit_epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.stream.close()
