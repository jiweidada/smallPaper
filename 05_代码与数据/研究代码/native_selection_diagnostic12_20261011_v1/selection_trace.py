"""Observe the native default query selector without recomputing its Top-K.

Use only in a separate post-training diagnostic process. Dense anchor logits
are positional priors, not 8400 regressed proposal boxes. No head is rerun.
"""
from types import MethodType
import torch
from torch.overrides import TorchFunctionMode


class _ObserveTopK(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.indices = None
        self.native_result = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.topk:
            self.calls += 1
            self.indices = result.indices.detach().clone()
            self.native_result = result
        return result


class NativeSelectionObserver:
    """A temporary, instance-local wrapper; native arguments/returns unchanged.

    take_record() returns CPU copies in their original native dtypes. The caller
    must pair the record with the image IDs of this same batch, and explicitly
    dispose of each record before the next batch. Nested/parallel use on one
    decoder and training use are unsupported in diagnostic mode.
    """
    def __init__(self, decoder, mode='post_training_diagnostic'):
        if mode not in ('post_training_diagnostic', 'synthetic_engineering'):
            raise ValueError('Unknown observation mode')
        self.decoder, self.mode = decoder, mode
        self.entered = False
        self.record = None
        self.calls = 0

    def __enter__(self):
        if self.entered:
            raise RuntimeError('Observer is already attached')
        d = self.decoder
        if d.query_select_method != 'default' or getattr(d, 'mgsa_groups', 0) or getattr(d, 'o2m_t', False):
            raise ValueError('Only the bound native default selector is supported')
        if self.mode == 'post_training_diagnostic' and d.training:
            raise ValueError('Post-training diagnostic must be in eval mode')
        self.had_instance_method = '_select_topk' in d.__dict__
        self.previous_instance_method = d.__dict__.get('_select_topk')
        self.original = d._select_topk
        observer = self
        def observed(decoder, memory, outputs_logits, outputs_anchors_unact, topk):
            if observer.record is not None:
                raise RuntimeError('Previous batch record was not consumed')
            if observer.mode == 'post_training_diagnostic' and decoder.training:
                raise ValueError('Diagnostic decoder switched to training')
            if memory.ndim != 3 or outputs_logits.ndim != 3 or outputs_anchors_unact.ndim != 3:
                raise ValueError('Unexpected native selector inputs')
            if memory.shape[:2] != outputs_logits.shape[:2] or outputs_anchors_unact.shape != (*memory.shape[:2], 4):
                raise ValueError('Dense memory/logit/anchor shapes differ')
            with _ObserveTopK() as scope:
                result = observer.original(memory, outputs_logits, outputs_anchors_unact, topk)
            if scope.calls != 1 or scope.indices.shape != (memory.shape[0], topk):
                raise ValueError('Expected one native Top-K operation')
            # Copies occur after the original selector has returned. Preserve
            # inf in invalid anchor priors; do not mask, rerank or select again.
            observer.record = dict(
                dense_class_logits=outputs_logits.detach().cpu().clone(),
                dense_anchor_logits=outputs_anchors_unact.detach().cpu().clone(),
                selected_indices=scope.indices.cpu().clone(),
                selected_anchor_logits=result[2].detach().cpu().clone(),
                selector_training=bool(decoder.training),
                original_dtypes=dict(class_logits=str(outputs_logits.dtype), anchors=str(outputs_anchors_unact.dtype)),
                native_topk_calls=scope.calls,
                dense_anchor_role='unregressed positional prior; not dense predicted boxes')
            observer.calls += 1
            return result
        self.installed_method = MethodType(observed, d)
        d._select_topk = self.installed_method
        self.entered = True
        return self

    def take_record(self):
        if self.record is None:
            raise RuntimeError('No completed native selector observation')
        record, self.record = self.record, None
        return record

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.entered:
            # Restore precisely the preexisting instance/class lookup, including
            # when native selection or subsequent processing raises an error.
            if self.had_instance_method:
                self.decoder._select_topk = self.previous_instance_method
            else:
                delattr(self.decoder, '_select_topk')
            self.entered = False
        return False
