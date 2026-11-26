"""First-order, final-main-loss feedback to readout parameters only.

The detached reference leaf exposes the native decoder VJP without connecting
native shared inputs. The same head is evaluated again on detached inputs;
the zero-valued proxy carries that VJP only to this head's parameters.
"""
import re
import time
import torch
from torch import nn

MAIN_KEYS = ('loss_mal', 'loss_bbox', 'loss_giou', 'loss_fgl')
CONTEXT_KEY = '_p3_feedback_context'


class FeedbackVJPError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Final-reference feedback failed: '+repr(receipt))


class ZeroValueSurrogate(torch.autograd.Function):
    """Zero forward, actual first-order VJP backward (including AMP nonfinite).

    Avoid inf-inf in the loss value. Do not sanitize, replace, or clip g:
    native GradScaler must see its real nonfinite parameter gradients.
    """
    @staticmethod
    def forward(ctx, shadow, g):
        if shadow.shape != g.shape or shadow.dtype != torch.float32 or g.dtype != torch.float32:
            raise RuntimeError('Expected same-shape FP32 shadow and detached VJP')
        ctx.save_for_backward(g)
        return shadow.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        (g,) = ctx.saved_tensors
        return grad_output*g, None


def _timer(tensor):
    if tensor.is_cuda:
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        return start, end
    return None


def make_feedback_context(head, semantic, p3, z0, z1):
    reference = z1.detach().requires_grad_(True)
    started = time.perf_counter()
    events = _timer(z1)
    shadow = head(semantic.detach(), p3.detach(), z0.detach())
    if events:
        events[1].record()
    return reference, dict(reference=reference, shadow_delta=shadow,
        head_parameters=tuple(head.parameters()), shadow_events=events,
        shadow_host_seconds=time.perf_counter()-started,
        native_autocast_enabled=torch.is_autocast_enabled())


def final_main_loss(losses):
    # This exact set was read from the frozen native criterion and main output:
    # final has pred_corners but no teacher_corners; DDF is auxiliary-only.
    suffix = r'_(?:(?:aux|enc|dn)_\d+|pre|dn_pre)$'
    observed = {key for key in losses if not re.search(suffix, key)}
    if observed != set(MAIN_KEYS):
        raise RuntimeError('Unexpected native main loss keys: '+repr(sorted(observed)))
    return sum(losses[key] for key in MAIN_KEYS)


class P3FeedbackCriterion(nn.Module):
    __inject__ = ['native_criterion']

    def __init__(self, native_criterion, feedback_mode='final_only_feedback'):
        super().__init__()
        if feedback_mode not in ('aux_only', 'final_only_feedback'):
            raise ValueError(feedback_mode)
        self.native_criterion = native_criterion
        self.feedback_mode = feedback_mode
        self.feedback_calls = 0
        self._native_scaler = None
        self.last_failure = None
        self.last_feedback_event = None
        self.nonfinite_vjp_calls = 0
        self.first8 = []
        self.pending_events = []
        self.cost = dict(shadow_host_seconds=0., main_vjp_host_seconds=0.,
                         shadow_gpu_ms=0., main_vjp_gpu_ms=0., audit_host_seconds=0.)

    def set_epoch(self, epoch):
        return self.native_criterion.set_epoch(epoch)

    def set_model(self, model):
        return self.native_criterion.set_model(model)

    def reset_model(self):
        # Native ModelSaverFunc removes/reinstalls this model reference around
        # checkpoint state capture. Do not leave the wrapper missing this API.
        return self.native_criterion.reset_model()

    def set_native_scaler(self, scaler):
        self._native_scaler = scaler

    def gradient_scale(self, context):
        # Native force_fp32 forwards execute outside autocast even when the
        # solver retains a GradScaler. Capture that forward state, not the
        # criterion's deliberately-disabled autocast context.
        if not context['native_autocast_enabled']:
            return 1.
        if self._native_scaler is None:
            raise RuntimeError('AMP forward without the native GradScaler binding')
        scale = float(self._native_scaler.get_scale())
        import math
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError('Invalid current native gradient scale')
        return scale

    @property
    def weight_dict(self):
        return self.native_criterion.weight_dict

    def forward(self, outputs, targets, **kwargs):
        native_outputs = {k: v for k, v in outputs.items() if k != CONTEXT_KEY}
        losses = self.native_criterion(native_outputs, targets, **kwargs)
        context = outputs.get(CONTEXT_KEY)
        if self.feedback_mode == 'aux_only':
            if context is not None:
                raise RuntimeError('Aux-only model unexpectedly supplied feedback')
            return losses
        if context is None:
            raise RuntimeError('Training feedback context is missing')
        reference, shadow = context['reference'], context['shadow_delta']
        if reference.shape != shadow.shape or not reference.is_leaf:
            raise RuntimeError('Expected same-shape native-reference leaf and shadow')
        main = final_main_loss(losses)
        scale = self.gradient_scale(context)
        started = time.perf_counter()
        events = _timer(reference)
        g_scaled = torch.autograd.grad(main.float()*scale, reference,
            retain_graph=True, allow_unused=False)[0]
        # Division MUST occur in FP32, after scaled backward has traversed all
        # FP16 intermediates. The outer native scaler scales the proxy once.
        g = (g_scaled.float()/scale).detach()
        if events:
            events[1].record()
        self.cost['main_vjp_host_seconds'] += time.perf_counter()-started
        self.cost['shadow_host_seconds'] += context['shadow_host_seconds']
        if events:
            self.pending_events.append((context['shadow_events'], events))
        # Strictly zero forward avoids inf-inf while preserving the actual VJP
        # in backward. g is detached: no second derivative/shared-input path.
        finite_fraction = float(torch.isfinite(g).float().mean())
        vjp_nonfinite = finite_fraction != 1.
        event = dict(feedback_attempt=self.feedback_calls+1,vjp_gradient_scale=scale,
            native_autocast_enabled=context['native_autocast_enabled'],
            reference_dtype=str(reference.dtype),g_scaled_dtype=str(g_scaled.dtype),
            reference_shape=list(reference.shape),g_scaled_finite_fraction=finite_fraction,
            g_unscaled_finite_fraction=finite_fraction,vjp_nonfinite=vjp_nonfinite)
        self.last_feedback_event = event
        if vjp_nonfinite:
            self.nonfinite_vjp_calls += 1
        if vjp_nonfinite and not context['native_autocast_enabled']:
            self.last_failure = dict(status='FAILED_SCALED_VJP_FAIL_FAST',
                **event,policy='non-AMP VJP must be finite; no automatic restart')
            raise FeedbackVJPError(self.last_failure)
        zero = ZeroValueSurrogate.apply(shadow.float(),g)
        self.feedback_calls += 1
        if self.feedback_calls <= 8:
            audit_started = time.perf_counter()
            params = context['head_parameters']
            proxy_grads = torch.autograd.grad(zero*scale, params, retain_graph=True,
                                              allow_unused=False)
            aux = losses['loss_bbox_enc_0']+losses['loss_giou_enc_0']
            aux_grads = torch.autograd.grad(aux.float()*scale, params, retain_graph=True,
                                            allow_unused=False)
            def norm(values):
                value=float(torch.stack([(v.detach().float()/scale).square().sum() for v in values]).sum().sqrt())
                import math
                return value if math.isfinite(value) else None
            pn, an = norm(proxy_grads), norm(aux_grads)
            def finite_norm(value):
                result=float(value.detach().float().norm())
                import math
                return result if math.isfinite(result) else None
            self.first8.append(dict(attempt=self.feedback_calls, main_loss_keys=list(MAIN_KEYS),
                reference_shape=list(reference.shape), reference_dtype=str(reference.dtype),
                shadow_dtype=str(shadow.dtype), native_autocast_enabled=context['native_autocast_enabled'],
                vjp_gradient_scale=scale,g_scaled_ref_norm=finite_norm(g_scaled),
                g_ref_nonzero=int(torch.count_nonzero(g)),g_ref_norm=finite_norm(g),
                vjp_nonfinite=vjp_nonfinite,g_scaled_finite_fraction=finite_fraction,
                proxy_head_grad_norm=pn, encoder_aux_head_grad_norm=an,
                proxy_to_encoder_aux_norm=pn/an if pn is not None and an is not None and an>0 else None,
                zero_scalar=float(zero.detach()),
                finite_head_grads=all(bool(torch.isfinite(v).all()) for v in (*proxy_grads,*aux_grads))))
            self.cost['audit_host_seconds'] += time.perf_counter()-audit_started
        losses['loss_final_only_feedback_zero'] = zero
        return losses

    def feedback_audit(self):
        # Called at the epoch boundary, never synchronize solely for timing
        # inside a training batch. CUDA times include the measured operation.
        if self.pending_events:
            torch.cuda.synchronize()
            for shadow_events, main_events in self.pending_events:
                self.cost['shadow_gpu_ms'] += shadow_events[0].elapsed_time(shadow_events[1])
                self.cost['main_vjp_gpu_ms'] += main_events[0].elapsed_time(main_events[1])
            self.pending_events.clear()
        return dict(feedback_calls=self.feedback_calls, nonfinite_vjp_calls=self.nonfinite_vjp_calls,first8=self.first8,
                    cumulative_cost=dict(self.cost), optimizer_steps_in_this_module=0,
                    boundary='AMP overflow and global clipping can change all optimizer updates; shared raw-gradient isolation is narrower')
