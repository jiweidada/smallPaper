"""Passive checks on actual training; no added forward/backward or RNG draws."""
import hashlib, json, math, pickle, random, time
from pathlib import Path
import numpy as np
import torch

def digest(value):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().as_subclass(torch.Tensor).cpu().contiguous()
            h.update(str((str(v.dtype),tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x): h.update(str(key).encode()); add(x[key])
        elif isinstance(x,(tuple,list)):
            for item in x: add(item)
        else: h.update(repr(x).encode())
    add(value)
    return h.hexdigest()

class TrainingAudit:
    def __init__(self,model,optimizer,output,criterion=None,scaler=None):
        self.model=model;self.optimizer=optimizer;self.output=Path(output)
        self.criterion=criterion;self.scaler=scaler;self.started=time.perf_counter()
        self.clip_rows=[];self.clip_calls=0;self.last_clip=None
        self.attempts=0;self.steps=0;self.handles=[];self.rows=[]
        self.pending_attempt=None;self.closed=False
        self.attempt_totals=dict(completed_attempts=0,incomplete_attempts=0,
            native_skipped_attempts=0,scale_backoffs=0,nonfinite_vjp_attempts=0,
            nonfinite_vjp_verified_skip_backoffs=0,
            nonfinite_vjp_without_verified_skip_backoff=0,
            feedback_event_mismatches=0,unexpected_step_deltas=0,logged_attempts=0)
        self.residual=getattr(model.decoder,'local_readout',None)
        self.stream=(self.output/'first100_training_inputs.jsonl').open('x',encoding='utf-8')
        self.attempt_stream=(self.output/'training_attempt_events.jsonl').open('x',encoding='utf-8')
        self.handles.append(model.register_forward_pre_hook(self.forward,with_kwargs=True))
        self.handles.append(optimizer.register_step_pre_hook(self.before_step))
        self.handles.append(optimizer.register_step_post_hook(self.after_step))
        # Observe the exact native clipping call and return its original value.
        # No gradient/threshold changes. Restore the process-local wrapper on close.
        self.original_clip=torch.nn.utils.clip_grad_norm_
        def observed_clip(parameters,*args,**kwargs):
            parameters=list(parameters)
            result=self.original_clip(parameters,*args,**kwargs)
            self.clip_calls+=1
            # Scaler skips do not increment optimizer steps. Never attach a
            # previous attempt's clipping record to a later successful step.
            self.last_clip=None
            if self.steps<8:
                post=torch.stack([p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]).sum().sqrt()
                pre=float(result);post=float(post)
                import math
                self.last_clip=dict(attempt=self.attempts,clip_call=self.clip_calls,
                    preclip_norm=pre if math.isfinite(pre) else None,
                    postclip_norm=post if math.isfinite(post) else None,
                    max_norm=float(args[0] if args else kwargs['max_norm']),
                    scaler_scale=float(self.scaler.get_scale()) if self.scaler is not None else None)
                if self.clip_calls<=8:
                    self.clip_rows.append(dict(self.last_clip))
            return result
        self.observed_clip=observed_clip
        torch.nn.utils.clip_grad_norm_=observed_clip
    def _scale(self):
        if self.scaler is None:return None
        value=float(self.scaler.get_scale())
        return value if math.isfinite(value) else None
    def _attempt_summary(self):
        return dict(**self.attempt_totals,attempted_training_forwards=self.attempts,
            actual_optimizer_steps=self.steps,
            pending_attempt=None if self.pending_attempt is None else self.pending_attempt['attempt'],
            criterion_nonfinite_vjp_calls=getattr(self.criterion,'nonfinite_vjp_calls',None),
            native_skipped_attempt_policy='Bound native engine still advances data, EMA and LR after an AMP-skipped optimizer step; per-attempt EMA/LR calls are not independently instrumented.')
    def _write_attempt_summary(self):
        (self.output/'training_attempt_summary.json').write_text(
            json.dumps(self._attempt_summary(),indent=2,allow_nan=False),encoding='utf-8')
    def _settle_attempt(self,source,completed=True):
        """Observe only after native scaler.update(), never in optimizer posthook.

        A following training forward or epoch completion proves the prior native
        iteration returned. close() preserves an interrupted attempt as incomplete;
        an absent optimizer step in that case is not evidence of an AMP skip.
        """
        pending=self.pending_attempt
        if pending is None:return
        self.pending_attempt=None
        event=getattr(self.criterion,'last_feedback_event',None)
        expected=pending['feedback_calls_before']
        event_matches=(expected is not None and isinstance(event,dict)
                       and event.get('feedback_attempt')==expected+1)
        feedback_completed=(expected is not None and
            getattr(self.criterion,'feedback_calls',None)==expected+1)
        event=dict(event) if event_matches else None
        steps_delta=self.steps-pending['optimizer_steps_before']
        after=self._scale();before=pending['scaler_scale_before']
        backoff=bool(completed and before is not None and after is not None and after<before)
        skipped=bool(completed and steps_delta==0 and pending['scaler_enabled'])
        nonfinite=bool(event is not None and event.get('vjp_nonfinite'))
        verified=bool(nonfinite and skipped and backoff and feedback_completed
                      and event.get('native_autocast_enabled'))
        issues=[]
        if completed and steps_delta not in (0,1):
            issues.append('unexpected_optimizer_step_delta')
            self.attempt_totals['unexpected_step_deltas']+=1
        if completed and pending['feedback_expected'] and not (event_matches and feedback_completed):
            issues.append('feedback_event_does_not_match_this_attempt')
            self.attempt_totals['feedback_event_mismatches']+=1
        if completed and nonfinite and not verified:
            issues.append('nonfinite_VJP_without_verified_native_skip_and_backoff')
        row=dict(**pending,settled_at=source,
            status='COMPLETED_NATIVE_ITERATION' if completed else 'INCOMPLETE_ATTEMPT',
            optimizer_steps_after=self.steps,optimizer_steps_delta=steps_delta,
            scaler_scale_after=after,scale_backoff_observed=backoff,
            native_skip_inferred_from_absent_optimizer_posthook=skipped,
            feedback_event_matches_attempt=event_matches,
            feedback_call_completed=feedback_completed,feedback_event=event,
            nonfinite_vjp_native_skip_backoff_verified=verified,audit_issues=issues)
        key='completed_attempts' if completed else 'incomplete_attempts'
        self.attempt_totals[key]+=1
        self.attempt_totals['native_skipped_attempts']+=int(skipped)
        self.attempt_totals['scale_backoffs']+=int(backoff)
        self.attempt_totals['nonfinite_vjp_attempts']+=int(nonfinite)
        self.attempt_totals['nonfinite_vjp_verified_skip_backoffs']+=int(verified)
        self.attempt_totals['nonfinite_vjp_without_verified_skip_backoff']+=int(completed and nonfinite and not verified)
        # Complete the corresponding first-eight actual-step record now that
        # GradScaler.update has run; its posthook necessarily precedes update.
        for step_row in self.rows:
            if step_row['attempt']==pending['attempt']:
                step_row.update(scaler_scale_before=before,scaler_scale_after=after,
                    attempt_settled_at=source,attempt_completed=completed,
                    feedback_event=event,scale_backoff_observed=backoff)
                (self.output/'first8_optimizer_steps.json').write_text(
                    json.dumps(self.rows,indent=2,allow_nan=False),encoding='utf-8')
                break
        if pending['attempt']<=8 or skipped or nonfinite or issues or not completed:
            self.attempt_totals['logged_attempts']+=1
            self.attempt_stream.write(json.dumps(row,allow_nan=False)+'\n')
            self.attempt_stream.flush()
            self._write_attempt_summary()
    def forward(self,module,args,kwargs):
        if not module.training:return
        self._settle_attempt('next_training_forward')
        self.attempts+=1
        self.pending_attempt=dict(attempt=self.attempts,optimizer_steps_before=self.steps,
            scaler_scale_before=self._scale(),
            scaler_enabled=self.scaler is not None and bool(self.scaler.is_enabled()),
            feedback_calls_before=getattr(self.criterion,'feedback_calls',None),
            feedback_expected=getattr(self.criterion,'feedback_mode',None)=='final_only_feedback')
        if self.attempts<=100:
            row={'attempt':self.attempts,'inputs_targets_sha256':digest((args,kwargs)),
                'rng_sha256':digest((torch.get_rng_state(),torch.cuda.get_rng_state(),
                    pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state())))}
            self.stream.write(json.dumps(row)+'\n');self.stream.flush()
    def before_step(self,opt,args,kwargs):
        if self.steps<8 and self.residual is not None:
            self.grad={n:None if p.grad is None else {'norm':float(p.grad.detach().float().norm()),
                'finite':bool(torch.isfinite(p.grad).all())} for n,p in self.residual.named_parameters()}
    def after_step(self,opt,args,kwargs):
        self.steps+=1
        if self.steps<=8:
            row={'actual_optimizer_step':self.steps,'attempt':self.attempts}
            if self.last_clip is not None and self.last_clip['attempt']!=self.attempts:
                raise RuntimeError('Stale clipping record crossed an AMP skipped attempt')
            row['clip']=self.last_clip
            if self.residual is not None:
                row.update(gradient=self.grad,output_weight_norm=float(self.residual.output.weight.detach().float().norm()),
                    output_bias_norm=float(self.residual.output.bias.detach().float().norm()))
            self.rows.append(row)
            (self.output/'first8_optimizer_steps.json').write_text(json.dumps(self.rows,indent=2),encoding='utf-8')
        if self.steps%100==0:
            (self.output/'training_audit_live.json').write_text(json.dumps({'attempts':self.attempts,
                'actual_optimizer_steps':self.steps}),encoding='utf-8')
    def epoch_complete(self,epoch,log_stats,ema):
        self._settle_attempt('epoch_complete')
        self._write_attempt_summary()
        row={'epoch_zero_based':epoch,'attempts_total':self.attempts,'actual_optimizer_steps_total':self.steps,
             'EMA_updates':int(ema.updates) if ema is not None else None,
             'elapsed_wall_seconds':time.perf_counter()-self.started,
             'amp_scaler_scale':self._scale(),'attempt_audit':self._attempt_summary()}
        if self.residual is not None:
            row['residual_output_weight_norm']=float(self.residual.output.weight.detach().float().norm())
        with (self.output/'training_audit_epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if self.criterion is not None and hasattr(self.criterion,'feedback_audit'):
            report=self.criterion.feedback_audit()
            report.update(first8_clip_calls=self.clip_rows,actual_optimizer_steps=self.steps,
                          attempts=self.attempts,elapsed_wall_seconds=time.perf_counter()-self.started,
                          attempt_audit=self._attempt_summary())
            (self.output/'feedback_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            self._settle_attempt('close_without_completed_iteration',completed=False)
            self._write_attempt_summary()
        finally:
            if torch.nn.utils.clip_grad_norm_ is self.observed_clip:
                torch.nn.utils.clip_grad_norm_=self.original_clip
            for handle in self.handles:handle.remove()
            self.stream.close()
            self.attempt_stream.close()
