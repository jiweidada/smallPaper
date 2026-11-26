"""Synthetic native-decoder/criterion integration: forwards/backwards, ZERO updates.

This deliberately does not read training data, construct an optimizer, or call
optimizer.step. It is an engineering receipt, never AP or training evidence.
"""
import argparse
import json
import time
from pathlib import Path
from runtime import HERE, activate_training_snapshot


def run(native_root, device='cpu', amp=False):
    activate_training_snapshot(native_root)
    import torch
    from model import P3ReadoutDecoder
    from integration import build_criterion
    from feedback import P3FeedbackCriterion, CONTEXT_KEY, MAIN_KEYS, final_main_loss, ZeroValueSurrogate
    if amp and not device.startswith('cuda'):
        raise ValueError('AMP check requires CUDA')
    started=time.perf_counter()
    torch.manual_seed(20261010)
    plan=json.loads((HERE/'configs/final_only_feedback_seed2.json').read_text())
    kwargs=dict(plan['model_recipe']['DFINETransformer'],num_classes=10,eval_spatial_size=[128,128])
    control=P3ReadoutDecoder('aux_only',**kwargs).to(device)
    candidate=P3ReadoutDecoder('final_only_feedback',**kwargs).to(device)
    candidate.load_state_dict(control.state_dict(),strict=True)
    inputs=[torch.randn(2,256,s,s,device=device) for s in (16,8,4)]
    targets=[dict(labels=torch.tensor([0,3],device=device),
                  boxes=torch.tensor([[.25,.25,.08,.1],[.7,.6,.15,.2]],device=device)) for _ in range(2)]
    # Exact eval equivalence at zero initialization; no shadow call in eval.
    calls=[]
    handle=candidate.local_readout.register_forward_hook(lambda *args:calls.append(True))
    control.eval();candidate.eval()
    try:
        with torch.no_grad():
            a=control(inputs);b=candidate(inputs)
    finally:
        handle.remove()
    if len(calls)!=1 or CONTEXT_KEY in b:
        raise AssertionError('Evaluation must execute the original head exactly once')
    for key in ('pred_logits','pred_boxes'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    # Nonzero synthetic head weights exercise every readout parameter; this is
    # fixture initialization, not an optimizer update or a trained checkpoint.
    with torch.no_grad():
        torch.nn.init.normal_(control.local_readout.output.weight,std=.01)
    candidate.load_state_dict(control.state_dict(),strict=True)
    before={n:p.detach().clone() for n,p in candidate.named_parameters()}
    models=[control.train(),candidate.train()]
    criteria=[P3FeedbackCriterion(build_criterion(plan['criterion_recipe']),mode).to(device)
              for mode in ('aux_only','final_only_feedback')]
    for criterion,model in zip(criteria,models):
        criterion.set_model(model)
        criterion.reset_model()
        if criterion.native_criterion.model is not None:
            raise AssertionError('Native checkpoint reset_model delegation failed')
        criterion.set_model(model)
        if criterion.native_criterion.model is not model:
            raise AssertionError('Native checkpoint set_model delegation failed')
    frozen_parameters=[(n,p) for n,p in candidate.named_parameters() if not p.requires_grad]
    # Independent finite algebra check: exact value and derivative equivalence
    # to the former dot-minus-detach surrogate, without native AMP rounding.
    algebra_shadow=torch.tensor([.25,-.5,2.],device=device,requires_grad=True)
    algebra_g=torch.tensor([2.,-.5,.125],device=device)
    dot=(algebra_shadow*algebra_g).sum()
    legacy=dot-dot.detach()
    custom=ZeroValueSurrogate.apply(algebra_shadow,algebra_g.detach())
    torch.testing.assert_close(custom,legacy,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(custom,algebra_shadow)[0],
        torch.autograd.grad(legacy,algebra_shadow)[0],rtol=0,atol=0)
    tiny_vjp=None
    if amp:
        # Predefined numerical fixture; official/native initial scale unchanged.
        default_scaler=torch.amp.GradScaler('cuda')
        default_scale=float(default_scaler.get_scale())
        if default_scale!=65536.:raise AssertionError('Unexpected native default initial scale')
        leaf=torch.ones(4,device=device,dtype=torch.float16,requires_grad=True)
        tiny_main=(leaf.float()*1e-8).sum()
        raw=torch.autograd.grad(tiny_main,leaf,retain_graph=True)[0]
        scaled=torch.autograd.grad(tiny_main*default_scale,leaf)[0]
        recovered=scaled.float()/default_scale
        if torch.count_nonzero(raw) or not bool(torch.isfinite(recovered).all()) or not bool((recovered>0).all()):
            raise AssertionError('Scaled-VJP underflow fixture failed')
        tiny_vjp=dict(scale=default_scale,unscaled_nonzero=int(torch.count_nonzero(raw)),
            recovered_nonzero=int(torch.count_nonzero(recovered)),recovered_mean=float(recovered.mean()))

    branch_diagnostics={}

    def branch(scaler, expect_overflow):
        scale=float(scaler.get_scale())
        for model,criterion in zip(models,criteria):
            model.zero_grad(set_to_none=True)
            criterion.set_native_scaler(scaler)
        losses=[];outputs=[];features=[]
        for model,criterion in zip(models,criteria):
            feature=[x.detach().clone().requires_grad_(True) for x in inputs]
            features.append(feature)
            torch.manual_seed(101)
            if device.startswith('cuda'):torch.cuda.manual_seed_all(101)
            with torch.autocast(device_type=device.split(':')[0],enabled=amp):
                output=model(feature,targets=targets)
            with torch.autocast(device_type=device.split(':')[0],enabled=False):
                loss=criterion(output,targets,epoch=0,step=0,global_step=0,epoch_step=1)
            outputs.append(output);losses.append(loss)
        for key in losses[0]:
            torch.testing.assert_close(losses[0][key],losses[1][key],rtol=0,atol=0)
        zero=losses[1]['loss_final_only_feedback_zero']
        if float(zero.detach())!=0 or not bool(torch.isfinite(zero)):
            raise AssertionError('Surrogate must be exactly finite zero')
        totals=[sum(loss.values()) for loss in losses]
        if not all(bool(torch.isfinite(total)) for total in totals):
            raise AssertionError('Native and candidate total losses must remain finite')
        torch.testing.assert_close(totals[0],totals[1],rtol=0,atol=0)
        context=outputs[1][CONTEXT_KEY]
        reference=context['reference']
        if bool(context['native_autocast_enabled'])!=amp:
            raise AssertionError('Forward AMP state was not captured faithfully')
        if criteria[1].gradient_scale({'native_autocast_enabled':False})!=1.:
            raise AssertionError('FP32 fallback must not reuse the retained scaler scale')
        # Independently select exactly the four weighted native main losses.
        native_main=sum(losses[1][key] for key in MAIN_KEYS)
        oracle_scaled=torch.autograd.grad(native_main.float()*scale,reference,retain_graph=True)[0]
        oracle_g=(oracle_scaled.float()/scale).detach()
        finite=bool(torch.isfinite(oracle_g).all())
        if finite==expect_overflow:
            raise AssertionError('Unexpected VJP finite state for predefined fixture')
        native_parameters=[p for n,p in candidate.named_parameters()
                           if p.requires_grad and not n.startswith('local_readout.')]
        isolated=torch.autograd.grad(zero,(*native_parameters,*features[1]),retain_graph=True,allow_unused=True)
        if any(value is not None for value in isolated):
            raise AssertionError('Surrogate connected to native parameters or feature inputs')
        head_parameters=tuple(candidate.local_readout.parameters())
        tested_head_scaled=torch.autograd.grad(zero*scale,head_parameters,retain_graph=True)
        expected_head_scaled=None
        gradient_comparison=None
        head_oracle_atol,head_oracle_rtol=(2e-3,2e-3) if amp else (0.,0.)
        if not expect_overflow:
            if not bool(torch.count_nonzero(oracle_g)):
                raise AssertionError('Independent finite VJP must be nonzero')
            oracle_dot=(oracle_g*context['shadow_delta'].float()).sum()
            expected_head_scaled=torch.autograd.grad(oracle_dot*scale,head_parameters,retain_graph=True)
            if not all(bool(torch.isfinite(v).all()) for v in expected_head_scaled) or not any(bool(torch.count_nonzero(v)) for v in expected_head_scaled):
                raise AssertionError('Independent head-gradient oracle must be finite and nonzero')
            # Repeat the SAME oracle graph to expose numerical backward
            # repeatability separately from applied-versus-oracle differences.
            repeat_head_scaled=torch.autograd.grad(oracle_dot*scale,head_parameters,retain_graph=True)
            repeat_g_scaled=torch.autograd.grad(native_main.float()*scale,reference,retain_graph=True)[0]
            repeat_g=(repeat_g_scaled.float()/scale).detach()
            if not bool(torch.isfinite(repeat_g).all()) or not all(bool(torch.isfinite(v).all()) for v in repeat_head_scaled):
                raise AssertionError('Repeated finite oracle became nonfinite')
            def differences(left,right,divisor):
                diff=(left.detach().float()-right.detach().float())/divisor
                return dict(max_abs=float(diff.abs().max()),l2_norm=float(diff.norm()),
                    unequal_elements=int(torch.count_nonzero(diff)),
                    left_l2_norm=float((left.detach().float()/divisor).norm()),
                    right_l2_norm=float((right.detach().float()/divisor).norm()))
            per_parameter=[]
            for (name,_),actual,expected,repeated in zip(candidate.local_readout.named_parameters(),
                    tested_head_scaled,expected_head_scaled,repeat_head_scaled):
                per_parameter.append(dict(parameter=name,
                    applied_vs_oracle=differences(actual,expected,scale),
                    oracle_repeat=differences(expected,repeated,scale)))
            def aggregate(key):
                values=[row[key] for row in per_parameter]
                result=dict(max_abs=max(v['max_abs'] for v in values),
                    l2_norm=sum(v['l2_norm']**2 for v in values)**.5,
                    unequal_elements=sum(v['unequal_elements'] for v in values),
                    left_l2_norm=sum(v['left_l2_norm']**2 for v in values)**.5,
                    right_l2_norm=sum(v['right_l2_norm']**2 for v in values)**.5)
                if result['right_l2_norm']<=0:
                    raise AssertionError('Independent head oracle has zero aggregate norm')
                result['relative_l2']=result['l2_norm']/result['right_l2_norm']
                return result
            gradient_comparison=dict(units='unscaled FP32 gradients',scale=scale,
                applied_vs_oracle=aggregate('applied_vs_oracle'),oracle_repeat=aggregate('oracle_repeat'),
                reference_VJP_repeat=differences(oracle_g,repeat_g,1.),per_parameter=per_parameter,
                head_oracle_tolerance=dict(atol=head_oracle_atol,rtol=head_oracle_rtol),
                policy='AMP uses the pre-existing unscaled mathematical tolerance; CPU and pure algebra remain exact')
            branch_diagnostics['finite128' if amp else 'finite_fp32']=gradient_comparison
            if amp and gradient_comparison['applied_vs_oracle']['relative_l2']>head_oracle_rtol:
                raise AssertionError('Applied head gradient failed prespecified relative L2 tolerance')
            for actual,expected,repeated in zip(tested_head_scaled,expected_head_scaled,repeat_head_scaled):
                torch.testing.assert_close(actual.float()/scale,expected.float()/scale,
                    rtol=head_oracle_rtol,atol=head_oracle_atol)
                torch.testing.assert_close(expected.float()/scale,repeated.float()/scale,
                    rtol=head_oracle_rtol,atol=head_oracle_atol)
            torch.testing.assert_close(oracle_g,repeat_g,rtol=head_oracle_rtol,atol=head_oracle_atol)
        elif all(bool(torch.isfinite(v).all()) for v in tested_head_scaled):
            raise AssertionError('Nonfinite VJP did not reach the real head-gradient graph')
        # Real scaler backward ONLY: no optimizer, unscale_, step, or update.
        for total in totals:scaler.scale(total).backward()
        if float(scaler.get_scale())!=scale:
            raise AssertionError('Preflight changed GradScaler scale')
        head_nonfinite=sum(int((~torch.isfinite(p.grad)).sum()) for p in head_parameters if p.grad is not None)
        largest_native_difference=0.
        atol,rtol=(2e-3,2e-3) if amp else (2e-5,2e-5)
        if expect_overflow:
            if head_nonfinite==0:
                raise AssertionError('Complete scaled backward failed to expose actual nonfinite head gradients')
        else:
            head_expected=dict(zip((n for n,_ in candidate.local_readout.named_parameters()),expected_head_scaled))
            for (name,left),(right_name,right) in zip(control.named_parameters(),candidate.named_parameters()):
                if name!=right_name:raise AssertionError('State key mismatch')
                if left.grad is None or right.grad is None:
                    if left.grad is not right.grad:raise AssertionError('Gradient support differs: '+name)
                    continue
                if not bool(torch.isfinite(right.grad).all()):raise AssertionError('Nonfinite gradient: '+name)
                if name.startswith('local_readout.'):
                    extra=head_expected[name.removeprefix('local_readout.')]
                    torch.testing.assert_close(right.grad.float()/scale,(left.grad.float()+extra.float())/scale,rtol=rtol,atol=atol)
                else:
                    difference=float((right.grad.float()-left.grad.float()).abs().max()/scale)
                    largest_native_difference=max(largest_native_difference,difference)
                    torch.testing.assert_close(right.grad.float()/scale,left.grad.float()/scale,rtol=rtol,atol=atol)
            for left,right in zip(features[0],features[1]):
                torch.testing.assert_close(left.grad.float()/scale,right.grad.float()/scale,rtol=rtol,atol=atol)
        for name,param in candidate.named_parameters():
            torch.testing.assert_close(param,before[name],rtol=0,atol=0)
        for model in models:
            for name,param in model.named_parameters():
                if not param.requires_grad and param.grad is not None:
                    raise AssertionError('Frozen native parameter received a gradient: '+name)
        try:final_main_loss({**{k:zero for k in MAIN_KEYS},'loss_unreviewed':zero})
        except RuntimeError:pass
        else:raise AssertionError('Unknown main loss was accepted')
        return dict(status='PASS',scale=scale,expected_overflow=expect_overflow,
            g_finite_fraction=float(torch.isfinite(oracle_g).float().mean()),
            actual_head_nonfinite_elements=head_nonfinite,loss_finite=True,proxy_scalar_zero=True,
            exact_native_loss_values=True,proxy_shared_gradients_none=True,parameters_unchanged=True,
            independent_scaled_VJP_oracle=True,
            exact_finite_surrogate_head_oracle=not expect_overflow and gradient_comparison['applied_vs_oracle']['unequal_elements']==0,
            finite_surrogate_head_oracle_within_prespecified_tolerance=not expect_overflow,
            gradient_comparison=gradient_comparison,
            native_preclip_max_abs_gradient_difference=None if expect_overflow else largest_native_difference,
            tolerance=dict(atol=atol,rtol=rtol),optimizer_constructions=0,optimizer_steps=0,
            actual_scaler_step_skip_update_tested=False,
            scope='nonfinite gradients exposed for native scaler; skip/backoff NOT empirically exercised here' if expect_overflow else
                  'fixed engineering finite fixture; does not configure the formal training scaler')

    overflow_default=branch(default_scaler,True) if amp else None
    finite_scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=128.)
    try:
        finite_fixture=branch(finite_scaler,False)
    except Exception as exc:
        exc.preflight_diagnostics=branch_diagnostics
        raise
    report=criteria[1].feedback_audit()
    report.update(status='PASS_SYNTHETIC_NATIVE_DECODER_ZERO_UPDATES',device=device,autocast=amp,
        grad_scaler_used=amp,native_default_initial_scale=default_scale if amp else 1.,
        overflow_default=overflow_default,finite128=finite_fixture if amp else None,
        finite_fp32=finite_fixture if not amp else None,
        finite_surrogate_exact_legacy_algebra=True,tiny_FP16_underflow_fixture=tiny_vjp,
        optimizer_constructions=0,optimizer_steps=0,real_data_used=False,
        exact_eval_equivalence=True,exact_native_loss_values=True,proxy_scalar_zero=True,
        proxy_shared_gradients_none=True,parameters_unchanged=True,
        actual_scaler_step_skip_update_tested=False,
        frozen_parameters_grad_none=[name for name,_ in frozen_parameters],
        fixture='2 synthetic 128x128 feature pyramids; native decoder/criterion, nonzero synthetic head',
        elapsed_seconds=time.perf_counter()-started,torch=str(torch.__version__))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=run(args.native_root,args.device,args.amp)
    except Exception as exc:
        report=dict(status='FAILED_PREFLIGHT_ZERO_UPDATES',error_type=type(exc).__name__,
                    error=str(exc),feedback_failure=getattr(exc,'receipt',None),
                    preflight_diagnostics=getattr(exc,'preflight_diagnostics',None),
                    optimizer_constructions=0,optimizer_steps=0,real_data_used=False)
        with output.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
        raise
    with output.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
