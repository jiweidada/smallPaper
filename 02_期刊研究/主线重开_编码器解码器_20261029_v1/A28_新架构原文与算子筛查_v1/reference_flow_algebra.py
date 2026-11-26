"""Synthetic affine-flow checks, not a trained visual encoder experiment.

Fractional affine flow is classical. These checks establish no novelty or AP.
The only proposed research direction is checkpoint-preserving resolution policy.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent

def fractional_input_ratio(log_a,rho):
    a=np.asarray(log_a,dtype=np.float64)
    assert rho>0 and np.isfinite(a).all() and (a<=0).all()
    if rho==1:return np.ones_like(a)
    return np.divide(np.expm1(rho*a),np.expm1(a),out=np.full_like(a,rho),where=a!=0)

def reference_preserving_coefficients(log_a,b,rho):
    if rho==1:return np.asarray(log_a).copy(),np.asarray(b).copy()
    return rho*np.asarray(log_a),fractional_input_ratio(log_a,rho)*np.asarray(b)

def main():
    assert not (ROOT/'algebra_result.json').exists()
    p=json.loads((ROOT/'algebra_protocol_FROZEN.json').read_text(encoding='utf-8'))
    assert p['status']=='FROZEN_SYNTHETIC_AFFINE_FLOW_ONLY'
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['script_sha256']
    rng=np.random.default_rng(p['seed']);n=p['random_cases']
    log_a=-np.exp(rng.uniform(np.log(1e-8),np.log(50),n));log_a[0]=0
    b=rng.normal(size=n);h0=rng.normal(size=n)
    native=np.exp(log_a)*h0+b
    a1,b1=reference_preserving_coefficients(log_a,b,1)
    assert np.array_equal(a1,log_a) and np.array_equal(b1,b)
    rows=[]
    for m in p['refinement_factors']:
        rho=1/m;aa,bb=reference_preserving_coefficients(log_a,b,rho)
        fixed=h0.copy();naive=h0.copy()
        for _ in range(m):
            fixed=np.exp(aa)*fixed+bb
            naive=np.exp(log_a/m)*naive+b/m
        error=float(np.max(np.abs(fixed-native)))
        assert error<=p['maximum_absolute_error'],(m,error)
        rows.append({'m':m,'reference_preserving_max_abs_error':error,
          'naive_step_rescale_max_abs_difference':float(np.max(np.abs(naive-native))),
          'naive_step_rescale_RMS_difference':float(np.sqrt(np.mean((naive-native)**2)))})
    # For N=1, the correction can be folded into per-channel input u.
    delta=np.exp(rng.uniform(-6,2,n));rate=-np.exp(rng.uniform(-4,2,n))
    u=rng.normal(size=n);B=rng.normal(size=n);rho=.25
    g=fractional_input_ratio(delta*rate,rho)/rho
    desired=fractional_input_ratio(delta*rate,rho)*(delta*B*u)
    folded=(rho*delta)*B*(u*g)
    folding_error=float(np.max(np.abs(desired-folded)))
    assert folding_error<=p['maximum_absolute_error']
    # Different state poles generally require different gains; one input scalar
    # cannot implement this exact correction for N>1 in the unchanged kernel.
    gains=fractional_input_ratio(np.array([-.1,-2.0]),rho)/rho
    assert gains[0]!=gains[1]
    np.savez_compressed(ROOT/'algebra_fixture.npz',log_a=log_a,b=b,h0=h0,native=native,
                        delta=delta,rate=rate,u=u,B=B)
    result={'status':'PASS_CONDITIONAL_AFFINE_FLOW_IDENTITY_ONLY','seed':p['seed'],'random_cases':n,
      'reference_coefficients_bitwise_preserved_at_rho1':True,'refinement_rows':rows,
      'single_state_input_folding_max_abs_error':folding_error,
      'two_state_counterexample_distinct_gains':gains.tolist(),
      'native_encoder_function_executed':False,'whole_detector_calls':0,'trained_model_calls':0,
      'new_optimizer_updates':0,'real_images_used':0,'novelty_cleared':False,'mainline_validated':False,
      'limit':'Constant coefficients within each refined block, scalar/diagonal affine recurrence. Not 2D spatial equivariance, arbitrary high-resolution inputs, trained-encoder fidelity, AP, or GPU performance.',
      'protocol_sha256':hashlib.sha256((ROOT/'algebra_protocol_FROZEN.json').read_bytes()).hexdigest(),
      'fixture_sha256':hashlib.sha256((ROOT/'algebra_fixture.npz').read_bytes()).hexdigest()}
    (ROOT/'algebra_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
