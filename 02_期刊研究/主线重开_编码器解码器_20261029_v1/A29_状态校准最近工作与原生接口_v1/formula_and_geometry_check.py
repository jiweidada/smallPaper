"""Analytic audit only: classical affine flow and 2D distance counterexample."""
from pathlib import Path
import hashlib,itertools,json,math,sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
def main():
    path=ROOT/'formula_and_geometry_protocol_FROZEN.json'
    cfg=json.loads(path.read_text(encoding='utf-8'));c=cfg['affine_cases'];rows=[]
    assert not (ROOT/'formula_and_geometry_result.json').exists()
    for a,rho in itertools.product(c['a'],c['rho']):
        b,h=c['b'],c['h0'];v=a*b/math.expm1(a)
        p=math.exp(rho*a)*h+math.expm1(rho*a)/math.expm1(a)*b
        z=math.exp(rho*a)*h+math.expm1(rho*a)/a*v
        delta=.2;A=a/delta
        rows.append({'a':a,'rho':rho,'P':p,'continuous_generator_exact_flow':z,'abs_error':abs(p-z),
          'step_scaled':math.exp(rho*a)*h+rho*b,
          'linear_A_scaled':math.exp(rho*a)*h+b,
          'published_log_A_scaled_form':math.exp(delta*(-(-A)**rho))*h+b})
    g=cfg['geometry_cases'];geometry=[]
    for H,r in itertools.product(g['square_reference_width'],g['linear_scale']):
        a=g['a'];rho=1/(r*r)
        reference=[math.exp(a),math.exp(a*H)]
        calibrated=[math.exp(a*r*rho),math.exp(a*r*r*H*rho)]
        geometry.append({'H':H,'r':r,'rho_area':rho,'reference_distances':[1,H],
          'mapped_distances':[r,r*r*H],'effective_calibrated_distances':[r*rho,r*r*H*rho],
          'reference_retention':reference,'calibrated_retention':calibrated,
          'within_row_abs_diff':abs(reference[0]-calibrated[0]),'column_abs_diff':abs(reference[1]-calibrated[1]),
          'rho_needed_for_within_row':1/r,'rho_needed_for_column':1/(r*r)})
    assert len(rows)==c['case_count'] and max(x['abs_error'] for x in rows)<=c['tolerance']
    assert len(geometry)==g['case_count'] and all(x['within_row_abs_diff']>1e-6 and x['column_abs_diff']<1e-13 for x in geometry)
    out={'phase':'A29','status':'CLASSICAL_FLOW_IDENTITY_AND_AREA_POLICY_GEOMETRY_COUNTEREXAMPLE_CONFIRMED',
      'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'affine_cases':rows,'geometry_cases':geometry,
      'max_affine_identity_error':max(x['abs_error'] for x in rows),'model_or_native_functions_executed':False,
      'scope':'Constant coefficients and route-distance analysis only. Not trained-model invariance, defect, novelty or AP evidence.'}
    (ROOT/'formula_and_geometry_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':out['status'],'affine_cases':len(rows),'geometry_cases':len(geometry),'max_affine_identity_error':out['max_affine_identity_error'],'native_model_calls':0}))
if __name__=='__main__':main()
