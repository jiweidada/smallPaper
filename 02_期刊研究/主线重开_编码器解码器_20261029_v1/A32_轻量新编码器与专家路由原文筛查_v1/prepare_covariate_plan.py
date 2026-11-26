from pathlib import Path
import datetime,hashlib,json,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((D/'geometry_exposure_plan_FROZEN.json').read_text(encoding='utf-8'))
files={k:Path(v['path']) for k,v in p['inputs'].items() if k!='worker'}
files.update(worker=D/'verify_geometry_and_covariates.py',primary_result=D/'geometry_result.json',primary_rows=D/'geometry_per_GT.json',primary_bootstrap=D/'geometry_bootstrap.npz')
plan=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='POSTHOC_SENSITIVITY_FROZEN_BEFORE_ITS_OUTCOMES',scope='Chosen after positive initial geometry association to address concrete residual size/occlusion/truncation confounding. Exploratory sensitivity,not independent confirmation,causality,or an algorithm promotion gate.',inputs={k:dict(path=str(v),sha256=sha(v)) for k,v in files.items()},primary_phase=[0,0],finer_area_bins=[0,16,36,64,100,144,196,256,400,576,784,1024],modes=['fine_area','fine_area_occlusion_truncation'],bootstrap_seed=432104,bootstrap_draws=1000,annotation_field_source='https://aiskyeye.com/evaluate/results-format-2024/',annotation_fields=dict(truncation='column7',occlusion='column8'),new_model_calls=0,new_optimizer_updates=0,no_algorithm_or_mainline_approval=True)
with (D/'geometry_covariate_plan_FROZEN.json').open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
print(plan['status'])
