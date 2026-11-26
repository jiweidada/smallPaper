from pathlib import Path
import datetime,hashlib,json
ROOT=Path(__file__).resolve().parent
frozen=json.loads((ROOT/'server_receipts/code_frozen_at_launch.json').read_bytes())
training=['study.py','run_train.py','run_queue.py','prepare_models.py']
for name in training:assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==frozen['files'][name],name
changes=[]
for name in ['evaluate_fixed.py','evaluator_fixture.py']:
    old=ROOT/'source_reference'/(Path(name).stem+'_before_native_clipping.py')
    before=hashlib.sha256(old.read_bytes()).hexdigest();after=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    assert before==frozen['files'][name],name
    changes.append({'file':name,'frozen_before_sha256':before,'corrected_sha256':after})
record={'client_date':'2026-10-29','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage':'D0 training; before any fixed-checkpoint scientific diagnostic evaluation',
        'reason':'Use native scale_boxes valid-image clipping for both original-coordinate AP and input-area AP; reproject the same clipped predictions into the input frame.',
        'training_code_unchanged_sha256_verified':training,'training_recipe_or_budget_changed':False,
        'scientific_results_seen_before_change':False,'changes':changes,'artificial_border_fixture':'PASS on local CPU',
        'previous_sources_preserved_in':'source_reference/*_before_native_clipping.py'}
(ROOT/'evaluator_amendment_v1.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))
