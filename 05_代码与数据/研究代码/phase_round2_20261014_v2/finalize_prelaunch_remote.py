"""Reuse successful tests and cost measurement; verify the corrected fused export."""
import json
import os
import pathlib
import subprocess
import sys
import time

phase = pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012')
addons = phase/'artifacts/round2_20261014_v1/addons_v2'
root = addons/'prelaunch_four_images'
assert (addons/'soep_cost_audit.json').is_file()
assert not (root/'receipt.json').exists()
plan = json.loads((addons/'queue_plan.json').read_text())
checkpoint = next(row['checkpoint'] for row in plan['existing_models'] if row['name']=='existing_c')
env = dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
def run(args,label):
    with (root/(label+'.log')).open('xb') as output:
        result = subprocess.run([sys.executable,'-u']+args,cwd=phase/'code',env=env,stdout=output,stderr=subprocess.STDOUT,timeout=600)
    assert result.returncode==0, label+' failed'
run([str(addons/'dump_preds.py'),'--config',str(root/'four_images.yml'),'--checkpoint',checkpoint,
     '--out',str(root/'dump_fused'),'--project-eval'],'dump_fused')
cli_predictions = root/'test_only_eval/pred_bbox.json'
run([str(addons/'analyze_preds.py'),'--annotations',str(root/'four_images.json'),
     '--reference',str(cli_predictions),'--candidate',str(root/'dump_fused/predictions.json'),
     '--out',str(root/'analysis_cli_vs_fused_dump')],'analysis_cli_vs_fused_dump')
first = json.loads(cli_predictions.read_text());second = json.loads((root/'dump_fused/predictions.json').read_text())
assert len(first)==len(second)==1200
def canonical(rows):
    return sorted([(row['image_id'],row['category_id'],row['score'],*row['bbox']) for row in rows])
maximum_difference = 0.
for left,right in zip(canonical(first),canonical(second)):
    assert left[:2]==right[:2]
    maximum_difference=max(maximum_difference,max(abs(a-b) for a,b in zip(left[2:],right[2:])))
assert maximum_difference<.001, ('Fused CLI and fused export disagree',maximum_difference)
report={'status':'passed','scope':'four-image implementation smoke only; not a paper result',
        'prediction_count':1200,'CLI_vs_dump_max_abs_difference':maximum_difference,
        'dtype':'FP32','deploy_fusion':True,'project_evaluator_and_CPU_analysis_completed':True,'finished':time.time()}
with (root/'receipt.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
