import json
import os
import pathlib
import subprocess
import sys
import time

phase=pathlib.Path('/25085400120/DEIM/experiments/phase_mechanism_20261012')
addons=phase/'artifacts/round2_20261014_v1/addons_v2';root=addons/'prelaunch_four_images'
assert not (root/'receipt.json').exists()
(root/'val_test_only_eval').symlink_to(root/'test_only_eval',target_is_directory=True)
plan=json.loads((addons/'queue_plan.json').read_text())
checkpoint=next(row['checkpoint'] for row in plan['existing_models'] if row['name']=='existing_c')
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
def run(args,label):
    with (root/(label+'.log')).open('xb') as output:
        result=subprocess.run([sys.executable,'-u']+args,cwd=phase/'code',env=env,stdout=output,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,label+' failed'
run([str(addons/'dump_preds.py'),'--config',str(root/'four_images.yml'),'--checkpoint',checkpoint,
     '--out',str(root/'val'),'--project-eval'],'CLI_reuse')
run([str(addons/'analyze_preds.py'),'--annotations',str(root/'four_images.json'),
     '--reference',str(root/'test_only_eval/pred_bbox.json'),'--candidate',str(root/'val/predictions.json'),
     '--out',str(root/'analysis_cli_reuse')],'analysis_cli_reuse')
analysis=json.loads((root/'analysis_cli_reuse/analysis.json').read_text())
reference=analysis['original_coordinate_appendix']['reference']['coco_original_coordinate']['metrics_raw_0_to_1']
candidate=analysis['original_coordinate_appendix']['candidate']['coco_original_coordinate']['metrics_raw_0_to_1']
saved=json.loads((root/'val/project_eval.json').read_text())['metrics_raw_0_to_1']
maximum=max(abs(a-b) for a,b in zip(reference,saved))
assert maximum<1e-8 and reference==candidate
report={'status':'passed','scope':'four-image implementation smoke only; not a paper result',
        'prediction_count':1200,'CLI_reused_without_new_inference':True,'all_12_metrics_CPU_vs_project_max_difference':maximum,
        'dtype':'FP32','deploy_fusion':True,'project_evaluator_and_CPU_analysis_completed':True,'finished':time.time(),
        'separate_inference_readback':'Retained fused/unfused diagnostic differences; these are not mixed into main tables'}
with (root/'receipt.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
