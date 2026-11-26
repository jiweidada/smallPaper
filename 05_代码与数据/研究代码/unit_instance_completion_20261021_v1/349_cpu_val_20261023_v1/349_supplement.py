"""Read-only aggregation of frozen formal counts and actual G training logs."""
import json,sys,importlib
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'348_cpu_training_20261023_v1'))
C=importlib.import_module('348_common')
d=C.read(HERE/'349_val548指标与G校准.json');summary={};updates={};ap=[]
training=C.read(C.HERE/'348_训练_result.json')
for seed in (1,2):
 counts=C.read(HERE/f'349_seed{seed}_正式逐图计数.json');summary[str(seed)]={}
 for arm,images in counts.items():
  summary[str(seed)][arm]={}
  for threshold in ('0.5','0.75'):
   sample=images[0]['counts'][threshold]
   summary[str(seed)][arm][threshold]={k:sum(i['counts'][threshold][k] for i in images) for k,v in sample.items() if isinstance(v,int)}
 r=next(r for r in training['runs'] if r['run_id']==f'G_seed{seed}')
 logs=C.read(Path(r['run_dir'])/'347_steps.json')
 updates[str(seed)]={'classes':list(range(10)),
  'steps_with_nonzero_clipped_gradient':[sum(x['kappa_grad_after_clip'][c]!=0 for x in logs) for c in range(10)],
  'steps_with_actual_parameter_change':[sum(x['kappa_after'][c]!=x['kappa_before'][c] for x in logs) for c in range(10)],
  'kappa_initial':logs[0]['kappa_before'],'kappa_final':logs[-1]['kappa_after'],
  'G_delta':d['G_effect'][str(seed)]['delta'],'verdict':d['G_effect'][str(seed)]['verdict']}
 for control in ('baseline','N','S','G'):
  delta=d['metrics'][str(seed)]['R']['AP']-d['metrics'][str(seed)][control]['AP']
  ap.append({'seed':seed,'control':control,'R_minus_control_AP':delta,'negative':delta<0})
C.write(HERE/'349_正式计数双阈值与G更新披露.json',{'status':'已验证','formal_counts_totals':summary,'G_actual_updates':updates,'overall_AP_all_differences':ap,'source_sha256':C.sha(__file__),'note':'0.75栏B仍用固定Geo0(.5)主资格组；tiny_complete_Geo075为独立另报组。'})
print('双IoU阈值正式计数与G逐类实际更新披露已保存')
