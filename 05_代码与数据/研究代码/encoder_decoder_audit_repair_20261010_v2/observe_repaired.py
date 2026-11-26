from pathlib import Path
import json,hashlib,datetime,re,sys,base64,textwrap
seed=int(sys.argv[1]);prefix={1:'/25085400120',2:'/24085403014'}[seed];version={1:4,2:3}[seed]
experiment=Path(prefix)/f'DEIM/experiments/small_object_encoder_decoder12_20261010_v{version}'
root=experiment/f'seed{seed}_fine';run=root/'run'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={'seed':seed,'version':version,'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'fixed_epoch_limit':12}
for name in ('controller_start.json','candidate_start.json'):
 p=root/name
 if p.exists():
  s=read(p);pid=s.get('child_pid',s.get('pid'));ticks=s.get('child_start_ticks',s.get('start_ticks'));proc=Path(f'/proc/{pid}/stat')
  out[name]={'pid':pid,'start_ticks':ticks,'alive_same_process':proc.exists() and proc.read_text().split()[21]==str(ticks)}
for name in ('baseline_completion.json','candidate_completion.json','controller_completion.json','controller_failure.json'):
 p=root/name
 if p.exists():out[name]={k:v for k,v in read(p).items() if k!='traceback'}
p=root/'baseline_reuse.json'
if p.exists():out['baseline_reuse']={k:v for k,v in read(p).items() if k!='pinned_files'}
p=run/'first100_training_inputs.jsonl';old=Path(prefix)/f'DEIM/experiments/fine_residual_screen24_20261009_v2/runs/baseline_seed{seed}'
if p.exists():
 actual=[json.loads(s) for s in p.read_text().splitlines()];ref=[json.loads(s) for s in (old/'first100_training_inputs.jsonl').read_text().splitlines()]
 out['first100_parity']={'recorded_rows':len(actual),'all_available_rows_equal':actual==ref[:len(actual)],'all100_equal':len(actual)==100 and actual==ref}
p=run/'training_audit_live.json'
if p.exists():out['last_observed_lower_bound']=read(p)
p=run/'training_audit_epochs.jsonl'
if p.exists():
 rows=[json.loads(s) for s in p.read_text().splitlines()];a=rows[-1]
 out['latest_epoch_audit']={k:a[k] for k in ('epoch_zero_based','attempts_total','actual_optimizer_steps_total','EMA_updates','attempts_without_optimizer_step')}
 g=a['added_changed_elements_from_initial']
 out['latest_epoch_audit']['changed_elements_by_group']={s:sum(v for k,v in g.items() if k.startswith(s)) for s in ('encoder.geometry_field.','decoder.enc_bbox_head.geometry_proposal.','decoder.decoder.geometry_layers.','decoder.decoder.geometry_attention.')}
 out['latest_epoch_audit']['all_four_groups_moved']=all(v>0 for v in out['latest_epoch_audit']['changed_elements_by_group'].values())
 out['latest_epoch_audit']['native_EMA_count_matches_attempts']=a['EMA_updates']==a['attempts_total']
p=root/'candidate.log'
if p.exists():
 t=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',p.read_text(errors='replace'))
 progress=re.findall(r'Epoch: (\d+)/132\s+\[\s*(\d+)/1617\]',t)
 if progress:out['latest_progress']={'epoch_one_based':int(progress[-1][0])+1,'logged_batch':int(progress[-1][1]),'batches_per_epoch':1617}
p=run/'log.txt';out['completed_validation_epochs']=0
if p.exists():
 rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
 if rows:
  a=rows[-1];b=[json.loads(s) for s in (old/'log.txt').read_text().splitlines() if json.loads(s)['epoch']==a['epoch']][0]
  out['completed_validation_epochs']=int(a['epoch'])+1
  out['latest_same_epoch_metrics']={}
  for name,idx in [('AP',0),('APs',3),('AP75',2)]:
   c=float(a['test_coco_eval_bbox'][idx])*100;v=float(b['test_coco_eval_bbox'][idx])*100
   out['latest_same_epoch_metrics'][name]={'candidate_percent':c,'baseline_percent':v,'delta_pp':c-v}
p=run/'training_completion.json'
if p.exists():out['fixed12_complete']=read(p)
print('\nB64_BEGIN\n'+'\n'.join(textwrap.wrap(base64.b64encode(json.dumps(out).encode()).decode(),76))+'\nB64_END')
