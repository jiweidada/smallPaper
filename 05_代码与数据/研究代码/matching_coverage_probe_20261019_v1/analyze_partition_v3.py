"""Partition-scoped analysis; heldout cannot load before code-freeze release."""
import argparse,datetime,json,pathlib,sys,traceback
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'v3'),str(ROOT/'probe'),str(ROOT/'extension_v2')]
from seal_v3 import sha,write_json,verify_partition,heldout_guard,verify_files
def main(seed,group):
 results=ROOT/f'isolated_results_seed{seed}'
 complete=json.loads((results/'complete.json').read_text());assert complete['state']=='complete'
 if group=='heldout':manifest=heldout_guard(ROOT,seed)
 else:
  assert not (ROOT/'analysis_freeze_v3.json').exists(),'Development code already frozen; do not overwrite development results'
  manifest=verify_partition(results/group,complete['partition_seal_sha256'][group])
 out=results/('analysis_'+group+'_v3');out.mkdir(exist_ok=False)
 import torch
 from diagnostic_extension_v2 import analyze
 torch.set_num_threads(4)
 y=json.loads((ROOT/'baseline_recipes.json').read_text())[str(seed)];done=[]
 for j,item in enumerate(manifest['files']):
  # Only the selected physical partition is ever deserialized.
  path=results/group/item['relative'];assert sha(path)==item['sha256']
  value=torch.load(path,map_location='cpu',weights_only=False);assert value['group']==group
  report=analyze(value['outputs'],value['targets'],y,check_grad=(j==0))
  report.update(tag=value['tag'],group=group,seed=seed,source_ids=item['source_ids'],prediction_sha256=item['sha256'])
  dest=out/(value['tag']+'_analysis.json');write_json(dest,report)
  done.append({'relative':dest.name,'bytes':dest.stat().st_size,'sha256':sha(dest),'batch':item['batch'],'source_ids':item['source_ids']})
  print(json.dumps({'phase':'analysis','group':group,'seed':seed,'finished_batches':len(done)}),flush=True)
 verify_partition(results/group,complete['partition_seal_sha256'][group])
 write_json(out/'complete.json',{'state':'complete','seed':seed,'group':group,'batches':8,'formal_inputs':32,'files':done,'input_seal_sha256':complete['partition_seal_sha256'][group],'all_DN_and_DDF_checks_passed':True,'optimizer_steps':0,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True);p.add_argument('--group',choices=['development','heldout'],required=True);a=p.parse_args()
 try:main(a.seed,a.group)
 except BaseException as e:
  write_json(ROOT/f'analysis_{a.group}_seed{a.seed}_failure_v3.json',{'error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
