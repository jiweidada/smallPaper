"""Read-only backup after heldout analysis; never regenerates predictions."""
import concurrent.futures,datetime,json
from control_isolated_v3 import ctl,ROOTS,BASE,execute
from transfer_probe import get,safe_extract
from seal_v3 import verify_partition,write_json
def one(a,pw):
 c=ctl.connect(ctl.ACCOUNTS[a]['port'],pw)
 try:
  program='''import pathlib,json,hashlib,tarfile,sys
root=pathlib.Path(PROBE_ROOT);sys.path.insert(0,str(root/'v3'))
from seal_v3 import verify_partition,sha
folder=root/f'isolated_results_seed{SEED}'
assert json.loads((folder/'analysis_heldout_v3/complete.json').read_text())['state']=='complete'
done=json.loads((folder/'complete.json').read_text())
for group in ['smoke','development','heldout']:verify_partition(folder/group,done['partition_seal_sha256'][group])
archive=root/f'frozen_predictions_v3_seed{SEED}.tar.gz'
if not archive.exists():
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for group in ['smoke','development','heldout']:
   for p in sorted((folder/group).iterdir()):t.add(p,arcname=str(p.relative_to(folder)),recursive=False)
  for name in ['complete.json','generation_seals.jsonl']:t.add(folder/name,arcname=name,recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}),flush=True)
'''
  meta=execute(c,a,program,180);path=BASE/(a+'_frozen_predictions.tar.gz')
  get(c.open_sftp(),meta['path'],path,meta['bytes'],meta['sha256'])
  dest=BASE/a/'frozen_predictions';safe_extract(path,dest)
  anchor=json.loads((BASE/'generation_anchor'/a/'complete.json').read_text())
  for group in ['smoke','development','heldout']:verify_partition(dest/group,anchor['partition_seal_sha256'][group])
  value={'account':a,'state':'all_frozen_prediction_bytes_match_generation_seals','archive':meta,'local':str(dest),'client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  write_json(BASE/a/'frozen_prediction_backup_verified.json',value);return value
 finally:c.close()
if __name__=='__main__':
 pw=json.loads(ctl.scheduled_monitor_v4.credentials_bytes())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={a:pool.submit(one,a,pw) for a in ['A','B']};results={a:f.result() for a,f in fs.items()}
 finally:pw.clear()
 print(json.dumps(results,ensure_ascii=False,indent=2))
