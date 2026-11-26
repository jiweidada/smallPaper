"""Repair only pause confirmation; adopt already stopped processes, never resume training."""
import datetime,hashlib,json,os,pathlib,signal,time,zipfile
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix(p.suffix+'.next');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ident(entry,root):
 p=pathlib.Path('/proc')/str(entry['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert s[19]==str(entry['start_ticks']) and s[0]!='Z'
 args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 assert any(x.startswith(str(root/'code')+'/') for x in args)
 return s[0]
def rows(path):
 r=[]
 for s in path.read_text().splitlines():
  try:v=json.loads(s)
  except ValueError:continue
  if 'test_uavdt_eval_bbox' in v:r.append(v)
 assert [v['epoch'] for v in r]==list(range(len(r)));return r
def main(root):
 dest=root/'pause_user_20261019_v1';receipt=read(dest/'receipt.json')
 assert ident(receipt['controller'],root)=='T'
 receipt.update(confirmation_repair='v1 used a 50ms confirmation; v2 waits up to30s for group-stop completion; no training resumed',repair_started_utc=now(),state='waiting_for_epoch_boundaries')
 write(dest/'receipt.json',receipt);start=time.monotonic()
 while any(not j['state'].startswith('stopped_after_saved_epoch') for j in receipt['jobs']):
  assert time.monotonic()-start<7200
  for j in receipt['jobs']:
   if j['state'].startswith('stopped_after_saved_epoch'):assert ident(j['identity'],root)=='T';continue
   state=ident(j['identity'],root);r=rows(root/'outputs'/j['job']/'log.txt')
   if r[-1]['epoch']<j['target_epoch']:
    assert state!='T','Stopped before desired new saved boundary';continue
   if state!='T':
    assert os.getpgid(j['identity']['pid'])==j['identity']['pid'];os.killpg(j['identity']['pid'],signal.SIGSTOP)
    deadline=time.monotonic()+30
    while ident(j['identity'],root)!='T':assert time.monotonic()<deadline;time.sleep(.05)
   j.update(state='stopped_after_saved_epoch',saved_epoch=r[-1]['epoch'],completed_epochs=r[-1]['epoch']+1,paused_utc=now(),metrics=r[-1]['test_uavdt_eval_bbox'])
   p=root/'jobs'/(j['job']+'.json');v=read(p);v.update(state='paused_saved_epoch_gpu_held',user_pause=True,paused_completed_epochs=j['completed_epochs'],pause_receipt=str(dest/'receipt.json'));write(p,v)
   write(dest/'receipt.json',receipt)
  time.sleep(.1)
 import torch
 for j in receipt['jobs']:
  cp=root/'outputs'/j['job']/'last.pth'
  with zipfile.ZipFile(cp) as z:assert z.testzip() is None
  v=torch.load(cp,map_location='cpu',weights_only=False);assert v['last_epoch']==j['saved_epoch']
  assert {'model','ema','optimizer','scaler','lr_warmup_scheduler'}.issubset(v)
  j.update(checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,checkpoint_keys=sorted(v),model_entries=len(v['model']),optimizer_state_entries=len(v['optimizer']['state']),ema_updates=v['ema']['updates'],zip_crc_passed=True);del v
 targets=[root/name for name in ['plan.json','frozen_manifest.json','queue.lock','launch_receipt.json','final_acceptance.json','capacity_receipt.json','release_after_preflight.json']]+list((root/'logs').glob('*.log'))
 for j in receipt['jobs']:
  o=root/'outputs'/j['job'];targets+=list(o.glob('*.pth'))+[o/'log.txt',o/'args.json',root/'jobs'/(j['job']+'.json')]
 targets+=list((root/'code/configs').rglob('*.yml'))+list((root/'code/configs').rglob('*.yaml'))
 files=[{'relative':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(targets)) if p.is_file()]
 for f in read(root/'frozen_manifest.json')['files']:assert sha(root/f['relative'])==f['sha256']
 write(dest/'backup_manifest.json',{'root':str(root),'created_utc':now(),'files':files,'bytes':sum(v['bytes'] for v in files)})
 receipt.update(state='all_saved_paused_waiting_local_backup',finished_utc=now(),backup_manifest_sha256=sha(dest/'backup_manifest.json'),frozen_manifest_verified=True);write(dest/'receipt.json',receipt)
 q=read(root/'queue_state.json');q.update(state='paused_by_user_waiting_local_backup',pause_receipt=str(dest/'receipt.json'));write(root/'queue_state.json',q)
 print(json.dumps({'state':receipt['state'],'jobs':[(j['job'],j['completed_epochs']) for j in receipt['jobs']]}),flush=True)
if __name__=='__main__':
 import sys
 root=pathlib.Path(sys.argv[1])
 try:main(root)
 except BaseException as e:write(root/'pause_user_20261019_v1/failure.json',{'time':now(),'error':type(e).__name__+': '+str(e),'no_automatic_training_retry':True});raise
