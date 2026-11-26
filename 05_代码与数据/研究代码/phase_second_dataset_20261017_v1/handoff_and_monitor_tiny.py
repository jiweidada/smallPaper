"""Guarded two-account continuation and read-only monitoring; pinned SSH, no secrets in logs."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
from zoneinfo import ZoneInfo

from scoped_data_transfer import connect, PYTHONS, scheduled_monitor_v4

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1] / 'data/07_第二数据集_20261017/每半小时跟进'
SELECTION_POLICY = DEST.parent / '第二数据集选择状态.json'
ACCOUNTS = {
    'A': {'port': 30512, 'old': '/25085400120/DEIM/experiments/phase_mechanism_20261012'},
    'B': {'port': 30059, 'old': '/24085403014/experiments/phase_mechanism_20261012'},
}
for entry in ACCOUNTS.values():
    entry['root'] = str(Path(entry['old']).parent / 'phase_second_dataset_tinyperson_20261017_v1').replace('\\', '/')

REMOTE = r'''
import datetime,hashlib,json,os,pathlib,re,subprocess
P=pathlib.Path;cfg=CONFIG;root=P(cfg['root']);old=P(cfg['old'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tail(p,size=30000):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-size));return f.read().decode(errors='replace')
def process(pid):
 try:
  p=P('/proc')/str(pid);args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
  if not any(str(root) in a for a in args):return None
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  env={}
  for part in (p/'environ').read_bytes().split(b'\0'):
   key,_,value=part.partition(b'=')
   if key.decode(errors='replace') in ['OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED','PYTHONDONTWRITEBYTECODE']:
    env[key.decode()]=value.decode(errors='replace')
  return {'pid':int(pid),'ppid':int(stat[1]),'start_ticks':int(stat[19]),'state':stat[0],'argv':args,'environment':env}
 except (OSError,ValueError,IndexError):return None
stage=old/'artifacts/round2_20261014_v1';pointer=read(stage/'current_execution.json')
execution=pointer['execution_directory'];assert execution in ['execution_v3','execution_v4','execution_v5']
qpath=stage/execution/'queue_state.json';oldq=read(qpath)
result={'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'account':cfg['account'],
 'old_queue':{'path':str(qpath),'sha256':sha(qpath),'value':oldq},'root':str(root),
 'acceptance':read(root/'final_acceptance.json'),'plan':read(root/'plan.json'),'queue':read(root/'queue_state.json'),
 'launch_receipt':read(root/'launch_receipt.json'),'launch_lock_exists':(root/'launch.lock').exists(),'jobs':{}}
if result['acceptance']:
 result['frozen_manifest_sha256']=sha(root/'frozen_manifest.json')
 result['plan_sha256']=sha(root/'plan.json')
q=result['queue'] or {};controller=process(q.get('pid',0))
result['controller']=controller
for arm in result['plan']['jobs']:
 job=arm+'_full96_seed'+str(result['plan']['seed']);out=root/'outputs'/job;status=read(root/'jobs'/(job+'.json'))
 item={'status':status,'completed_epochs':0,'latest_evaluation':None,'process':None,'independent':{}}
 if status:item['process']=process(status.get('pid',0))
 if item['process']:
  argv=item['process']['argv'];expected=str(out)
  actual=argv[argv.index('--output-dir')+1] if '--output-dir' in argv else None
  for option in ['--resume','-r']:
   if option in argv:actual=str(P(argv[argv.index(option)+1]).parent)
  item.update(expected_output_directory=expected,runtime_output_directory=actual,runtime_save_path_verified=actual==expected)
 logfile=out/'log.txt'
 if logfile.is_file():
  rows=[]
  for line in logfile.read_text().splitlines():
   try:row=json.loads(line)
   except ValueError:continue
   if 'test_tiny_eval_bbox' in row:rows.append(row)
  item.update(log_bytes=logfile.stat().st_size,log_mtime=logfile.stat().st_mtime,completed_epochs=len(rows))
  if rows:
   r=rows[-1];v=r['test_tiny_eval_bbox'];assert len(v)==42
   item['latest_evaluation']={'epoch_zero_based':r['epoch'],'metrics_raw_0_to_1':dict(zip(result['plan']['metric_names'],v))}
 console=tail(root/'logs'/(job+'.log'));steps=[]
 for line in console.splitlines():
  m=re.search(r'Epoch:\s*(\d+)/(\d+)\s+\[\s*(\d+)/(\d+)\]',line)
  if m:
   sec=re.search(r'\btime:\s*([0-9.]+)',line)
   steps.append({'epoch_zero_based':int(m[1]),'batch':int(m[3]),'total_batches':int(m[4]),'seconds_per_step':float(sec[1]) if sec else None})
 item['recent_steps']=steps[-10:]
 if status and status.get('state')=='failed':item['failure_tail']='\n'.join(console.splitlines()[-50:])
 for split in ['val','test']:
  metric=root/'results'/job/split/'metrics.json'
  if metric.is_file():item['independent'][split]=read(metric)
 result['jobs'][job]=item
def command(args):
 try:
  p=subprocess.run(args,text=True,capture_output=True,timeout=20)
  return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr[:500]}
 except (OSError,subprocess.TimeoutExpired) as exc:return {'returncode':1,'error':type(exc).__name__}
result['gpu']=command(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
result['gpu_compute']=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
result['owned_gpu_processes']=[]
for line in result['gpu_compute'].get('stdout','').splitlines():
 pid=line.split(',',1)[0].strip()
 if pid.isdigit():
  owned=process(int(pid))
  if owned:result['owned_gpu_processes'].append(owned)
if q.get('state')=='failed':
 result['controller_failure_tail']='\n'.join(tail(root/'logs/controller.log').splitlines()[-50:])
 result['gpu_preflight_failure_tail']='\n'.join(tail(root/'logs/gpu_preflight.log').splitlines()[-50:])
print(json.dumps(result))
'''

LAUNCH = r'''
import datetime,hashlib,json,os,pathlib,subprocess,sys
P=pathlib.Path;cfg=CONFIG;proof=PROOF;root=P(cfg['root']);old=P(cfg['old'])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if (root/'launch.lock').exists():
 print(json.dumps({'account':cfg['account'],'state':'previous_launch_attempt_preserved_no_retry'}));sys.exit(0)
stage=old/'artifacts/round2_20261014_v1';execution=read(stage/'current_execution.json')['execution_directory']
assert execution=='execution_v5';qpath=stage/execution/'queue_state.json'
assert read(qpath)['state']=='complete' and sha(qpath)==proof[cfg['account']]['sha256']
assert all(p['value']['state']=='complete' for p in proof.values())
accept=read(root/'final_acceptance.json');assert accept['status']=='ready_waiting_for_both_original_queues'
assert sha(root/'frozen_manifest.json')==accept['frozen_manifest_sha256']
assert sha(root/'plan.json')==accept['plan_sha256']
for entry in read(root/'frozen_manifest.json')['files']:
 path=root/entry['relative'];assert path.is_relative_to(root) and sha(path)==entry['sha256'],str(path)
plan=read(root/'plan.json');assert sha(P(plan['initialization']))==plan['initialization_sha256']
portable=P(plan['data_root'])/'portable_file_manifest.json'
if not portable.is_file():portable=P(plan['data_root']).parent/'TinyPerson_v1_portable_file_manifest.json'
manifest=read(portable)
assert sha(portable)=='9cb751e1fe3680b5816d578dbc99859bdaa1f5fe6d6b9e4dfdffb511b5b4ae9c'
assert isinstance(manifest,list) and len(manifest)==1541
for entry in manifest:
 path=P(plan['data_root'])/entry['relative'];assert path.is_relative_to(P(plan['data_root']))
 assert sha(path)==entry['sha256'],'Dataset content changed: '+str(path)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=20)
assert gpu.returncode==0 and not any(x.strip().isdigit() for x in gpu.stdout.splitlines()),'GPU unavailable or occupied'
assert not (root/'queue.lock').exists() and not (root/'queue_state.json').exists()
assert not any((root/'outputs').iterdir()) and not any((root/'jobs').iterdir())
release={'both_round2_queues_complete':True,'old_queue_completion_proof':proof,'plan_sha256':accept['plan_sha256'],
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'authorization':'User requested preparing the second dataset and continuing automatically after this round.'}
with (root/'launch.lock').open('x') as f:json.dump(release,f,indent=2)
with (root/'release_after_round2.json').open('x') as f:json.dump(release,f,indent=2)
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'logs/controller.log').open('xb') as log:
 proc=subprocess.Popen([cfg['python'],'-u',str(root/'code'/accept['controller'])],cwd=root/'code',env=env,
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 stat=(P('/proc')/str(proc.pid)/'stat').read_text().rsplit(')',1)[1].split()
 receipt={'account':cfg['account'],'state':'launched_once','pid':proc.pid,'start_ticks':int(stat[19]),
  'controller':str(root/'code'/accept['controller']),'release_sha256':sha(root/'release_after_round2.json')}
with (root/'launch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''


def execute(client, account, source, proof=None):
    entry = dict(ACCOUNTS[account], account=account, python=PYTHONS[ACCOUNTS[account]['port']])
    program = 'CONFIG=' + repr(entry) + '\n'
    if proof is not None: program += 'PROOF=' + repr(proof) + '\n'
    program += source
    command = 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(entry['python']) + ' -u -'
    channel = client.get_transport().open_session(timeout=20); channel.exec_command(command)
    channel.sendall(program.encode()); channel.shutdown_write(); output = bytearray(); error = bytearray()
    deadline = time.monotonic() + (300 if proof is not None else 90)
    while True:
        while channel.recv_ready(): output.extend(channel.recv(65536))
        while channel.recv_stderr_ready(): error.extend(channel.recv_stderr(65536))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready(): break
        if time.monotonic() > deadline: raise TimeoutError('Scoped SSH operation timed out; do not repeat a launch attempt blindly')
        time.sleep(.05)
    code = channel.recv_exit_status()
    if code: raise RuntimeError('Scoped remote command failed: ' + error.decode(errors='replace')[-1400:])
    return json.loads(output)


def query(account, passwords):
    client = None
    try:
        client = connect(ACCOUNTS[account]['port'], passwords)
        return {'ok': True, 'snapshot': execute(client, account, REMOTE)}
    except Exception as exc:
        return {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}
    finally:
        if client: client.close()


def collect(client, account, snapshot, destination):
    """Bounded raw evidence only; never downloads checkpoint files."""
    sftp = client.open_sftp(); root = snapshot['root']; receipts = []
    relative = ['plan.json', 'final_acceptance.json', 'frozen_manifest.json', 'queue_state.json']
    for job, item in snapshot['jobs'].items():
        if (item['status'] or {}).get('state') != 'complete': continue
        relative += ['jobs/'+job+'.json', 'outputs/'+job+'/log.txt', 'outputs/'+job+'/checkpoint_log_binding.json']
        relative += ['results/'+job+'/'+split+'/metrics.json' for split in ['val', 'test']]
    for rel in relative:
        path = destination/account/rel; path.parent.mkdir(parents=True, exist_ok=True)
        raw = sftp.open(root+'/'+rel, 'rb').read(); digest = hashlib.sha256(raw).hexdigest()
        if path.exists(): assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Previously collected evidence changed'
        else:
            with path.open('xb') as handle: handle.write(raw)
        receipts.append({'account':account,'source':root+'/'+rel,'local':str(path),'bytes':len(raw),'sha256':digest})
    return receipts


def summarize_comparisons(servers):
    rows=[];pairs=[]
    for account, result in servers.items():
        if not result['ok']:continue
        snap=result['snapshot'];by_arm={}
        for job,item in snap['jobs'].items():
            test=item['independent'].get('test')
            if not test or (item['status'] or {}).get('state')!='complete':continue
            status=item['status'];arm=status['arm'];by_arm[arm]=test
            rows.append({'account':account,'seed':status['seed'],'arm':arm,'job':job,
                         'best_round':test['checkpoint_epoch_zero_based']+1,'metrics_raw_0_to_1':test['metrics_raw_0_to_1']})
        for left,right in [('c','baseline'),('c','c_mean'),('c','lowpass'),('c_mean','lowpass')]:
            if left not in by_arm or right not in by_arm:continue
            l,r=by_arm[left],by_arm[right]
            assert l['ground_truth_sha256']==r['ground_truth_sha256'] and l['protocol']==r['protocol']
            pairs.append({'account':account,'seed':snap['plan']['seed'],'comparison':left+' - '+right,
                          'difference_percentage_points':{k:(v-r['metrics_raw_0_to_1'][k])*100 for k,v in l['metrics_raw_0_to_1'].items()}})
    return {'rows':rows,'paired_comparisons':pairs,'scope':'Same seed and same official TinyPerson protocol. Two seeds do not establish significance or equivalence.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch-once',action='store_true');parser.add_argument('--collect',action='store_true')
    args=parser.parse_args();now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S_%fZ')
    DEST.mkdir(parents=True,exist_ok=True);passwords=json.loads(scheduled_monitor_v4.credentials_bytes())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending={a:pool.submit(query,a,passwords) for a in ACCOUNTS};servers={a:f.result() for a,f in pending.items()}
        both=all(r['ok'] and r['snapshot']['old_queue']['value'].get('state')=='complete' for r in servers.values())
        ready=all(r['ok'] and (r['snapshot']['acceptance'] or {}).get('status')=='ready_waiting_for_both_original_queues' for r in servers.values())
        if ready:
            assert len({r['snapshot']['acceptance']['initialization_sha256'] for r in servers.values()})==1
            for r in servers.values():
                s=r['snapshot'];assert s['plan_sha256']==s['acceptance']['plan_sha256']
                assert s['frozen_manifest_sha256']==s['acceptance']['frozen_manifest_sha256']
        selection=json.loads(SELECTION_POLICY.read_text(encoding='utf-8')) if SELECTION_POLICY.is_file() else {}
        selection_allows_start=(selection.get('allow_automatic_training_start') is True
                                and selection.get('selected_dataset')=='TinyPerson_v1')
        launches={}
        if args.launch_once and both and ready and selection_allows_start:
            proof={a:r['snapshot']['old_queue'] for a,r in servers.items()}
            for account,result in servers.items():
                if result['snapshot']['launch_lock_exists']:continue
                client=None
                try:
                    client=connect(ACCOUNTS[account]['port'],passwords);launches[account]=execute(client,account,LAUNCH,proof)
                except Exception as exc:launches[account]={'error':type(exc).__name__+': '+str(exc),'automatic_retry':False}
                finally:
                    if client:client.close()
        report={'checked_utc':now.isoformat(),'checked_los_angeles':now.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
                'both_original_queues_complete':both,'both_tiny_preparations_ready':ready,
                'selection_policy':selection,'selection_allows_start':selection_allows_start,
                'handoff_state':('held_pending_dataset_selection' if not selection_allows_start else
                                 'eligible' if both and ready else 'waiting_for_current_round_or_missing_evidence'),
                'launch_requested':args.launch_once,'launches':launches,'servers':servers}
        if args.collect:
            receipts=[]
            for account,result in servers.items():
                if not result['ok'] or not result['snapshot']['queue']:continue
                client=None
                try:
                    client=connect(ACCOUNTS[account]['port'],passwords)
                    receipts+=collect(client,account,result['snapshot'],DEST.parent/'结果收集')
                finally:
                    if client:client.close()
            report['collected_files']=receipts;report['comparisons']=summarize_comparisons(servers)
        raw=json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8');file=DEST/(stamp+'.json')
        with file.open('xb') as handle:handle.write(raw)
        (DEST/'latest_snapshot.json').write_bytes(raw)
        summary={'checked_los_angeles':report['checked_los_angeles'],'snapshot':str(file),'sha256':hashlib.sha256(raw).hexdigest(),
                 'both_original_queues_complete':both,'preparations_ready':ready,
                 'selection_allows_start':selection_allows_start,'launches':launches,'accounts':{}}
        for account,result in servers.items():
            if not result['ok']:summary['accounts'][account]={'error':result['error']};continue
            s=result['snapshot'];summary['accounts'][account]={'old_state':s['old_queue']['value'].get('state'),
                'tiny_state':(s['queue'] or {}).get('state','not_started'),'completed_training':len((s['queue'] or {}).get('training_completed_jobs',[])),
                'complete':len((s['queue'] or {}).get('completed_jobs',[]))}
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:passwords.clear()


if __name__=='__main__':main()
