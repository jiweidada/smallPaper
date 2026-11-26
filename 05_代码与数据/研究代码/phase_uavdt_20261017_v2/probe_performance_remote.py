"""Scoped read-only CPU/I/O/GPU/log sampling; no trainer attachment or configuration changes."""
import datetime,json,os,re,shutil,subprocess,time
from pathlib import Path
import yaml
root=Path(CONFIG['root']);plan=json.loads((root/'plan.json').read_text());ansi=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
def read(p):
 try:return Path(p).read_text()
 except (FileNotFoundError,PermissionError,OSError):return None
def cmd(args,timeout=25):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr[:1200]}
 except Exception as e:return {'error':repr(e)}
def procstat(pid):
 base=Path('/proc')/str(pid);raw=read(base/'stat')
 if not raw:return None
 stat=raw.rsplit(')',1)[1].split();status=read(base/'status') or '';io=read(base/'io') or ''
 return {'pid':pid,'state':stat[0],'start_ticks':stat[19],'utime_ticks':int(stat[11]),'stime_ticks':int(stat[12]),'nice':int(stat[16]),'threads':int(stat[17]),'last_cpu':int(stat[36]),
 'rss_pages':int(stat[21]),'status':{k:v.strip() for k,v in re.findall(r'^(Name|Threads|VmRSS|Cpus_allowed_list|NSpid|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):\s*(.+)$',status,re.M)},
 'io':{k:int(v) for k,v in re.findall(r'^(\w+):\s*(\d+)$',io,re.M)},'main_wchan':read(base/'wchan')}
def scope(job):
 status=json.loads((root/'jobs'/(job+'.json')).read_text());pid=status['pid'];before=procstat(pid)
 assert before and before['start_ticks']==str(status['start_ticks'])
 args=[s.decode(errors='replace') for s in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if s]
 assert str(root/'code/train_uavdt.py') in args
 children=[int(s) for s in (read(f'/proc/{pid}/task/{pid}/children') or '').split()]
 return {'job':job,'pid':pid,'children':children,'before':{str(p):procstat(p) for p in [pid]+children},
 'thread_environment':{k:v for item in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0') if b'=' in item for k,v in [item.decode(errors='replace').split('=',1)] if k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONUNBUFFERED']},
 'cgroup':read(f'/proc/{pid}/cgroup'),'affinity':sorted(os.sched_getaffinity(pid))}
report={'account':CONFIG['account'],'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'available_tools':{n:shutil.which(n) for n in ['py-spy','perf','pidstat','iostat','nvidia-smi']},
 'cpu_count':os.cpu_count(),'clock_ticks':os.sysconf('SC_CLK_TCK'),'meminfo':read('/proc/meminfo'),
 'cgroup':{p:read('/sys/fs/cgroup/'+p) for p in ['cpu.max','cpu.stat','cpu.pressure','memory.max','memory.current','memory.events','io.pressure','cpu/cpu.cfs_quota_us','cpu/cpu.cfs_period_us','cpu/cpu.stat']},
 'system_pressure':{p:read('/proc/pressure/'+p) for p in ['cpu','io','memory']},'loadavg':read('/proc/loadavg'),'jobs':[]}
for arm in plan['jobs']:
 job=f'uavdt_{arm}_full96_seed{plan["seed"]}';item=scope(job);cfg=yaml.safe_load(Path(plan['configs'][arm]).read_text())
 item['recipe']={k:cfg.get(k) for k in ['train_dataloader','val_dataloader','use_amp','use_ema','epoches','HGNetv2_Light','HGNetv2_Light_Phase']}
 console=root/'logs'/(job+'.log');text=ansi.sub('',console.read_text(errors='replace'));records=[]
 for line in text.splitlines():
  m=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?Epoch:\s*(\d+)/96\s+\[\s*(\d+)/(\d+)\].*?time:\s*([\d.]+)\s+data:\s*([\d.]+)',line)
  if m:records.append({'remote_time':m[1],'epoch_zero_based':int(m[2]),'batch':int(m[3]),'batch_total':int(m[4]),'seconds_per_step':float(m[5]),'data_seconds':float(m[6])})
 item['progress_last30']=records[-30:]
 log=root/'outputs'/job/'log.txt';rows=[json.loads(s) for s in log.read_text().splitlines() if s.strip()] if log.exists() else []
 item['epoch_times']=[{k:r.get(k) for k in ['epoch','experiment_train_seconds','experiment_epoch_seconds','train_amp_skipped_steps','train_amp_final_scale']} for r in rows]
 report['jobs'].append(item)
fields='index,name,pstate,clocks.current.sm,clocks.current.memory,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free'
report['GPU_before']=cmd(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'])
started=time.monotonic();report['GPU_process_samples']=cmd(['nvidia-smi','pmon','-c','12','-s','um'],timeout=25)
elapsed=time.monotonic()-started;report['sample_seconds']=elapsed
if elapsed<10:time.sleep(10-elapsed);elapsed=time.monotonic()-started;report['sample_seconds']=elapsed
for item in report['jobs']:
 item['after']={p:procstat(int(p)) for p in item['before']}
 item['CPU_core_equivalents']={p:((item['after'][p]['utime_ticks']+item['after'][p]['stime_ticks'])-(v['utime_ticks']+v['stime_ticks']))/os.sysconf('SC_CLK_TCK')/elapsed for p,v in item['before'].items() if v and item['after'][p]}
 item['thread_waits']={p.name:read(p/'wchan') for p in (Path('/proc')/str(item['pid'])/'task').iterdir()}
report['GPU_after']=cmd(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'])
report['cgroup_after']={p:read('/sys/fs/cgroup/'+p) for p in ['cpu.stat','cpu.pressure','io.pressure','cpu/cpu.stat']}
print(json.dumps(report),flush=True)
