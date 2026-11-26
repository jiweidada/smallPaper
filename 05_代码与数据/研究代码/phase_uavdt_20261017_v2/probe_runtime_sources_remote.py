"""Read source evidence and GPU/CPU locality before considering reversible OS tuning."""
import ast,hashlib,json,os,subprocess
from pathlib import Path
root=Path(CONFIG['root'])
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);return {'code':p.returncode,'stdout':p.stdout,'stderr':p.stderr[:1000]}
report={'account':CONFIG['account'],'gpu_identity':cmd(['nvidia-smi','--query-gpu=uuid,pci.bus_id,persistence_mode,compute_mode','--format=csv,noheader']),
 'performance_power':cmd(['nvidia-smi','-q','-d','PERFORMANCE,POWER,CLOCK,TEMPERATURE']),
 'topology':cmd(['nvidia-smi','topo','-m']),'CPU_affinity':sorted(os.sched_getaffinity(0)),'sources':[],'CPU_topology':[]}
for rel,names in [('engine/extre_module/ops.py',['Profile']),('engine/data/dataloader.py',['BatchImageCollateFunction']),('engine/misc/logger.py',['MetricLogger']),('engine/backbone/hgnetv2_light_phase.py',['P2ToP3PhaseMeanResidual'])]:
 p=root/'code'/rel
 if not p.is_file():report['sources'].append({'relative':rel,'missing':True});continue
 source=p.read_text();tree=ast.parse(source);snips=[];lines=source.splitlines()
 for node in tree.body:
  if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:snips.append({'name':node.name,'line':node.lineno,'text':'\n'.join(lines[node.lineno-1:node.end_lineno])})
 report['sources'].append({'relative':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_definitions':snips})
for cpu in report['CPU_affinity']:
 base=Path('/sys/devices/system/cpu')/('cpu'+str(cpu));top=base/'topology'
 def read(p):return p.read_text().strip() if p.is_file() else None
 report['CPU_topology'].append({'cpu':cpu,'socket':read(top/'physical_package_id'),'core':read(top/'core_id'),'siblings':read(top/'thread_siblings_list'),'NUMA_nodes':[p.name for p in base.glob('node[0-9]*')]})
print(json.dumps(report),flush=True)
