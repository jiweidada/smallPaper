from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess
R=Path("/root/uav_a60_20261102_v1/pilot_v1")
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(x):
 if not x:return None
 try:
  f=Path("/proc/"+str(x["pid"])+"/stat").read_text().split(") ",1)[1].split()
  same=f[19]==str(x["start_ticks"])
  return dict(pid=x["pid"],expected_start_ticks=str(x["start_ticks"]),actual_start_ticks=f[19],same_start=same,identity_matched=same,state=f[0],live=same and f[0]!="Z")
 except FileNotFoundError:return dict(pid=x["pid"],exists=False,live=False)
def rows(p):
 out=[]
 if p.is_file():
  for s in p.read_text().splitlines():
   try:out.append(json.loads(s))
   except json.JSONDecodeError:pass
 return out
parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["mini","full"],default="mini")
mode=parser.parse_args().mode
scope=read(R/"protocol_FROZEN_v1.json")
bound=all(hashlib.sha256((R/"code"/n).read_bytes()).hexdigest()==h for n,h in scope["source_sha256"].items())
now=datetime.now(timezone.utc).isoformat()
d=dict(utc=now,recorded_utc=now,protocol_sha256=hashlib.sha256((R/"protocol_FROZEN_v1.json").read_bytes()).hexdigest(),source_bound=bound,source_binding=bound,supervisor=proc(read(R/"launch_receipt_v1.json")),supervisor_exit=read(R/"supervisor_exit_v1.json"),arms={})
for a in ["B","M","Q"]:
 p=R/a;launch=read(R/(a+"_launch_v1.json"));epochs=[read(q) for q in sorted(p.glob("epoch??.json"))]
 item=dict(process=proc(launch),progress=read(p/"progress.json"),epochs=len(epochs),evaluations=[dict(epoch=e["epoch"],metrics={k:e["metrics"][k] for k in ["AP","AP50","AP75","APs","APs75","APm","APl","ARs"]}) for e in epochs if "metrics" in e],exit=read(p/"exit.json"),terminal=(read(p/"result.json") or {}).get("status"))
 if mode=="full":
  events=rows(p/"actual_events.jsonl")
  item.update(exists=p.exists(),launch=launch,completed_epochs=len(epochs),epochs=epochs,journal_counts=dict(started=sum(e["event"]=="forward_start" for e in events),completed=sum(e["event"]=="forward_complete" for e in events),updates=sum(e["event"]=="optimizer_returned" for e in events)),first8_flow=rows(p/"first8_input_flow.jsonl"),initialization=read(p/"initialization.json"),result=read(p/"result.json"),worker_reaped=read(R/(a+"_worker_reaped_v1.json")))
 d["arms"][a]=item
if mode=="full":
 d["supervisor_start"]=read(R/"supervisor_start_v1.json")
 d["judgment"]=read(R/"fixed12_judgment_v1.json")
 d["GPU"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total,utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True).strip()
print(json.dumps(d),flush=True)

