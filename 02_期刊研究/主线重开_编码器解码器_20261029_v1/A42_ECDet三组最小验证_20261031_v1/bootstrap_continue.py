import pathlib,hashlib,json
R=pathlib.Path(__file__).resolve().parent
source=R/'source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg'
p=source/'engine/core/_config.py'
s=p.read_text()
old='from torch.utils.tensorboard import SummaryWriter'
new='try:\n    from torch.utils.tensorboard import SummaryWriter\nexcept ModuleNotFoundError:\n    SummaryWriter = None  # Optional TensorBoard logging not used by A42 JSON runner.'
assert s.count(old)==1
p.write_text(s.replace(old,new),encoding='utf-8')
(R/'optional_logging_compatibility.json').write_text(json.dumps({'path':str(p),'reason':'tensorboard not installed; training uses JSON logger and never constructs BaseConfig.writer','model_math_changed':False}),encoding='utf-8')
s=(R/'bootstrap_remote.py').read_text()
start=s.index('sys.path.insert(0,str(source))')
pre='import hashlib,json,pathlib,sys\nR=pathlib.Path(__file__).resolve().parent\nsource=R/"source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg"\necvit=source/"engine/edgecrafter/ecvit.py"\nbefore=hashlib.sha256(ecvit.read_text().removeprefix("from __future__ import annotations\\n").encode()).hexdigest()\ndef digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()\ndef emit(name,x):\n (R/name).write_text(json.dumps(x,indent=2),encoding="utf-8");print(json.dumps(x),flush=True)\n'
exec(compile(pre+s[start:],str(R/'bootstrap_remote.py'),'exec'),{'__file__':str(R/'bootstrap_remote.py'),'__name__':'__main__'})
