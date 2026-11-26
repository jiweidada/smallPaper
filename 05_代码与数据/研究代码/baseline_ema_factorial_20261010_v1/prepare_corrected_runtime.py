"""Minimal independent runtime correction; freezes original EMA gradient flags."""
from pathlib import Path
import ast,base64,copy,datetime,hashlib,io,json,zipfile
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'英文小论文/code/small_object_encoder_decoder12_20261010_v1'
NEW=OLD.with_name('small_object_encoder_decoder12_20261010_v2')
OUT=ROOT/'outputs/small_object_encoder_decoder12_20261010_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    NEW.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
    names=['runtime.py','experiment.py','baseline_eval.py','supervise.py','training_audit.py','results.py','stage_trace.py','small_object_model.py','geometry_memory.py','evaluator_binding.py','evaluator_sources.json','source_binding.json','prepare_remote.py']
    for n in names:(NEW/n).write_bytes((OLD/n).read_bytes())
    p=NEW/'baseline_eval.py';s=p.read_text();needle="model.load_state_dict(payload['ema']['module'],strict=True);del payload";assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    # Native ModelEMA freezes every parameter; loading its tensor values alone\n    # does not preserve this inference state. Keep original historical gate.\n    model.requires_grad_(False);assert not any(p.requires_grad for p in model.parameters())")
    s=s.replace("'optimizer_steps':0,'epochs_added':0", "'optimizer_steps':0,'epochs_added':0,'all_parameters_frozen_like_native_EMA':True")
    p.write_text(s,encoding='utf8')
    protocol=read(OLD/'protocol.json');protocol.update(schema='small_object_encoder_decoder12_20261010_v2',status='FROZEN_BEFORE_CORRECTED_BASELINE_REPLAY_AND_CANDIDATE_TRAINING',frozen_local_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor_protocol_sha256=sha(OLD/'protocol.json'),engineering_correction={'change':'Independent BASE12 evaluation now freezes ALL model parameter gradient flags like native ModelEMA, after strict load. No model/optimizer/data/training/evaluator/tolerance/scientific gate change.','cause_evidence':'Same input/weights/buffers six-forward seed1 probe toggles output exactly with requires_grad flags; deepcopy and wrapper frozen outputs agree. Full original-wrapper/nativeEMA seed1 replay reproduces all12 historical statistics exactly (0 absolute difference).','predecessor_new_epochs':0,'predecessor_candidate_updates':0,'failed_controllers':'Both immutable v1 failures/locks retained; not retried or erased.','factorial_protocol_sha256':sha(ROOT/'outputs/baseline_ema_factorial_20261010_v1/protocol.json'),'factorial_script_sha256':sha(Path(__file__).with_name('probe.py'))},baseline='Existing same-seed native BASE12 EMA checkpoint,strict tensor load AND every parameter frozen like native ModelEMA. Original historical all12stat1e-8 gate retained unchanged.')
    write(NEW/'protocol.json',protocol);write(OUT/'protocol.json',protocol)
    (NEW/'configs').mkdir()
    for p in (OLD/'configs').iterdir():
        plan=read(p);plan.update(schema=protocol['schema'],screen_protocol_sha256=sha(NEW/'protocol.json'));write(NEW/'configs'/p.name,plan)
    for n in ('geometry_memory.py','small_object_model.py'):assert sha(NEW/n)==sha(OLD/n),n
    for p in NEW.glob('*.py'):ast.parse(p.read_text(),filename=p.name)
    files={p.relative_to(NEW).as_posix():sha(p) for p in NEW.rglob('*') if p.is_file()}
    write(NEW/'deployment_manifest.json',files)
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in list(files)+['deployment_manifest.json']:z.writestr(n,(NEW/n).read_bytes())
    raw=data.getvalue();(OUT/'deploy_bundle.zip').write_bytes(raw);(OUT/'deploy_bundle.zip.b64').write_text(base64.b64encode(raw).decode())
    meta={'status':'FROZEN_NOT_LAUNCHED','files':len(files)+1,'zip_sha256':hashlib.sha256(raw).hexdigest(),'zip_bytes':len(raw),'protocol_sha256':sha(NEW/'protocol.json'),'exact_model_unchanged':True,'historical_gate_unchanged':True,'candidate_updates':0,'new_epochs':0}
    write(OUT/'deployment_local.json',meta);print(json.dumps(dict(meta,b64=base64.b64encode(raw).decode())))
if __name__=='__main__':main()
