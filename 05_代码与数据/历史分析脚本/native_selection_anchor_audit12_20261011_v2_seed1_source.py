from pathlib import Path
import json,base64,hashlib,os,datetime,ast
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
torch.set_num_threads(1);b=Path('/25085400120/DEIM/experiments');out=b/'native_selection_anchor_audit12_20261011_v2/seed1';out.mkdir(parents=True,exist_ok=False)
native=Path(json.loads((b/'original_stem12_20261011_v1/seed1_original_stem/bundle/receipt.json').read_bytes())['paths']['native_root']);source=native/'engine/deim/dfine_decoder.py';raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()=='b31408e49d4961e1e82692703b8f9fc3f07f1e280fd13ee12f896cb337f9bae5'
tree=ast.parse(raw.decode('utf-8-sig'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DFINETransformer');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_anchors');ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
class Settings:eval_spatial_size=[640,640];feat_strides=[8,16,32];eps=.01
generated=ns['_generate_anchors'](Settings())[0][0].numpy();cases={}
for family,arm in [('original_stem12_20261011_v1','original_stem'),('native_stem_scratch12_20261011_v3','native_stem_scratch'),('stem_component12_20261011_v1','highfreq_rgb')]:
 root=b/family/f'seed1_{arm}';done=json.loads((root/'run/training_completion.json').read_bytes());h=hashlib.sha256((root/'run/last.pth').read_bytes()).hexdigest();assert h==done['last_checkpoint_sha256'];x=torch.load(root/'run/last.pth',map_location='cpu',weights_only=False);assert x['last_epoch']==11
 E=x['ema']['module']['decoder.anchors'][0].numpy().copy();M=x['model']['decoder.anchors'][0].numpy().copy();assert E.dtype==M.dtype==np.float32 and E.shape==M.shape==generated.shape==(8400,4)
 np.savez_compressed(out/f'{arm}.npz',EMA_anchors=E,model_anchors=M,generated_anchors=generated)
 good=np.isfinite(E).all(1);assert np.array_equal(good,np.isfinite(generated).all(1));cases[arm]=dict(checkpoint_sha256=h,state_key='decoder.anchors',EMA_equal_to_original_export=True,array_sha256=hashlib.sha256(E.tobytes()).hexdigest(),export_sha256=hashlib.sha256((out/f'{arm}.npz').read_bytes()).hexdigest(),model_equal_generated=np.array_equal(M,generated),EMA_model_finite_logits_max_abs_difference=float(np.max(np.abs(E[good]-M[good]))),model_generated_max_abs_difference=float(np.max(np.abs(M[good]-generated[good]))),EMA_center_max_abs_difference=float(np.max(np.abs(1/(1+np.exp(-E[good,:2].astype(np.float64)))-1/(1+np.exp(-generated[good,:2].astype(np.float64)))))))
 del x,E,M
ema=native/'engine/optim/ema.py';(out/'native_ema_source.txt').write_bytes(ema.read_bytes());report=dict(status='PASS_CHECKPOINT_BOUND_EMA_MODEL_AND_NATIVE_GENERATED_ANCHOR_TENSOR_AUDIT',seed=1,cases=cases,native_generator_source_sha256=hashlib.sha256(raw).hexdigest(),EMA_source_sha256=hashlib.sha256(ema.read_bytes()).hexdigest(),CUDA_initialized=torch.cuda.is_initialized(),model_forwards=0,training_epochs=0,optimizer_updates=0,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());assert not report['CUDA_initialized'];(out/'report.json').write_text(json.dumps(report,indent=2));o=report
raw=json.dumps(o).encode();h=hashlib.sha256(raw).hexdigest();s=base64.b64encode(raw).decode()
for i in range(4):print('\rH'+str(i)+' '+h[i*16:i*16+16])
print('\rBYTES '+str(len(raw)));print('\rB64_BEGIN')
for i in range(0,len(s),64):print('\r'+s[i:i+64])
print('\rB64_END')

