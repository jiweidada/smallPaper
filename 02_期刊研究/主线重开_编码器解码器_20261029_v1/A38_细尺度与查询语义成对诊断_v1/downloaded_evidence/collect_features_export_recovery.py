from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from pool_tensor import pool_tensor
from npz_export import save_npz

sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(); parser.add_argument('--manifest-sha256',required=True)
parser.add_argument('--recovery-manifest-sha256',required=True)
args=parser.parse_args()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2)
assert sha(D/'execution_inputs_manifest.json')==args.manifest_sha256
manifest=json.loads((D/'execution_inputs_manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files_sha256'].items():
    p=(D/name).resolve(); assert p.is_relative_to(D)
    assert sha(p)==digest,name
assert sha(D/'export_recovery_execution_manifest.json')==args.recovery_manifest_sha256
recovery_manifest=json.loads((D/'export_recovery_execution_manifest.json').read_text(encoding='utf-8'))
for name,digest in recovery_manifest['files_sha256'].items(): assert sha(D/name)==digest,name
amendment=json.loads((D/'export_recovery_amendment_FROZEN.json').read_text(encoding='utf-8'))
assert amendment['old_completed_entries']==2 and amendment['remaining_native_entries']==112
for name,record in amendment['preserved_partial_files'].items(): assert sha(D/name)==record['sha256'],name
protocol=json.loads((D/'semantic_protocol_FROZEN.json').read_text(encoding='utf-8'))
inputs=json.loads((D/'collector_inputs.json').read_text(encoding='utf-8'))
assert protocol['status']=='FROZEN_BEFORE_ANY_NEW_AUTHOR_MODEL_FORWARD'
assert protocol['caps']['whole_detector_entries']==114
assert len(inputs['cases'])==112
assert not (D/'collection_result.json').exists()
lock=os.open(str(D/'collector_recovery.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
os.write(lock,str(os.getpid()).encode()); os.close(lock)
torch.set_num_threads(4); random.seed(381031); np.random.seed(381031); torch.manual_seed(381031)
torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
device=torch.device('cuda:0')
loader_path=Path(protocol['loader_path'])
assert sha(loader_path)==protocol['loader_sha256']
assert sha(protocol['checkpoint_path'])==protocol['checkpoint_sha256']
spec=importlib.util.spec_from_file_location('a38_original_dome_loader',loader_path)
loader=importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
model,postprocessor,transform,load_receipt=loader.load_model(receipt_path=D/'recovery_native_load_receipt.json')
before=loader.state_digest(model)
assert before==load_receipt['state_digest']
assert before==json.loads((D/'native_load_receipt.json').read_text())['state_digest']
model=model.to(device).eval(); postprocessor=postprocessor.to(device).eval()
assert model.decoder.eval_idx==model.decoder.decoder.eval_idx==2
assert model.decoder.min_num_select==250 and model.decoder.max_num_select==500
assert len(model.decoder.decoder.layers)==3
assert not any(m.training for m in model.modules())
assert not any(p.requires_grad for p in model.parameters())
assert postprocessor.use_focal_loss and postprocessor.num_classes==12
assert not postprocessor.remap_mscoco_category and not postprocessor.deploy_mode
captures=D/'captures'; assert captures.is_dir() and not list(captures.iterdir())
ledger=(D/'recovery_forward_entries.jsonl').open('x',encoding='utf-8')
entries=2; completed=2; current=''; observed={}; hooks=[]
def event(value):
    ledger.write(json.dumps(value,ensure_ascii=False)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
def count_entry(module,arguments):
    global entries
    assert entries<114,'Author-detector-entry cap exhausted'
    entries+=1; event(dict(event='author_detector_entry',entry=entries,scope=current,time=time.time()))
def count_exit(module,arguments,outputs):
    global completed
    completed+=1; event(dict(event='author_detector_completed',entry=entries,scope=current,time=time.time()))
counter_pre=model.register_forward_pre_hook(count_entry)
counter_post=model.register_forward_hook(count_exit)
def memory_hook(module,arguments):
    assert 'memory' not in observed
    observed['memory']=arguments[2].detach()
    observed['spatial_shapes']=[list(x) for x in arguments[3]]
def query_hook(module,arguments):
    assert 'query' not in observed
    observed['query']=arguments[0].detach()
def score_hook(module,arguments,outputs):
    assert 'raw_logits' not in observed
    observed['raw_logits']=outputs.detach()
def native_arrays(outputs,width,height):
    logits=outputs['pred_logits']; boxes=outputs['pred_boxes']
    assert logits.ndim==3 and logits.shape[0]==1 and logits.shape[2]==12
    n=logits.shape[1]; assert 250<=n<=500
    probabilities=logits.sigmoid()
    norm_xyxy=torchvision.ops.box_convert(boxes,in_fmt='cxcywh',out_fmt='xyxy')
    size=torch.tensor([[width,height]],dtype=boxes.dtype,device=device)
    pp=postprocessor(outputs,size)[0]
    top_scores,flat=torch.topk(probabilities.flatten(1),n,dim=-1)
    labs=flat%12; query_ids=flat//12
    full_xyxy=norm_xyxy*size.repeat(1,2).unsqueeze(1)
    top_boxes=full_xyxy.gather(1,query_ids.unsqueeze(-1).repeat(1,1,4))
    assert torch.equal(labs[0],pp['labels'])
    assert torch.equal(top_scores[0],pp['scores'])
    assert torch.equal(top_boxes[0],pp['boxes'])
    arrays=dict(query_boxes_xyxy=norm_xyxy[0].detach().cpu().numpy(),
        native_logits=logits[0].detach().cpu().numpy(),native_probabilities=probabilities[0].detach().cpu().numpy(),
        native_labels=labs[0].cpu().numpy(),native_scores=top_scores[0].cpu().numpy(),
        native_query_ids=query_ids[0].cpu().numpy(),native_boxes_xyxy_original=top_boxes[0].cpu().numpy())
    assert all(np.isfinite(a).all() for a in arrays.values())
    return arrays,n,norm_xyxy[0]
def load_case(row):
    assert sha(row['image_path'])==row['image_sha256']
    assert sha(row['annotation_path'])==row['annotation_sha256']
    with Image.open(row['image_path']) as image:
        assert image.size==(row['width'],row['height'])
        tensor=transform(image.convert('RGB')).to(device)
    assert tensor.shape==(1,3,800,800) and tensor.dtype==torch.float32
    return tensor
torch.cuda.reset_peak_memory_stats()
first=inputs['cases'][0]
baseline=dict(np.load(D/'baseline_first_no_capture_hooks.npz',allow_pickle=False))
original_control=dict(np.load(D/'first_control_memory_and_head.npz',allow_pickle=False))
hooks=[model.decoder.decoder.register_forward_pre_hook(memory_hook),
       model.decoder.dec_score_head[2].register_forward_pre_hook(query_hook),
       model.decoder.dec_score_head[2].register_forward_hook(score_hook)]
query_counts=[]
with torch.inference_mode():
    for row in inputs['cases']:
        current='case_%03d'%row['index']; observed.clear()
        x=load_case(row); output=model(x)
        arrays,n,boxes=native_arrays(output,row['width'],row['height'])
        assert observed['spatial_shapes']==[[200,200],[100,100],[50,50],[25,25]]
        memory=observed['memory']; query=observed['query']; raw=observed['raw_logits']
        assert memory.shape==(1,53125,256) and query.shape==(1,n,256) and raw.shape==(1,n,12)
        fine_map=memory[:,:40000].transpose(1,2).reshape(1,256,200,200)
        fine,contrast,ring_counts=pool_tensor(fine_map,boxes,64)
        arrays.update(query_features=query[0].cpu().numpy(),fine_features=fine.cpu().numpy(),
            contrast_features=contrast.cpu().numpy(),ring_counts=ring_counts.cpu().numpy(),
            raw_class_logits=raw[0].cpu().numpy(),width=np.array(row['width']),height=np.array(row['height']),
            index=np.array(row['index']),file=np.array(row['file']),split=np.array(row['split']))
        annotation_text=Path(row['annotation_path']).read_text().strip()
        arrays['annotations']=np.loadtxt(row['annotation_path'],delimiter=',',ndmin=2).reshape(-1,8) if annotation_text else np.empty((0,8))
        for name in ['query_features','fine_features','contrast_features','raw_class_logits']:
            assert np.isfinite(arrays[name]).all(),name
        if row['index']==0:
            for name in baseline: np.testing.assert_array_equal(arrays[name],baseline[name])
            np.testing.assert_array_equal(fine_map[0].cpu().numpy(),original_control['fine_memory'])
            np.testing.assert_array_equal(model.decoder.dec_score_head[2].weight.detach().cpu().numpy(),original_control['head_weight'])
            np.testing.assert_array_equal(model.decoder.dec_score_head[2].bias.detach().cpu().numpy(),original_control['head_bias'])
            save_json(D/'recovery_first_fidelity.json',dict(status='NATIVE_OUTPUTS_AND_FULL_FINE_MAP_BITWISE_EQUAL_PRESERVED_ORIGINAL',
                recovered_entry=entries,reserved_repeat_used_here=True,full_query_feature_repeat_available=False,
                scope='Native prediction arrays,entire256x200x200fine map andheadparameters equal preserved originals;originalqueryfeatureswere not exported.'))
        save_npz(captures/(current+'.npz'),arrays)
        query_counts.append(n)
        assert torch.cuda.max_memory_allocated()<20*1024**3
        progress=dict(case=row['index'],split=row['split'],query_count=n,entries=entries,completed=completed)
        event(dict(event='case_exported',**progress)); print(json.dumps(progress),flush=True)
        del output,x,fine_map,fine,contrast,ring_counts,memory,query,raw,boxes,arrays
        observed.clear()
assert entries==completed==114
after=loader.state_digest(model); assert after==before
result=dict(status='COMPLETE_114_NATIVE_FROZEN_ENTRIES_NO_OPTIMIZER',entries=entries,completed=completed,
    cases=112,train_cases=64,val_cases=48,query_counts=query_counts,
    native_no_hook_fidelity='All baseline prediction arrays bitwise equal first instrumented case.',
    repeated_fidelity='Recovery firstnativepredictionandwholefinemapagreewithpreservedoriginal;fullqueryfeaturebitwiserepeatunavailableandnotclaimed.',
    initial_state_digest=before,final_state_digest=after,state_unchanged=True,
    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
    source_commit=protocol['source_commit'],checkpoint_sha256=protocol['checkpoint_sha256'],
    detector_optimizer_updates=0,diagnostic_data_probe_fits=0,export_recovery_amendment='export_recovery_amendment_FROZEN.json',
    original_failed_entries=2,recovery_process_entries=112,full_query_feature_repeat_available=False,
    files_sha256={str(p.relative_to(D)):sha(p) for p in sorted(list(captures.glob('*.npz'))+[D/'baseline_first_no_capture_hooks.npz',D/'first_control_memory_and_head.npz',D/'native_load_receipt.json',D/'recovery_native_load_receipt.json',D/'recovery_first_fidelity.json'])})
save_json(D/'collection_result.json',result)
event(dict(event='collection_terminal_success',entries=entries,completed=completed)); ledger.close()
print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','query_counts']},ensure_ascii=False),flush=True)
