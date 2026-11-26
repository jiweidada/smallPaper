from __future__ import annotations
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,random,sys,time
import numpy as np
import torch
import torchvision
from PIL import Image
from pool_tensor import pool_tensor
from npz_export import save_npz
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--manifest-sha256',required=True);args=p.parse_args()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,default=scalar)
assert sha(D/'execution_inputs_manifest.json')==args.manifest_sha256
manifest=json.loads((D/'execution_inputs_manifest.json').read_text(encoding='utf-8'))
for name,h in manifest['files_sha256'].items():assert sha(D/name)==h,name
protocol=json.loads((D/'semantic_protocol_FROZEN.json').read_text(encoding='utf-8'))
inputs=json.loads((D/'collector_inputs.json').read_text(encoding='utf-8'))
limit=protocol['caps']['whole_detector_entries'];assert limit==len(inputs['cases'])+1==113
fd=os.open(str(D/'collector.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
torch.set_num_threads(4);random.seed(381031);np.random.seed(381031);torch.manual_seed(381031)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();device=torch.device('cuda:0')
loader_path=Path(protocol['loader_path']);assert sha(loader_path)==protocol['loader_sha256']
assert sha(protocol['checkpoint_path'])==protocol['checkpoint_sha256']
spec=importlib.util.spec_from_file_location('a39_original_dome_loader',loader_path)
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
model,postprocessor,transform,load_receipt=loader.load_model(receipt_path=D/'native_load_receipt.json')
before=loader.state_digest(model);assert before==load_receipt['state_digest']==protocol['state_digest']
model=model.to(device).eval();postprocessor=postprocessor.to(device).eval()
assert model.decoder.eval_idx==model.decoder.decoder.eval_idx==2 and len(model.decoder.decoder.layers)==3
assert model.decoder.min_num_select==250 and model.decoder.max_num_select==500
assert not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters())
assert postprocessor.use_focal_loss and postprocessor.num_classes==12 and not postprocessor.remap_mscoco_category and not postprocessor.deploy_mode
captures=D/'captures';captures.mkdir(exist_ok=False)
ledger=(D/'forward_entries.jsonl').open('x',encoding='utf-8')
entries=completed=0;current='';observed={}
def event(row):
    ledger.write(json.dumps(row,default=scalar)+'\n');ledger.flush();os.fsync(ledger.fileno())
def count_entry(module,arguments):
    global entries
    assert entries<limit;entries+=1;event(dict(event='author_detector_entry',entry=entries,scope=current))
def count_exit(module,arguments,output):
    global completed
    completed+=1;event(dict(event='author_detector_completed',entry=entries,scope=current))

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

def capture(row):
    observed.clear()
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
    return arrays,n,fine_map
model.register_forward_pre_hook(count_entry);model.register_forward_hook(count_exit)
model.decoder.decoder.register_forward_pre_hook(memory_hook)
model.decoder.dec_score_head[2].register_forward_pre_hook(query_hook)
model.decoder.dec_score_head[2].register_forward_hook(score_hook)
torch.cuda.reset_peak_memory_stats()
first=inputs['cases'][0];first_arrays=None;query_counts=[]
with torch.inference_mode():
    for row in inputs['cases']:
        current='case_%03d'%row['index']
        arrays,n,fine_map=capture(row)
        if first_arrays is None:
            first_arrays={k:v.copy() for k,v in arrays.items()}
            save_npz(D/'first_control_memory_and_head.npz',dict(fine_memory=fine_map[0].cpu().numpy(),
                head_weight=model.decoder.dec_score_head[2].weight.detach().cpu().numpy(),head_bias=model.decoder.dec_score_head[2].bias.detach().cpu().numpy()))
        save_npz(captures/(current+'.npz'),arrays);query_counts.append(n)
        assert torch.cuda.max_memory_allocated()<20*1024**3
        event(dict(event='case_exported',case=row['index'],split=row['split'],query_count=n,entries=entries,completed=completed))
        print(json.dumps(dict(case=row['index'],split=row['split'],entries=entries)),flush=True)
        del arrays,fine_map;observed.clear()
    current='repeat_first_new';arrays,n,fine_map=capture(first)
    for key,value in arrays.items():np.testing.assert_array_equal(value,first_arrays[key])
    save_npz(D/'repeat_first_new.npz',arrays)
assert entries==completed==limit==113
after=loader.state_digest(model);assert after==before
result=dict(status='COMPLETE_113_NEW_NATIVE_ENTRIES_WITH_FULL_FEATURE_REPEAT',entries=entries,completed=completed,
    unique_new_cases=112,new_train_cases=55,new_val_cases=57,query_counts=query_counts,
    initial_state_digest=before,final_state_digest=after,state_unchanged=True,
    full_query_feature_repeat_available=True,all_first_new_case_arrays_bitwise_repeat_equal=True,
    repeated_case_index=first['index'],peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),detector_optimizer_updates=0,data_probe_fits=0,
    source_commit=protocol['source_commit'],checkpoint_sha256=protocol['checkpoint_sha256'],
    files_sha256={str(q.relative_to(D)):sha(q) for q in sorted(list(captures.glob('*.npz'))+[D/'first_control_memory_and_head.npz',D/'repeat_first_new.npz',D/'native_load_receipt.json'])})
save_json(D/'collection_result.json',result);event(dict(event='collection_terminal_success',entries=entries,completed=completed));ledger.close()
print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','query_counts']},default=scalar),flush=True)
