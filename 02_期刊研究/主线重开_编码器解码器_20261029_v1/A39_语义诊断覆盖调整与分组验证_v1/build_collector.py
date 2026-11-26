from pathlib import Path
import ast,hashlib,json,sys,textwrap
sys.stdout.reconfigure(encoding='utf-8')
D=Path(__file__).resolve().parent;OLD=D.parent/'A38_细尺度与查询语义成对诊断_v1'
source=(OLD/'collect_features.py').read_text(encoding='utf-8');tree=ast.parse(source)
names=['memory_hook','query_hook','score_hook','native_arrays','load_case']
functions='\n\n'.join(ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names)
start=source.index('        x=load_case(row); output=model(x)');stop=source.index("        if row['index']==0:",start)
feature_block=textwrap.indent(textwrap.dedent(source[start:stop]),'    ')
header='''from __future__ import annotations
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
    ledger.write(json.dumps(row,default=scalar)+'\\n');ledger.flush();os.fsync(ledger.fileno())
def count_entry(module,arguments):
    global entries
    assert entries<limit;entries+=1;event(dict(event='author_detector_entry',entry=entries,scope=current))
def count_exit(module,arguments,output):
    global completed
    completed+=1;event(dict(event='author_detector_completed',entry=entries,scope=current))
'''
driver='''
def capture(row):
    observed.clear()
__FEATURE_BLOCK__
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
'''
driver=driver.replace('__FEATURE_BLOCK__',feature_block.rstrip())
text=header+'\n'+functions+'\n'+driver
parsed=ast.parse(text,feature_version=(3,9))
for name in names:
    old=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    new=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(old)==ast.dump(new),name
with (D/'collect_features.py').open('x',encoding='utf-8') as f:f.write(text)
with (D/'native_operator_preservation.json').open('x',encoding='utf-8') as f:
    json.dump(dict(status='NUMERICAL_HELPERS_AND_FEATURE_BLOCK_UNCHANGED_FROM_A38',helpers=names,
        feature_block_identical=True,source_original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        collector_sha256=hashlib.sha256(text.encode()).hexdigest(),NPZ_writer='Previouslyverifiedstandardnonpicklewriter',
        source_model_modifications=False,new_author_model_calls=0),f,indent=2)
print(json.dumps(dict(status='A39_COLLECTOR_CREATED_STATIC_ONLY',model_call_cap=113,unchanged_numerical_helpers=len(names))))
