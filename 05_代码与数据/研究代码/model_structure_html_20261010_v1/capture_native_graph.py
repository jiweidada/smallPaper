"""Exact native operation dataflow plus full uncollapsed module inventory."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
import weakref
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    serialized=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
    with Path(path).open('x',encoding='utf8') as f:f.write(serialized)

def tensor_items(v,path='value'):
    if isinstance(v,torch.Tensor):yield path,v
    elif isinstance(v,dict):
        for k,x in v.items():yield from tensor_items(x,path+'.'+str(k))
    elif isinstance(v,(tuple,list)):
        for i,x in enumerate(v):yield from tensor_items(x,path+'.'+str(i))

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def rng():return [torch.get_rng_state().clone()]+[s.clone() for s in torch.cuda.get_rng_state_all()]
def reset(seed=1723):torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)

class OperationGraph(TorchDispatchMode):
    def __init__(self,model,native):
        super().__init__();self.model=model;self.native=native;self.nodes=[];self.edges=[];self.refs={};self.stack=[];self.call_stack=[];self.handles=[];self.calls=[];self.call_counts={};self.inventory=[]
        attributes=('in_channels','out_channels','kernel_size','stride','padding','dilation','groups','eps','momentum','normalized_shape','num_heads','embed_dim','in_features','out_features','p','num_embeddings','embedding_dim','num_levels','num_points_list','hidden_dim','reg_max','reg_scale','num_queries')
        for name,module in model.named_modules():
            attrs={}
            for key in attributes:
                if hasattr(module,key):
                    value=getattr(module,key)
                    if isinstance(value,(str,int,float,bool,tuple,list)) and not isinstance(value,torch.Tensor):attrs[key]=value
            try:
                file=Path(inspect.getfile(type(module)));line=inspect.getsourcelines(type(module))[1]
                source=dict(file=file.relative_to(native).as_posix() if file.is_relative_to(native) else 'torch.nn/'+type(module).__name__,line=line)
            except (TypeError,OSError):source=None
            self.inventory.append(dict(path=name or 'model',class_name=type(module).__name__,attributes=attrs,direct_parameters=sum(p.numel() for p in module.parameters(recurse=False)),parameters_total=sum(p.numel() for p in module.parameters()),children=[k for k,_ in module.named_children()],source=source))
            self.handles.append(module.register_forward_pre_hook(self.pre(name or 'model'),with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.post(name or 'model'),with_kwargs=True))
        for name,t in list(model.named_parameters())+list(model.named_buffers()):self.refs[id(t)]=(weakref.ref(t),None,'state:'+name)

    def pre(self,name):
        def hook(module,args,kwargs):
            self.stack.append(name)
            n=self.call_counts.get(name,0);self.call_counts[name]=n+1
            inputs=[dict(argument=p,shape=list(v.shape),dtype=str(v.dtype),producer=self.source(v)[0],slot=self.source(v)[1]) for p,v in tensor_items((args,kwargs))]
            call=dict(path=name,invocation=n,class_name=type(module).__name__,call_id=len(self.calls),node_start=len(self.nodes),leaf=not bool(module._modules),inputs=inputs)
            self.calls.append(call);self.call_stack.append(call)
            return None
        return hook

    def post(self,name):
        def hook(module,args,kwargs,output):
            assert self.stack[-1]==name
            call=self.call_stack.pop();assert call['path']==name
            # Identity and p=0 Dropout execute without an aten operator. They
            # remain separate graph nodes and preserve actual input provenance.
            if call['leaf'] and len(self.nodes)==call['node_start']:
                node='n'+str(len(self.nodes))
                self.nodes.append(dict(id=node,kind='module_identity',operator=type(module).__name__+' · passthrough',module=name,module_stack=list(self.stack),call_id=call['call_id'],outputs=[dict(slot=p,shape=list(v.shape),dtype=str(v.dtype)) for p,v in tensor_items(output)],scalar_arguments=dict(native_tensor_operators=0)))
                for value in call['inputs']:
                    if value['producer'] is not None:self.edges.append(dict(source=value['producer'],target=node,source_slot=value['slot'],target_argument=value['argument'],shape=value['shape']))
                for path,value in tensor_items(output):self.refs[id(value)]=(weakref.ref(value),node,path)
            call['node_end']=len(self.nodes)
            self.stack.pop()
            call['outputs']=[dict(argument=p,shape=list(v.shape),dtype=str(v.dtype)) for p,v in tensor_items(output)]
            return None
        return hook

    def source(self,t,label='external tensor'):
        old=self.refs.get(id(t))
        if old is not None and old[0]() is t:return old[1],old[2]
        node='n'+str(len(self.nodes));self.nodes.append(dict(id=node,kind='input',operator=label,module=self.stack[-1] if self.stack else 'input',outputs=[dict(slot='value',shape=list(t.shape),dtype=str(t.dtype))],scalar_arguments={}))
        self.refs[id(t)]=(weakref.ref(t),node,'value');return node,'value'

    def bind_input(self,t,label):self.source(t,label)

    def argument(self,v):
        if isinstance(v,torch.Tensor):
            producer,slot=self.source(v)
            return dict(tensor=True,producer=producer,slot=slot,shape=list(v.shape),dtype=str(v.dtype))
        if isinstance(v,(tuple,list)):return [self.argument(x) for x in v]
        if isinstance(v,dict):return {str(k):self.argument(x) for k,x in v.items()}
        if isinstance(v,float) and not math.isfinite(v):return {'nonfinite_scalar':str(v)}
        if isinstance(v,(str,int,float,bool,type(None))):return v
        return str(v)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        inputs=[(p,v,*self.source(v)) for p,v in tensor_items((args,kwargs))]
        arguments=self.argument(dict(args=args,kwargs=kwargs))
        output=func(*args,**kwargs)
        node='n'+str(len(self.nodes))
        values=[dict(slot=p,shape=list(v.shape),dtype=str(v.dtype)) for p,v in tensor_items(output)]
        self.nodes.append(dict(id=node,kind='operator',operator=str(func),module=self.stack[-1] if self.stack else 'model',module_stack=list(self.stack),call_id=self.call_stack[-1]['call_id'] if self.call_stack else None,outputs=values,scalar_arguments=arguments))
        for argument,t,producer,slot in inputs:
            if producer is not None:self.edges.append(dict(source=producer,target=node,source_slot=slot,target_argument=argument,shape=list(t.shape)))
        for path,t in tensor_items(output):self.refs[id(t)]=(weakref.ref(t),node,path)
        return output

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def finish(self,outputs):
        sinks=[]
        for path,t in tensor_items(outputs):
            producer,slot=self.source(t);sinks.append(dict(path=path,producer=producer,slot=slot,shape=list(t.shape),dtype=str(t.dtype)))
        assert not self.stack and not self.call_stack
        covered={n.get('call_id') for n in self.nodes}
        assert all(not c['leaf'] or c['call_id'] in covered for c in self.calls)
        return dict(nodes=self.nodes,edges=self.edges,output_tensors=sinks,module_inventory=self.inventory,module_calls=self.calls,call_counts=self.call_counts,
            every_called_leaf_invocation_individually_represented=True,scope='All native tensor operators plus every called leaf module, including explicit no-op Identity/p=0 Dropout; every registered module individually inventoried. Parameter/buffer inputs listed in operator arguments rather than drawn as extra network layers. Fused aten operators retain their actual native names; mathematical attention expansion separately labeled in HTML.',extra_forward_head_calls=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--modes',default='eval,train');args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();tracer=None
    save(out/'start.json',dict(status='RUNNING_NATIVE_NETWORK_GRAPH',training_epochs_added=0,optimizer_steps=0,automatic_retry=False))
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        native=Path('/25085400120/DEIM/experiments/fine_residual_screen24_20261009_v2/native');sys.path.insert(0,str(native))
        sources={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        from engine.core import YAMLConfig
        from engine.backbone.hgnetv2 import HGNetv2
        from engine.deim.hybrid_encoder import HybridEncoder
        from engine.deim.dfine_decoder import DFINETransformer
        from engine.deim.deim import DEIM
        run=Path('/25085400120/DEIM/experiments/baseline_official_s_640_20261007_v1/seed1/run')
        checkpoint=run/'last.pth';assert sha(checkpoint)=='30febfa18e6e21dc67b16169bd26875b82340eb3435534719a963904dadf68b5'
        config=copy.deepcopy(json.loads((run/'args.json').read_text())['yaml_cfg']);fixture=copy.deepcopy(config);fixture.pop('__include__',None);fixture.pop('config',None)
        fixture['val_dataloader']['num_workers']=0;fixture['val_dataloader']['dataset']['transforms']['ops'].append(dict(type='ConvertBoxes',fmt='cxcywh',normalize=True));save(out/'fixture.yml',fixture)
        cfg=YAMLConfig(str(out/'fixture.yml'));x,target=cfg.val_dataloader.dataset[1];x=x.as_subclass(torch.Tensor).unsqueeze(0).cuda();targets=[{k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in target.items()}]
        model=DEIM(HGNetv2(**config['HGNetv2']),HybridEncoder(**config['HybridEncoder'],eval_spatial_size=[640,640]),DFINETransformer(**config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640]),num_classes=10).cuda()
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['last_epoch']==131;weights=payload['ema']['module'];results=[]
        modes=args.modes.split(',');assert modes and len(modes)==len(set(modes)) and all(m in ('eval','train') for m in modes)
        for mode in modes:
            model.load_state_dict(weights,strict=True);model.train(mode=='train');reset()
            with torch.no_grad():plain=model(x,targets=targets if mode=='train' else None)
            buffers={n:v.detach().clone() for n,v in model.named_buffers()};plain_rng=rng()
            model.load_state_dict(weights,strict=True);reset();tracer=OperationGraph(model,native);tracer.bind_input(x,'RGB image · 1×3×640×640')
            if mode=='train':
                for k in ('boxes','labels'):tracer.bind_input(targets[0][k],'GT '+k+' · training only')
            with torch.no_grad(),tracer:observed=model(x,targets=targets if mode=='train' else None)
            assert equal(plain,observed)
            assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
            assert all(torch.equal(a,b) for a,b in zip(plain_rng,rng()))
            graph=tracer.finish(observed);tracer.close();tracer=None
            graph.update(mode=mode,model='DEIM-D-FINE-S / HGNetv2-B0',input_hw=[640,640],image_id=int(target['image_id']),GT_rows=len(target['labels']),actual_output_equal=True,actual_buffer_update_equal=True,actual_RNG_equal=True,model_recipe={k:config[k] for k in ('HGNetv2','HybridEncoder','DFINETransformer')})
            save(out/f'graph_{mode}.json',graph);results.append(dict(mode=mode,nodes=len(graph['nodes']),operators=sum(n['kind']=='operator' for n in graph['nodes']),edges=len(graph['edges']),registered_modules=len(graph['module_inventory']),called_modules=len(graph['call_counts']),module_invocations=len(graph['module_calls']),output_equal=True))
            print(json.dumps(results[-1]),flush=True)
        model.load_state_dict(weights,strict=True)
        assert all(torch.equal(value.detach().cpu(),weights[name]) for name,value in model.state_dict().items())
        assert sources=={p.relative_to(native).as_posix():sha(p) for p in (native/'engine').rglob('*.py')}
        snapshot=out/'native_source';snapshot.mkdir()
        for rel in ['engine/backbone/hgnetv2.py','engine/deim/hybrid_encoder.py','engine/deim/dfine_decoder.py','engine/deim/dfine_utils.py','engine/deim/deim.py']:
            p=snapshot/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((native/rel).read_bytes())
        save(out/'completion.json',dict(status='PASS_NATIVE_OPERATION_GRAPHS_REQUESTED_MODES',results=results,modes=modes,checkpoint_sha256=sha(checkpoint),source_python_sha256=sources,code_sha256=sha(__file__),model_forwards=2*len(modes),training_epochs_added=0,optimizer_steps=0,model_parameters_unchanged=True,private_train_BN_updates_equal=True,native_sources_unchanged=True,elapsed_seconds=time.monotonic()-start))
        save(out/'manifest.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
    except BaseException:
        save(out/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',error=traceback.format_exc()));raise
    finally:
        if tracer is not None:tracer.close()


if __name__=='__main__':main()
