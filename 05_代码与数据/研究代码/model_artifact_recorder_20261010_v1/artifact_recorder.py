"""Passive structural outputs from one actual forward; no extra prediction heads."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import torch

DENSE_PATTERNS = (
    r'^backbone\.stem$', r'^backbone\.stages\.\d+$', r'^backbone$',
    r'^encoder\.(input_proj|encoder|lateral_convs|fpn_blocks|downsample_convs|pan_blocks)\.\d+$',
    r'^encoder$', r'^decoder\.input_proj\.\d+$',
    r'^decoder\.(enc_output|enc_score_head|enc_bbox_head|pre_bbox_head|integral)$',
    r'^decoder\.dec_(score|bbox)_head\.\d+$',
    r'^decoder\.decoder\.layers\.\d+(\.(self_attn|cross_attn|gate))?$',
    r'^decoder\.decoder\.lqe_layers\.\d+$', r'^decoder\.decoder$', r'^decoder$', r'^$',
)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def tensors(value, path='value'):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for k in sorted(value):
            yield from tensors(value[k], path+'.'+str(k))
    elif isinstance(value, (tuple,list)):
        for i,v in enumerate(value):
            yield from tensors(v, path+'.'+str(i))

class ArtifactRecorder:
    """Every named-module output is inventoried; structural tensors saved exactly.

    This initial version supports evaluation only. A training history requires
    an explicit trainer integration to identify augmented GT, DN queries,
    matcher calls, losses and AMP/unscale/clip phases; not inferred here.
    """
    def __init__(self, model, max_tensor_bytes=512*1024*1024):
        if model.training:
            raise ValueError('Initial recorder is evaluation-only')
        self.model=model; self.max_tensor_bytes=max_tensor_bytes
        self.modules=dict(model.named_modules())
        self.handles=[]; self.restores=[]; self.active=False
        self.data={}; self.calls=[]; self.counts={}; self.bytes=0
        self.current_attention=None
        self.inventory={n:dict(class_name=type(m).__name__,direct_parameter_count=sum(p.numel() for p in m.parameters(recurse=False)),dense_output=any(re.search(p,n) for p in DENSE_PATTERNS)) for n,m in self.modules.items()}
        try:
            for name,module in self.modules.items():
                self.handles.append(module.register_forward_hook(self._hook(name)))
            for i,layer in enumerate(model.decoder.decoder.layers):
                name=f'decoder.decoder.layers.{i}'
                self.handles.append(layer.register_forward_pre_hook(self._reference(name)))
                attn=layer.cross_attn
                self.handles.append(attn.register_forward_pre_hook(self._attn_begin(name)))
                self.handles.append(attn.register_forward_hook(self._attn_end))
                original=attn.ms_deformable_attn_core
                self.restores.append((attn,'ms_deformable_attn_core',original))
                attn.ms_deformable_attn_core=self._core(original,name+'.sampling_fallback')
            import inspect
            native=inspect.getmodule(type(model.decoder.decoder.layers[0].cross_attn))
            if hasattr(native,'deformable_detr_msda_core_default'):
                original=native.deformable_detr_msda_core_default
                self.restores.append((native,'deformable_detr_msda_core_default',original))
                native.deformable_detr_msda_core_default=self._core(original,None)
            dec=model.decoder
            original=dec._select_topk
            self.restores.append((dec,'_select_topk',original))
            def select(memory,logits,anchors,k):
                result=original(memory,logits,anchors,k)
                if self.active:
                    if dec.query_select_method != 'default':
                        raise ValueError('Only bound default TopK mapping supported')
                    indices=logits.max(-1).values.topk(k,dim=-1).indices
                    assert torch.equal(result[0],memory.gather(1,indices[...,None].expand(-1,-1,memory.shape[-1])))
                    assert torch.equal(result[2],anchors.gather(1,indices[...,None].expand(-1,-1,anchors.shape[-1])))
                    self.put('native_topk.indices',indices)
                    self.put('native_topk.selected_logits',logits.gather(1,indices[...,None].expand(-1,-1,logits.shape[-1])))
                    self.put('native_topk.selected_anchors_unact',result[2])
                return result
            dec._select_topk=select
        except BaseException:
            self.close()
            raise

    def put(self,key,value):
        if key in self.data: raise ValueError('Duplicate tensor key: '+key)
        # Copy now, before downstream in-place operations can change storage.
        array=value.detach().cpu().contiguous().numpy().copy()
        self.bytes+=array.nbytes
        if self.bytes>self.max_tensor_bytes:
            raise RuntimeError('Capture storage cap exceeded; no silently truncated tensors')
        self.data[key]=array

    def _hook(self,name):
        def hook(module,args,output):
            if not self.active: return None
            index=self.counts.get(name,0); self.counts[name]=index+1
            event=dict(module=name,invocation=index,class_name=type(module).__name__,outputs=[])
            for path,value in tensors(output):
                array=value.detach().cpu().contiguous().numpy()
                numeric=np.issubdtype(array.dtype,np.number)
                entry=dict(path=path,shape=list(array.shape),dtype=str(array.dtype),bytes=int(array.nbytes))
                if numeric and array.size:
                    finite=np.isfinite(array)
                    entry['nonfinite']=int((~finite).sum())
                    vals=array[finite].astype(np.float64)
                    entry.update(min=float(vals.min()) if vals.size else None,max=float(vals.max()) if vals.size else None,mean=float(vals.mean()) if vals.size else None,std=float(vals.std()) if vals.size else None)
                if self.inventory[name]['dense_output']:
                    key=f'{name or "model"}@{index}.{path}'
                    self.put(key,value); entry['saved_key']=key
                event['outputs'].append(entry)
            self.calls.append(event)
            return None
        return hook

    def _reference(self,name):
        def hook(module,args):
            if self.active:
                self.put(name+'.native_reference_input',args[1])
            return None
        return hook

    def _attn_begin(self,name):
        def hook(module,args):
            if self.active:
                self.current_attention=name
                self.put(name+'.native_spatial_shapes',torch.tensor(args[3],dtype=torch.int64))
            return None
        return hook

    def _attn_end(self,module,args,output):
        self.current_attention=None
        return None

    def _core(self,original,name):
        def wrapped(*args,**kwargs):
            if self.active:
                prefix=name or (self.current_attention+'.sampling_fast' if self.current_attention else None)
                if prefix is None: raise ValueError('Sampling call missing stage identity')
                self.put(prefix+'.actual_locations',args[2])
                self.put(prefix+'.actual_weights',args[3])
            return original(*args,**kwargs)
        return wrapped

    def begin(self,metadata,input_tensor):
        if self.active or self.model.training: raise ValueError('Recorder state mismatch')
        self.data={}; self.calls=[]; self.counts={}; self.bytes=0
        self.metadata=metadata; self.active=True
        self.put('input.normalized_tensor',input_tensor)

    def end(self,path):
        if not self.active: raise ValueError('No active capture')
        self.active=False
        path=Path(path); path.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(path/'tensors.npz',**self.data)
        summary=dict(metadata=self.metadata,tensor_bytes_uncompressed=self.bytes,tensor_count=len(self.data),invocations=self.calls,called_module_counts=self.counts,module_inventory=self.inventory,additional_model_or_head_calls=0,original_forward_outputs_returned=True,feature_map_error_labels_assumed=False,training_history_recorded=False)
        (path/'trace.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        (path/'manifest.json').write_text(json.dumps({'files':{p.name:sha(p) for p in path.iterdir() if p.is_file()}},indent=2),encoding='utf8')
        return summary

    def close(self):
        self.active=False
        for h in self.handles: h.remove()
        self.handles=[]
        for obj,name,original in reversed(self.restores): setattr(obj,name,original)
        self.restores=[]
