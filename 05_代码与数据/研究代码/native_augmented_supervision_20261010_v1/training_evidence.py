"""Passive actual supervision and backward observations for bounded batches."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import torch

PATTERNS=(r'^backbone\.stages\.\d+$',r'^encoder\.(fpn_blocks|pan_blocks)\.\d+$',r'^encoder$',r'^decoder\.(enc_output|enc_score_head|enc_bbox_head)$',r'^decoder\.decoder\.layers\.\d+$')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_items(value,path='value'):
    if isinstance(value,torch.Tensor):yield path,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from tensor_items(v,path+'.'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):yield from tensor_items(v,path+'.'+str(i))

def finite_stats(value):
    array=value.detach().cpu().float().numpy().copy()
    mask=np.isfinite(array);vals=array[mask].astype(np.float64)
    return dict(shape=list(array.shape),numel=array.size,nonfinite=int((~mask).sum()),finite_l2=float(np.linalg.norm(vals)),finite_max_abs=float(np.abs(vals).max()) if vals.size else None)

def scalar(value):
    number=float(value.detach().cpu())
    return number if np.isfinite(number) else None

class TrainingEvidenceRecorder:
    """Observer only: original matcher/loss calls and gradients returned unchanged.

    Source annotation identity is optional and must be provided by a separately
    verified augmentation provenance path. Missing identity is never invented.
    The object does not compute extra losses, matches, or backward passes.
    """
    def __init__(self,model,criterion,max_array_bytes=64*1024*1024):
        self.model=model;self.criterion=criterion;self.max_array_bytes=max_array_bytes
        self.handles=[];self.gradient_handles=[];self.active=False
        self.optimizer_steps=0;self.active_steps=0;self.errors=[]
        self.handles.append(model.register_forward_hook(self._model_output))
        self.handles.append(criterion.matcher.register_forward_hook(self._matcher,with_kwargs=True))
        self.handles.append(criterion.register_forward_hook(self._criterion_output,with_kwargs=True))
        for name,module in model.named_modules():
            if any(re.search(p,name) for p in PATTERNS):
                self.handles.append(module.register_forward_hook(self._activation(name)))
        self.original_get_loss=criterion.get_loss
        def get_loss(loss,outputs,targets,indices,num_boxes,**kwargs):
            result=self.original_get_loss(loss,outputs,targets,indices,num_boxes,**kwargs)
            if self.active:self._observe(lambda:self._loss_call(loss,outputs,targets,indices,num_boxes,result))
            return result
        criterion.get_loss=get_loss

    def _observe(self,fn):
        try:fn()
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))

    def put(self,name,value):
        if name in self.data:raise ValueError('Duplicate evidence key: '+name)
        array=value.detach().cpu().contiguous().numpy().copy()
        if self.array_bytes+array.nbytes>self.max_array_bytes:raise ValueError('Evidence cap exceeded')
        self.data[name]=array;self.array_bytes+=array.nbytes

    def begin(self,metadata,targets,loss_scale=1.0,source_annotation_ids=None,input_tensor=None):
        if self.active:raise ValueError('Overlapping sampled batches')
        if not self.model.training:raise ValueError('Training recorder requires training mode')
        if not np.isfinite(loss_scale) or loss_scale<=0:raise ValueError('Invalid loss scale')
        self.data={};self.array_bytes=0;self.events=[];self.errors=[];self.stage_map={};self.tensor_stage_map={}
        self.activation_calls={};self.gradient_events=[];self.active_steps=0
        self.parameter_gradient_snapshots={}
        self.metadata=dict(metadata);self.loss_scale=float(loss_scale)
        self.source_ids=source_annotation_ids;self.active=True
        self.metadata.update(loss_scale_before_backward=self.loss_scale,source_annotation_identity_supplied=source_annotation_ids is not None,activation_gradient_phase='during backward, before native parameter unscale/clip; stored statistics divided by supplied loss scale only')
        if input_tensor is not None:self.put('input.normalized_tensor',input_tensor)
        if source_annotation_ids is not None:
            assert len(source_annotation_ids)==len(targets)
            assert all(len(ids)==len(t['labels']) for ids,t in zip(source_annotation_ids,targets))
        for b,target in enumerate(targets):
            for name in ('boxes','labels','image_id','orig_size'):
                if name in target and isinstance(target[name],torch.Tensor):self.put(f'target.{b}.{name}',target[name])
            if source_annotation_ids is not None:self.put(f'target.{b}.verified_source_annotation_ids',torch.tensor(source_annotation_ids[b],dtype=torch.int64))

    def _register_stages(self,value,path):
        if isinstance(value,dict):
            if 'pred_boxes' in value and 'pred_logits' in value:
                self.stage_map[id(value)]=path
                self.tensor_stage_map[(id(value['pred_boxes']),id(value['pred_logits']))]=path
            for key,v in value.items():self._register_stages(v,path+'.'+str(key))
        elif isinstance(value,(list,tuple)):
            for i,v in enumerate(value):self._register_stages(v,path+'.'+str(i))

    def stage(self,outputs):
        return self.stage_map.get(id(outputs),self.tensor_stage_map.get((id(outputs.get('pred_boxes')),id(outputs.get('pred_logits'))),'UNRESOLVED'))

    def _gradient(self,name,value):
        if not value.requires_grad:return
        def hook(grad):
            if self.active:
                def collect():
                    normalized=grad.detach().float()/self.loss_scale
                    row=dict(name=name,**finite_stats(normalized))
                    self.gradient_events.append(row)
                    if normalized.ndim>=3:
                        axis=1 if normalized.ndim==4 else -1
                        self.put(name+'.gradient_row_l2',normalized.square().sum(axis).sqrt())
                self._observe(collect)
            return None
        self.gradient_handles.append(value.register_hook(hook))

    def _activation(self,name):
        def hook(module,args,output):
            if self.active:
                def collect():
                    n=self.activation_calls.get(name,0);self.activation_calls[name]=n+1
                    for key,value in tensor_items(output):
                        identity=f'{name}@{n}.{key}'
                        self.events.append(dict(kind='activation',name=identity,**finite_stats(value)))
                        self._gradient(identity,value)
                self._observe(collect)
            return None
        return hook

    def _model_output(self,module,args,outputs):
        if self.active:
            def collect():
                self._register_stages(outputs,'model')
                for key,value in tensor_items(outputs,'model'):
                    self.put(key,value)
                    if any(part in key for part in ('pred_boxes','pred_logits')):self._gradient(key,value)
            self._observe(collect)
        return None

    def _pairs(self,indices,targets,prefix):
        rows=[]
        for b,(q,g) in enumerate(indices):
            assert q.ndim==g.ndim==1 and len(q)==len(g)
            assert len(g)==0 or int(g.max())<len(targets[b]['labels'])
            self.put(prefix+f'.batch{b}.query_index',q)
            self.put(prefix+f'.batch{b}.GT_row_index',g)
            rows.append(dict(batch=b,pairs=len(q),unique_queries=len(q.unique()),unique_GT_rows=len(g.unique()),GT_rows=len(targets[b]['labels']),source_annotation_identity_available=self.source_ids is not None))
        return rows

    def _matcher(self,module,args,kwargs,result):
        if self.active:
            def collect():
                stage=self.stage(args[0]);n=sum(e['kind']=='matcher' for e in self.events)
                prefix=f'matcher{n}'
                pairs=self._pairs(result['indices'],args[1],prefix)
                self.events.append(dict(kind='matcher',stage=stage,prefix=prefix,batches=pairs,epoch=kwargs.get('epoch'),num_queries_list=kwargs.get('num_queries_list')))
            self._observe(collect)
        return None

    def _loss_call(self,loss,outputs,targets,indices,num_boxes,result):
        stage=self.stage(outputs);n=sum(e['kind']=='loss_call' for e in self.events);prefix=f'loss_call{n}'
        pairs=self._pairs(indices,targets,prefix)
        for b,(q,g) in enumerate(indices):
            self.put(prefix+f'.batch{b}.effective_labels',targets[b]['labels'][g])
            self.put(prefix+f'.batch{b}.target_boxes',targets[b]['boxes'][g])
            self.put(prefix+f'.batch{b}.pred_boxes',outputs['pred_boxes'][b,q])
            self.put(prefix+f'.batch{b}.pred_logits',outputs['pred_logits'][b,q])
        raw={name:scalar(v) for name,v in result.items()}
        self.events.append(dict(kind='loss_call',stage=stage,loss=loss,prefix=prefix,batches=pairs,num_boxes_normalizer=float(num_boxes),native_unweighted_loss_scalars=raw,is_DN='dn_' in stage or '.dn_outputs.' in stage,weight_dict={k:float(v) for k,v in self.criterion.weight_dict.items()}))

    def _criterion_output(self,module,args,kwargs,result):
        if self.active:
            def collect():
                for name,value in result.items():self.put('weighted_loss.'+name,value)
                self.events.append(dict(kind='criterion',weighted_loss_scalars={k:scalar(v) for k,v in result.items()},native_metadata={k:v for k,v in kwargs.items() if isinstance(v,(str,int,float,bool,type(None)))}))
            self._observe(collect)
        return None

    def parameter_gradients(self,phase):
        if not self.active:raise ValueError('Inactive batch')
        self.parameter_gradient_snapshots[phase]={name:None if p.grad is None else finite_stats(p.grad) for name,p in self.model.named_parameters()}

    def attach_optimizer(self,optimizer,phase='optimizer step pre-hook; caller metadata must specify native unscale/clip state'):
        def before(optimizer,args,kwargs):
            if self.active:self._observe(lambda:self.parameter_gradients(phase))
        def after(optimizer,args,kwargs):
            self.optimizer_steps+=1
            if self.active:self.active_steps+=1
        self.handles.extend((optimizer.register_step_pre_hook(before),optimizer.register_step_post_hook(after)))

    def end(self,path):
        if not self.active:raise ValueError('Inactive batch')
        self.active=False
        for h in self.gradient_handles:h.remove()
        self.gradient_handles=[]
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(path/'arrays.npz',**self.data)
        result=dict(status='CAPTURED' if not self.errors else 'OBSERVATION_ERRORS_NATIVE_CALCULATION_UNCHANGED',metadata=self.metadata,events=self.events,gradients=self.gradient_events,parameter_gradient_snapshots=self.parameter_gradient_snapshots,actual_optimizer_steps_in_sample=self.active_steps,errors=self.errors,array_bytes=self.array_bytes,arrays=len(self.data),additional_matcher_loss_forward_backward_calls=0,source_annotation_identity_supplied=self.source_ids is not None,global_source_identity_verified_by_recorder=False,caller_source_identity_claim=self.metadata.get('source_identity_verified',False),gradient_cause_interpreted=False)
        (path/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        (path/'manifest.json').write_text(json.dumps({'files':{p.name:sha(p) for p in path.iterdir() if p.is_file()}},indent=2),encoding='utf8')
        return result

    def close(self):
        self.active=False
        for h in self.handles+self.gradient_handles:h.remove()
        self.handles=[];self.gradient_handles=[]
        self.criterion.get_loss=self.original_get_loss

class SampledTrainingLoader:
    """Wrap a native loader without replaying or changing its input batches.

    metadata_fn supplies epoch/attempt and the GradScaler scale BEFORE backward.
    A selected batch closes only after the native loop resumes the iterator,
    i.e. after its native loss/backward/step/EMA sequence. Global annotation IDs
    are unknown unless a separately verified provenance_fn supplies them.
    """
    def __init__(self,original,recorder,output,selected,metadata_fn,provenance_fn=None):
        self.original=original;self.recorder=recorder;self.output=Path(output)
        self.selected=selected;self.metadata_fn=metadata_fn;self.provenance_fn=provenance_fn
    def __len__(self):return len(self.original)
    def __getattr__(self,name):return getattr(self.original,name)
    def __iter__(self):
        for i,batch in enumerate(self.original):
            sample=self.selected(i)
            if sample:
                metadata=self.metadata_fn(i)
                provenance=self.provenance_fn(batch) if self.provenance_fn is not None else None
                self.recorder.begin(metadata,batch[1],metadata['loss_scale_before_backward'],provenance,input_tensor=batch[0])
            try:yield batch
            finally:
                if sample:self.recorder.end(self.output/f"epoch{metadata['epoch_one_based']:02d}_batch{i:05d}")
