import os, pathlib, sys, types, hashlib, json, copy, re
CODE=pathlib.Path(__file__).resolve().parent
WORK=CODE.parents[1]
OLD=WORK/'data/08_匹配聚合覆盖探针_20261019_v1'
OUT=WORK/'data/09_DN初始化冻结诊断_20261020_v1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['MPLBACKEND']='Agg'
sys.path[:0]=[str(CODE/'runtime'),str(CODE/'native')]
# A baseline module imports this profiler only for its __main__ demo. Fail closed
# if computation ever calls it; no baseline tensor operator is replaced.
profiler=types.ModuleType('calflops')
def forbidden_profiler(*a,**kw):raise RuntimeError('Profiling is outside this diagnostic')
profiler.calculate_flops=forbidden_profiler
sys.modules['calflops']=profiler
import numpy as np
import torch
import torch.nn.functional as F
import psutil
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.utils import inverse_sigmoid
from engine.deim.box_ops import box_cxcywh_to_xyxy,box_iou
from torch.optim import _functional as optim_functional

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
 return h.hexdigest()
def save_json(p,value):
 p=pathlib.Path(p)
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def cpu_tree(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return x
def resource_limits():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 p=psutil.Process();status={}
 try:p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);status['priority']='below_normal'
 except (OSError,psutil.Error) as e:status['priority_error']=str(e)
 try:
  allowed=p.cpu_affinity();chosen=allowed[-1::-2][:2];p.cpu_affinity(chosen);status['affinity']=p.cpu_affinity()
 except (OSError,psutil.Error) as e:status['affinity_error']=str(e)
 status.update(threads=torch.get_num_threads(),available_RAM_GiB=psutil.virtual_memory().available/2**30,torch=torch.__version__,cuda_runtime=torch.version.cuda)
 assert torch.version.cuda is None and not torch.cuda.is_available()
 assert status['available_RAM_GiB']>8
 return status
def model_hash(model):
 h=hashlib.sha256()
 for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
def build_model(y):
 return DEIM(HGNetv2(**copy.deepcopy(y['HGNetv2'])),HybridEncoder(**copy.deepcopy(y['HybridEncoder']),eval_spatial_size=[640,640]),DFINETransformer(**copy.deepcopy(y['DFINETransformer']),num_classes=10,eval_spatial_size=[640,640]),num_classes=10)

class RecordingCriterion(DEIMCriterion):
 def mark(self,out):
  self.ptrs={out['pre_outputs']['pred_boxes'].data_ptr():'ordinary',out['dn_pre_outputs']['pred_boxes'].data_ptr():'dn'}
  self.record={}
 def loss_boxes(self,outputs,targets,indices,num_boxes,boxes_weight=None):
  kind=self.ptrs.get(outputs['pred_boxes'].data_ptr())
  if kind:
   assert kind not in self.record
   self.record[kind]={'indices':cpu_tree(indices),'denominator':float(num_boxes),'boxes_weight':cpu_tree(boxes_weight)}
  return super().loss_boxes(outputs,targets,indices,num_boxes,boxes_weight)
def build_criterion(y):
 c=copy.deepcopy(y['DEIMCriterion']);m=c.pop('matcher');m.pop('type',None)
 c=RecordingCriterion(matcher=HungarianMatcher(**m,use_focal_loss=True),num_classes=10,**c)
 assert c.use_uni_set and c.boxes_weight_format is None and c.boxes_iou_type=='giou'
 assert c.weight_dict['loss_bbox']==5 and c.weight_dict['loss_giou']==2
 return c

def optimizer_binding(model,raw,y):
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 groups=[];visited=[]
 for cfg in y['optimizer']['params']:
  names=[n for n,p in named if re.findall(cfg['params'],n)]
  groups.append(names);visited+=names
 assert len(visited)==len(set(visited))
 groups.append([n for n,p in named if n not in set(visited)])
 original=raw['optimizer'];assert len(groups)==len(original['param_groups'])
 binding={}
 for names,group in zip(groups,original['param_groups']):
  assert len(names)==len(group['params'])
  for name,ident in zip(names,group['params']):
   state=original['state'].get(ident,{})
   if state:
    assert state['exp_avg'].shape==dict(named)[name].shape
    assert state['exp_avg_sq'].shape==dict(named)[name].shape
   binding[name]={'id':ident,'group':{k:v for k,v in group.items() if k!='params'},'state':cpu_tree(state)}
 names=['decoder.pre_bbox_head.layers.2.weight','decoder.pre_bbox_head.layers.2.bias']
 for n in names:assert binding[n]['state'] and n in dict(named)
 return names,{n:binding[n] for n in names}

def clip_factor(norm,maxnorm=.1):return min(1.,maxnorm/(norm+1e-6))
def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
def norm(g):return max(0.,dot(g,g))**.5
def adam_virtual(weights,grads,bindings,factor):
 """Only copied tensors are updated. Never create/call a training optimizer."""
 result=[]
 for name,theta,g in zip(bindings,weights,grads):
  z=bindings[name];group=z['group'];state=copy.deepcopy(z['state']);param=theta.detach().clone()
  maximum=[state['max_exp_avg_sq']] if group['amsgrad'] else []
  with torch.no_grad():
   optim_functional.adamw([param],[g.detach()*factor],[state['exp_avg']],[state['exp_avg_sq']],maximum,[state['step']],foreach=False,capturable=False,differentiable=False,fused=False,amsgrad=group['amsgrad'],beta1=group['betas'][0],beta2=group['betas'][1],lr=group['lr'],weight_decay=group['weight_decay'],eps=group['eps'],maximize=group['maximize'])
  result.append(param)
 return result

def indices_subset(cache,hard=False):
 indices=cache['ordinary']['indices']
 if not hard:return indices
 result=[]
 for i,(q,g) in enumerate(indices):
  anchor=cache['anchors'][:,cache['dn_split']:][i,q]
  gt=cache['targets'][i]['boxes'][g]
  ious=torch.diag(box_iou(box_cxcywh_to_xyxy(anchor),box_cxcywh_to_xyxy(gt))[0])
  keep=ious<.5;result.append((q[keep],g[keep]))
 return result
def head_boxes(cache,weights,dtype=torch.float64,branch='ordinary'):
 sl=slice(cache['dn_split'],None) if branch=='ordinary' else slice(None,cache['dn_split'])
 return (F.linear(cache['hidden'][:,sl].to(dtype),weights[0].to(dtype),weights[1].to(dtype))+cache['anchor_logits'][:,sl].to(dtype)).sigmoid()
def head_loss(cache,weights,criterion,hard=False,dtype=torch.float64,branch='ordinary'):
 indices=indices_subset(cache,hard) if branch=='ordinary' else cache['dn']['indices']
 targets=[{**v,'boxes':v['boxes'].to(dtype)} for v in cache['targets']]
 boxes=head_boxes(cache,weights,dtype,branch)
 item=cache[branch if branch=='dn' else 'ordinary']
 losses=DEIMCriterion.loss_boxes(criterion,{'pred_boxes':boxes},targets,indices,item['denominator'],None)
 total=5*losses['loss_bbox']+2*losses['loss_giou']
 return total,losses,boxes,indices
def metrics(cache,weights,criterion,hard=False):
 with torch.no_grad():loss,terms,boxes,indices=head_loss(cache,weights,criterion,hard)
 n=sum(len(v[0]) for v in indices)
 if n==0:return {'pairs':0,'loss_contribution':None}
 src=torch.cat([boxes[i,q] for i,(q,g) in enumerate(indices)])
 gt=torch.cat([t['boxes'][g].double() for t,(q,g) in zip(cache['targets'],indices)])
 iou=torch.diag(box_iou(box_cxcywh_to_xyxy(src),box_cxcywh_to_xyxy(gt))[0])
 return {'pairs':n,'loss_contribution':float(loss),'L1_unweighted_original_normalization':float(terms['loss_bbox']),'GIoU_loss_unweighted_original_normalization':float(terms['loss_giou']),'mean_IoU':float(iou.mean()),'IoU_ge_0_5_pairs':int((iou>=.5).sum()),'IoU_ge_0_75_pairs':int((iou>=.75).sum())}
