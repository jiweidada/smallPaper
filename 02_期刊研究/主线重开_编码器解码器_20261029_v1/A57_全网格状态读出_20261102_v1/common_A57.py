"""Shared frozen source-native host, initialization and runtime accounting."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time
import torch
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'host_compat_v2'))
from model import MambaSOD
from native_variants_v2 import install
from native_data import VisDroneDataset,collate,targets_to_device,seed_all,fp32_tree
from native_criterion import HungarianMatcher,SetCriterion

MODEL_COUNTER={'entries':0,'completions':0,'decoder_entries_including_recomputation':0}
EVENT_JOURNAL=None


def set_event_journal(path):
    global EVENT_JOURNAL
    EVENT_JOURNAL=Path(path)
    with EVENT_JOURNAL.open('x',encoding='utf-8'):pass


def event(kind,**extra):
    if EVENT_JOURNAL is not None:
        with EVENT_JOURNAL.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(event=kind,model_counter=dict(MODEL_COUNTER),
                monotonic=time.monotonic(),pid=os.getpid(),**extra),allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text('utf-8'))


def save_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def validate_files(protocol_name):
    plan=load(ROOT/protocol_name)
    assert all(digest(ROOT/n)==h for n,h in plan['files_sha256'].items())
    assert digest(ROOT/'common_initialization.pth')==plan['initialization_sha256']
    for name,row in plan.get('external_files',{}).items():
        assert digest(Path(name))==row['sha256'],name
    return plan


def make_model(arm,forward_limit):
    assert arm in ['B','M','Q']
    model=MambaSOD(img_size=640,patch_size=16,vim_depth=24,vim_embed_dim=192,d_state=16,
                   out_channels=256,bifpn_repeats=3,num_queries=400,num_decoder_layers=6,
                   d_ffn=1024,num_classes=10,dropout=0.0,drop_path_rate=0.1,
                   num_groups=6,use_ca=False,use_sa=False,use_fpn=False)
    initial=torch.load(ROOT/'common_initialization.pth',map_location='cpu',weights_only=True)
    assert initial['seed']==20261101
    assert initial['parameter_tensor_digest']=='bbd38e8e9c87e4f164158177cd33252a34382f479cadf0fdd837011f95ef8a28'
    model.load_state_dict(initial['model'],strict=True)
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in initial['model'].items())
    install(model,arm)
    assert model.state_dict().keys()==initial['model'].keys()
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in initial['model'].items())
    assert sum(p.numel() for p in model.parameters())==72697131
    del initial
    for layer in model.decoder.layers:
        original=layer.forward
        def wrapper(tgt,query_pos,memories,pos_embeds,num_groups=1,source=original):
            def core(a,b,c,d):
                MODEL_COUNTER['decoder_entries_including_recomputation']+=1
                return source(a,b,c,d,num_groups=num_groups)
            if not torch.is_grad_enabled() or not model.training:
                return core(tgt,query_pos,memories,pos_embeds)
            return checkpoint(core,tgt,query_pos,memories,pos_embeds,
                              use_reentrant=False,preserve_rng_state=True)
        layer.forward=wrapper
    def entered(module,args):
        MODEL_COUNTER['entries']+=1
        event('model_forward_entered')
        assert MODEL_COUNTER['entries']<=forward_limit
    def completed(module,args,outputs):
        MODEL_COUNTER['completions']+=1
        event('model_forward_completed')
    model.register_forward_pre_hook(entered)
    model.register_forward_hook(completed)
    return model


def make_criterion():
    matcher=HungarianMatcher(cost_class=2,cost_bbox=2,cost_giou=5,num_classes=10,
                             focal_alpha=.25,focal_gamma=1.5)
    return SetCriterion(10,matcher,{'loss_ce':2,'loss_bbox':2,'loss_giou':5},
                        ['labels','boxes'],focal_alpha=.25,focal_gamma=1.5)


def make_optimizer(model):
    return torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if n.startswith('encoder.backbone.')],
         'lr':1e-5,'initial_lr':1e-5},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.backbone.')],
         'lr':1e-4,'initial_lr':1e-4}
    ],weight_decay=1e-4,betas=(.9,.999))


def lr_ratio(update):
    assert 0<=update<9708
    if update<500:return (update+1)/500
    return .1+.9*.5*(1+math.cos(math.pi*(update-500)/(9708-500)))


def weighted_loss(criterion,outputs,targets):
    losses=criterion(fp32_tree(outputs),targets)
    value=sum(v*criterion.weight_dict[k.split('_aux')[0]] for k,v in losses.items())
    assert torch.isfinite(value) and all(torch.isfinite(v) for v in losses.values())
    return value,losses


def assert_output(outputs,batch,queries):
    assert outputs['pred_logits'].shape==(batch,queries,10)
    assert outputs['pred_boxes'].shape==(batch,queries,4)
    assert len(outputs['aux_outputs'])==5
    for row in [outputs]+outputs['aux_outputs']:
        assert torch.isfinite(row['pred_logits']).all() and torch.isfinite(row['pred_boxes']).all()


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def input_digest(images,targets):
    h=hashlib.sha256(images.contiguous().numpy().tobytes())
    for row in targets:
        for n in sorted(row):
            h.update(n.encode());h.update(row[n].contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_digest():
    h=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    h.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
    return h.hexdigest()
