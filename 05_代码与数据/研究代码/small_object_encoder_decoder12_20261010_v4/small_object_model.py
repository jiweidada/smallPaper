"""Small-object encoder/decoder with separate semantic and geometry states.

Preserves the bound native DEIM interfaces, losses, query budget and FDR chart.
Added modules are initialized after all native modules, in an isolated RNG
scope. load_native_initialization also copies loaded semantic-layer weights
into the independent geometry layers for exact initial predictions.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer,TransformerDecoder,MLP,inverse_sigmoid
from engine.deim.dfine_utils import weighting_function,distance2bbox
from engine.deim.deim import DEIM
from geometry_memory import GeometryField,GeometryMemoryRead


class SmallObjectEncoder(HybridEncoder):
    def forward(self,features):
        if len(features)!=4:raise ValueError('Expected native HGNetv2 stages0/1/2/3')
        semantic=super().forward(features[1:])
        geometry=self.geometry_field(features[0],semantic[0])
        return {'semantic':semantic,'geometry':geometry}


class GeometryProposalMLP(MLP):
    def forward(self,selected_semantic):
        native=super().forward(selected_semantic)
        if self.geometry_context is None or self.selected_anchor_context is None:
            raise RuntimeError('Proposal geometry context is missing')
        correction=self.geometry_proposal(selected_semantic,self.geometry_context,self.selected_anchor_context.sigmoid())
        return native+correction.to(native.dtype)


class DualStateDecoder(TransformerDecoder):
    def forward(self,target,ref_points_unact,memory,spatial_shapes,bbox_head,score_head,query_pos_head,pre_bbox_head,integral,up,reg_scale,attn_mask=None,memory_mask=None,dn_meta=None):
        if self.geometry_context is None:raise RuntimeError('Decoder geometry context is missing')
        semantic=geometry=target
        geometry_detach=previous_corners=0
        value=self.value_op(memory,None,None,memory_mask,spatial_shapes)
        boxes=[];scores=[];corners=[];refs=[]
        project=self.project if hasattr(self,'project') else weighting_function(self.reg_max,up,reg_scale)
        reference=ref_points_unact.sigmoid()
        for i,(semantic_layer,geometry_layer) in enumerate(zip(self.layers,self.geometry_layers)):
            pos=query_pos_head(reference).clamp(min=-10,max=10)
            semantic=semantic_layer(semantic,reference.unsqueeze(2),value,spatial_shapes,attn_mask,pos)
            geometry=geometry_layer(geometry,reference.unsqueeze(2),value,spatial_shapes,attn_mask,pos)
            geometry=geometry+self.geometry_attention[i](geometry,self.geometry_context,reference)
            if i==0:
                pre_boxes=(pre_bbox_head(geometry)+inverse_sigmoid(reference)).sigmoid()
                pre_scores=score_head[0](semantic)
                fixed_reference=pre_boxes.detach()
            pred_corners=bbox_head[i](geometry+geometry_detach)+previous_corners
            refined=distance2bbox(fixed_reference,integral(pred_corners,project),reg_scale)
            if self.training or i==self.eval_idx:
                logits=self.lqe_layers[i](score_head[i](semantic),pred_corners)
                boxes.append(refined);scores.append(logits);corners.append(pred_corners);refs.append(fixed_reference)
                if not self.training:break
            previous_corners=pred_corners
            reference=refined.detach();geometry_detach=geometry.detach()
        return torch.stack(boxes),torch.stack(scores),torch.stack(corners),torch.stack(refs),pre_boxes,pre_scores


class SmallObjectTransformer(DFINETransformer):
    def _select_topk(self,memory,logits,anchors,topk):
        selected=super()._select_topk(memory,logits,anchors,topk)
        self.enc_bbox_head.selected_anchor_context=selected[2]
        return selected

    def forward(self,features,targets=None):
        if self.mgsa_groups or self.o2m_t:raise ValueError('This prototype binds the native nongroup model only')
        self.enc_bbox_head.geometry_context=features['geometry']
        self.decoder.geometry_context=features['geometry']
        try:return super().forward(features['semantic'],targets)
        finally:
            self.enc_bbox_head.geometry_context=None
            self.enc_bbox_head.selected_anchor_context=None
            self.decoder.geometry_context=None


EXTRA_PREFIXES=('encoder.geometry_field.','decoder.enc_bbox_head.geometry_proposal.','decoder.decoder.geometry_layers.','decoder.decoder.geometry_attention.')


class SmallObjectDEIM(DEIM):
    __inject__=[]
    __share__=[]

    def __init__(self,recipe,variant='fine',initialization_seed=713):
        if variant not in ('fine','degraded'):raise ValueError(variant)
        config=deepcopy(recipe)
        # The original constructor order is preserved, including backbone init.
        backbone=HGNetv2(**config['HGNetv2'])
        encoder=HybridEncoder(**config['HybridEncoder'],eval_spatial_size=[640,640])
        decoder=DFINETransformer(**config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640])
        super().__init__(backbone,encoder,decoder,num_classes=10)
        if encoder.in_channels!=[256,512,1024] or decoder.hidden_dim!=256 or decoder.num_queries!=300:
            raise ValueError('Expected the bound HGNetv2-B0 / native C256 S model')
        if len(decoder.decoder.layers)!=3 or decoder.decoder.layer_scale!=1 or decoder.decoder.eval_idx!=2:
            raise ValueError('This first implementation supports exactly3 normal C256 layers')
        if any(isinstance(m,nn.Dropout) and m.p!=0 for m in self.modules()):
            raise ValueError('Exact RNG-preserving dual-state initialization requires native dropout0')
        self.variant=variant
        self.backbone.return_idx=[0,1,2,3]
        self.encoder.__class__=SmallObjectEncoder
        self.decoder.__class__=SmallObjectTransformer
        self.decoder.enc_bbox_head.__class__=GeometryProposalMLP
        self.decoder.decoder.__class__=DualStateDecoder
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also changes CUDA generators. All added modules
            # are built on CPU; seed only that generator inside its saved scope.
            torch.random.default_generator.manual_seed(initialization_seed)
            self.encoder.geometry_field=GeometryField(degraded=variant=='degraded')
            self.decoder.enc_bbox_head.geometry_proposal=GeometryMemoryRead(output_channels=4)
            self.decoder.decoder.geometry_layers=nn.ModuleList([deepcopy(m) for m in self.decoder.decoder.layers])
            self.decoder.decoder.geometry_attention=nn.ModuleList([GeometryMemoryRead() for _ in range(3)])
        self.decoder.enc_bbox_head.geometry_context=None
        self.decoder.enc_bbox_head.selected_anchor_context=None
        self.decoder.decoder.geometry_context=None

    def forward(self,images,targets=None):
        features=self.encoder(self.backbone(images))
        outputs=self.decoder(features,targets)
        if self.training and self.o2m_c is not None:outputs['o2m_c']=self.o2m_c(features['semantic'])
        return outputs


def load_native_initialization(model,native_state):
    current=model.state_dict();extra=set(current)-set(native_state)
    if set(native_state)-set(current) or any(not key.startswith(EXTRA_PREFIXES) for key in extra):
        raise ValueError('Unexpected native/extra parameter namespaces')
    result=model.load_state_dict(native_state,strict=False)
    if set(result.missing_keys)!=extra or result.unexpected_keys:raise ValueError('Common state load mismatch')
    # Do this only for a common/native initialization, never on trained candidate
    # checkpoint reload: learned geometry-state weights must stay independent.
    for semantic,geometry in zip(model.decoder.decoder.layers,model.decoder.decoder.geometry_layers):
        geometry.load_state_dict(semantic.state_dict(),strict=True)
    loaded=model.state_dict()
    if any(value.dtype!=loaded[key].dtype or not torch.equal(value,loaded[key].to(value.device)) for key,value in native_state.items()):
        raise ValueError('Common weights were changed')
    return {'native_state_keys':len(native_state),'extra_state_keys':sorted(extra),'native_state_exact':True,'independent_geometry_layer_copies_exact':True,'variant':model.variant,'parameters':sum(p.numel() for p in model.parameters()),'extra_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith(EXTRA_PREFIXES))}

