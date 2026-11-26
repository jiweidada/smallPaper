class ResidualDecoder(DFINETransformer):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        if source not in ('fine', 'fused'):
            raise ValueError(source)
        if self.query_select_method != 'default' or self.mgsa_groups or self.o2m_t:
            raise ValueError('Only the native default single query group is supported')
        if self.feat_strides != [8,16,32] or any(not isinstance(m, nn.Identity) for m in self.input_proj):
            raise ValueError('The bound three-level identity decoder projections are required')
        self.geometry_source = source
        self.geometry_residual = LocalGeometryResidual(self.hidden_dim)
        self._geometry_map = None
        self.audit_trace = False
        self.last_trace = None

    def forward(self, feats, targets=None):
        self.last_trace = None
        if not isinstance(feats, dict) or set(feats) != {'features','fine_p3'}:
            raise ValueError('Expected the explicit encoder feature bundle')
        self._geometry_map = feats['fine_p3'] if self.geometry_source == 'fine' else feats['features'][0]
        try:
            return super().forward(feats['features'], targets)
        finally:
            self._geometry_map = None

    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        output_memory = self.enc_output(valid_mask.to(memory.dtype) * memory)
        logits = self.enc_score_head(output_memory)
        _, ids = torch.topk(logits.max(-1).values, self.num_queries, dim=-1)
        selected_anchors = anchors.gather(1, ids[...,None].repeat(1,1,4))
        selected_logits = logits.gather(1,ids[...,None].repeat(1,1,logits.shape[-1])) if self.training else None
        semantic = output_memory.gather(1,ids[...,None].repeat(1,1,output_memory.shape[-1]))
        if self._geometry_map is None or list(self._geometry_map.shape[-2:]) != list(spatial_shapes[0]):
            raise ValueError('Geometry feature grid differs from the native P3 grid')
        native_raw = self.enc_bbox_head(semantic)
        delta = self.geometry_residual(semantic, self._geometry_map, ids, valid_mask)
        raw = native_raw + delta.to(native_raw.dtype)
        reference = raw + selected_anchors
        enc_boxes = [reference.sigmoid()] if self.training else []
        enc_logits = [selected_logits] if self.training else []
        content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0],1,1]) if self.learn_query_content else semantic.detach()
        if self.audit_trace:
            self.last_trace = {name:value.detach().clone() for name,value in dict(
                ids=ids,semantic=semantic,native_raw=native_raw,delta=delta,
                reference=reference,all_logits=logits,content=content).items()}
        reference = reference.detach()
        if denoising_bbox_unact is not None:
            reference = torch.concat([denoising_bbox_unact,reference],dim=1)
            content = torch.concat([denoising_logits,content],dim=1)
        return content,reference,enc_boxes,enc_logits,logits


class ResidualGeometryDEIM(DEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ARMS:
            raise ValueError(arm)
        config = deepcopy(recipe)
        backbone = HGNetv2(**config['HGNetv2'])
        encoder_type = HybridEncoder if arm == 'baseline' else FeatureBundleEncoder
        encoder = encoder_type(**config['HybridEncoder'], eval_spatial_size=[640,640])
        kwargs = dict(config['DFINETransformer'],num_classes=10,eval_spatial_size=[640,640])
        decoder = DFINETransformer(**kwargs) if arm == 'baseline' else ResidualDecoder(
            source='fine' if arm == 'fine_residual' else 'fused', **kwargs)
        super().__init__(backbone,encoder,decoder,num_classes=10)
        self.geometry_arm = arm
