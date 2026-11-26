"""Isolated subclasses of the frozen native detector and criterion."""
from copy import deepcopy
import torch
from torch import nn
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from losses import admission_losses, output_budget_loss

ARMS = ('baseline', 'admission_quality', 'admission_cover', 'output_budget')


class AdmissionDecoder(DFINETransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.query_select_method != 'default' or self.mgsa_groups or self.o2m_t:
            raise ValueError('Only the verified native default single query group is supported')
        # No extra RNG consumption; same native initialization across arms.
        self.admission_weight = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self._admission_work = None

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        residual = nn.functional.linear(memory, self.admission_weight).squeeze(-1)
        priority = outputs_logits.max(-1).values.detach() + residual.to(outputs_logits.dtype)
        ids = torch.topk(priority, topk, dim=-1).indices
        chosen_memory = memory.gather(1, ids[..., None].expand(-1, -1, memory.shape[-1]))
        chosen_anchors = outputs_anchors_unact.gather(1, ids[..., None].expand(-1, -1, 4))
        chosen_logits = outputs_logits.gather(1, ids[..., None].expand(-1, -1, self.num_classes)) if self.training else None
        if self.training:
            self._admission_work = {'memory': memory, 'anchors': outputs_anchors_unact,
                                    'ids': ids, 'priority': priority, 'residual': residual,
                                    'classes': outputs_logits.detach().argmax(-1)}
        return chosen_memory, chosen_logits, chosen_anchors

    def _get_decoder_input(self, *args, **kwargs):
        result = super()._get_decoder_input(*args, **kwargs)
        if self.training:
            work = self._admission_work
            if work is None:
                raise RuntimeError('Admission selection was not recorded')
            memory, ids, anchors = work['memory'], work['ids'], work['anchors']
            with torch.no_grad():
                b, n, _ = memory.shape
                selected_mask = torch.zeros((b, n), dtype=torch.bool, device=memory.device)
                selected_mask.scatter_(1, ids, True)
                rest = torch.arange(n, device=memory.device).expand(b, -1)[~selected_mask].reshape(b, n - ids.shape[1])
                values = self.enc_bbox_head(memory.detach().gather(1, rest[..., None].expand(-1, -1, memory.shape[-1])))
                values = (values + anchors.gather(1, rest[..., None].expand(-1, -1, 4))).sigmoid()
                boxes = anchors.new_empty((b, n, 4))
                boxes.scatter_(1, rest[..., None].expand(-1, -1, 4), values.to(boxes.dtype))
                boxes.scatter_(1, ids[..., None].expand(-1, -1, 4), result[2][0].detach().to(boxes.dtype))
                work['boxes'] = boxes
                work['valid'] = torch.isfinite(anchors).all(-1).expand(b, -1)
            del work['memory'], work['anchors']
        return result

    def forward(self, feats, targets=None):
        self._admission_work = None
        try:
            output = super().forward(feats, targets)
            if self.training:
                output['admission_aux'] = self._admission_work
            return output
        finally:
            self._admission_work = None


class BudgetDEIM(DEIM):
    # The native registry otherwise inherits DEIM's backbone/encoder injection.
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ARMS:
            raise ValueError(arm)
        config = deepcopy(recipe)
        backbone = HGNetv2(**config['HGNetv2'])
        encoder = HybridEncoder(**config['HybridEncoder'], eval_spatial_size=[640, 640])
        decoder_kw = dict(config['DFINETransformer'], num_classes=10, eval_spatial_size=[640, 640])
        decoder_type = AdmissionDecoder if arm.startswith('admission_') else DFINETransformer
        decoder = decoder_type(**decoder_kw)
        super().__init__(backbone, encoder, decoder, num_classes=10)
        self.budget_arm = arm


class BudgetCriterion(DEIMCriterion):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline', margin=0.1, weight=0.1,
                 residual_weight=1e-4, threshold=0.5, budget=300):
        if arm not in ARMS or budget != 300:
            raise ValueError('Unreviewed arm or budget')
        config = deepcopy(recipe)
        matcher = config.pop('matcher')
        matcher.pop('type', None)
        matcher = HungarianMatcher(**matcher, use_focal_loss=True)
        super().__init__(matcher=matcher, num_classes=10, **config)
        self.budget_arm, self.budget_margin = arm, margin
        self.budget_weight, self.residual_weight = weight, residual_weight
        self.budget_threshold, self.output_budget = threshold, budget
        self.last_budget_diagnostics = None
        self._budget_final_outputs = None
        self._budget_final_indices = None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if outputs is self._budget_final_outputs and loss == 'mal':
            self._budget_final_indices = indices
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        if any(k in outputs for k in ('mgsa_outputs', 'o2m_t_outputs', 'num_queries_list')):
            raise ValueError('Only fixed native query count and one query group are supported')
        clean = {k: v for k, v in outputs.items() if k != 'admission_aux'}
        self._budget_final_outputs, self._budget_final_indices = clean, None
        self.last_budget_diagnostics = None
        try:
            result = super().forward(clean, targets, **kwargs)
            if self.budget_arm.startswith('admission_'):
                extra, diag = admission_losses(outputs['admission_aux'], targets,
                    self.budget_arm.removeprefix('admission_'), self.output_budget,
                    self.budget_margin, self.budget_weight, self.residual_weight, self.budget_threshold)
            elif self.budget_arm == 'output_budget':
                if self._budget_final_indices is None:
                    raise RuntimeError('Native final MAL indices were not captured')
                extra, diag = output_budget_loss(clean, targets, self._budget_final_indices,
                    self.output_budget, self.budget_margin, self.budget_weight, self.budget_threshold)
            else:
                extra, diag = {}, []
            if set(result) & set(extra):
                raise RuntimeError('Loss name collision')
            result.update(extra)
            self.last_budget_diagnostics = diag
            return result
        finally:
            self._budget_final_outputs = self._budget_final_indices = None
