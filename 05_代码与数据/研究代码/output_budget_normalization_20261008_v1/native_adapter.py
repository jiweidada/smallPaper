"""Native architecture unchanged; final MAL matching/normalizer reused once."""
from copy import deepcopy
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim import DEIM
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from losses import output_budget_loss

ARMS = ('baseline', 'output_image', 'output_gt')


class BudgetNormDEIM(DEIM):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline'):
        if arm not in ARMS:
            raise ValueError(arm)
        config = deepcopy(recipe)
        super().__init__(
            HGNetv2(**config['HGNetv2']),
            HybridEncoder(**config['HybridEncoder'], eval_spatial_size=[640, 640]),
            DFINETransformer(**config['DFINETransformer'], num_classes=10,
                             eval_spatial_size=[640, 640]), num_classes=10)
        self.budget_arm = arm


class BudgetNormCriterion(DEIMCriterion):
    __inject__ = []
    __share__ = []

    def __init__(self, recipe, arm='baseline', margin=0.1, weight=0.1,
                 threshold=0.5, budget=300):
        if arm not in ARMS or budget != 300:
            raise ValueError('Unreviewed arm or budget')
        config = deepcopy(recipe)
        matcher = config.pop('matcher')
        matcher.pop('type', None)
        super().__init__(matcher=HungarianMatcher(**matcher, use_focal_loss=True),
                         num_classes=10, **config)
        self.budget_arm = arm
        self.budget_settings = dict(margin=margin, weight=weight, threshold=threshold, budget=budget)
        self.last_budget_diagnostics = None
        self._budget_final_outputs = None
        self._budget_final_indices = None
        self._budget_num_boxes = None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if outputs is self._budget_final_outputs and loss == 'mal':
            self._budget_final_indices = indices
            self._budget_num_boxes = num_boxes
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        if any(k in outputs for k in ('mgsa_outputs', 'o2m_t_outputs', 'num_queries_list', 'admission_aux')):
            raise ValueError('Only the fixed native single query group is supported')
        self._budget_final_outputs = outputs
        self._budget_final_indices = self._budget_num_boxes = None
        self.last_budget_diagnostics = None
        try:
            result = super().forward(outputs, targets, **kwargs)
            if self.budget_arm != 'baseline':
                if self._budget_final_indices is None or self._budget_num_boxes is None:
                    raise RuntimeError('Native final MAL matching/normalizer was not captured')
                extra, diagnostics = output_budget_loss(
                    outputs, targets, self._budget_final_indices, self._budget_num_boxes,
                    self.budget_arm.removeprefix('output_'), **self.budget_settings)
                if set(extra) & set(result):
                    raise RuntimeError('Loss name collision')
                result.update(extra)
                self.last_budget_diagnostics = diagnostics
            else:
                self.last_budget_diagnostics = []
            return result
        finally:
            self._budget_final_outputs = self._budget_final_indices = self._budget_num_boxes = None
