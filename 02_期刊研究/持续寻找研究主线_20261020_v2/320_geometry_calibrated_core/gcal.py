"""316/317 G prototype; read-only reuse of the frozen 311 implementation.

Only ten class-null scalars are added to R. The learned association is still
evaluated for auxiliary supervision, but never supplies G's generation field.
No loader, optimizer, M0, native detector, or task loss is implemented here.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FROZEN_CORE = Path(__file__).resolve().parents[1] / "311_unit_instance_core/core.py"
FROZEN_SHA256 = "8cd5dbc61052c7d3c3b4058b4d25b150b906ecc06c6eb005793c737cef3f17ef"
assert hashlib.sha256(FROZEN_CORE.read_bytes()).hexdigest() == FROZEN_SHA256
_spec = importlib.util.spec_from_file_location("_frozen_unit_core311", FROZEN_CORE)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)  # Definitions only; 311 core has no executable main.


class GeometryCalibratedHead(base.CompletionHead):
    """Fixed geometric generation cost plus ten trainable class-null costs.

    `plan` retains the original auxiliary-plan interface. `generation_plan` is
    explicitly separate; `aux_plan` is an alias of `plan`, not a second solve.
    A later optimizer must put class_null in the no-weight-decay group. This
    prototype contains no optimizer and cannot itself enforce an LR/decay rule.
    """
    def __init__(self):
        super().__init__("R")
        self.arm = "G"
        self.class_null = nn.Parameter(torch.zeros(base.CLASSES))

    def generation_plan(self, mass, safe_boxes, safe_classes, valid):
        xy = self.association.xy.to(mass.dtype)[None, :, None, :]
        outside = (safe_boxes[:, None, :, :2] - xy).clamp_min(0.)
        outside = outside + (xy - safe_boxes[:, None, :, 2:]).clamp_min(0.)
        wh = safe_boxes[..., 2:] - safe_boxes[..., :2]
        geometry = (outside / wh[:, None]).sum(dim=-1)
        null_cost = self.class_null[safe_classes]
        return base.independent_cap(mass, geometry, null_cost, safe_classes, valid)

    def forward_features(self, f, boxes, classes, scores, valid):
        assert f.device.type == "cpu" and f.shape[1:] == (base.DIM, base.SIDE, base.SIDE)
        assert boxes.shape[0] == f.shape[0] and boxes.dtype == scores.dtype == f.dtype
        tokens, safe_boxes, safe_classes, counts = base.candidate_inputs(
            boxes, classes, scores, valid)
        mass = F.softplus(self.mass_head(f)).flatten(2)
        a, aux_plan = self.association(f, tokens, safe_boxes, safe_classes, valid, mass)
        generation_plan = self.generation_plan(mass, safe_boxes, safe_classes, valid)
        field = generation_plan["residual"]
        dv, db = base.density_inputs(field)
        q = self.queries[None].expand(f.shape[0], -1, -1)
        image = f.flatten(2).transpose(1, 2)
        for block in self.blocks:
            # R-style blocks have no P cross-attention, so `a` is not a generator path.
            assert block.p_attention is None
            q = block(q, image, dv, db, a, valid)
        return {"pred_boxes": (self.box_head(q) + self.reference_logits).sigmoid(),
                "pred_logits": self.classifier(q), "query_states": q,
                "features": f, "mass": mass, "plan": aux_plan,
                "aux_plan": aux_plan, "generation_plan": generation_plan,
                "counts": counts, "tokens15": tokens,
                "density_field": field, "density_value": dv, "density_bias": db}


def copy_R_common(source, target):
    """Copy the exact R common state; keep G's separately initialized ten scalars."""
    assert source.arm == "R" and isinstance(target, GeometryCalibratedHead)
    result = target.load_state_dict(source.state_dict(), strict=False)
    assert result.missing_keys == ["class_null"] and not result.unexpected_keys

