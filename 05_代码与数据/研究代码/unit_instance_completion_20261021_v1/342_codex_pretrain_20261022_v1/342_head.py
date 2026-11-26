"""Adapters preserve frozen 311_v2/320_v2 modules and state-dict layout.

Only the two geometry calculations are replaced by the shared geom342 entry.
No frozen file or module-global method is modified.
"""
import importlib, math
C = importlib.import_module('342_context')
import torch
import core
import gcal
import geom342 as G

class SharedAssociation(core.Association):
    def forward(self, f, tokens, boxes, classes, valid, mass):
        a = self.encoder(tokens)
        image = f.flatten(2).transpose(1,2)
        content = -torch.einsum('bid,bjd->bij', self.key(image), self.query(a)) / math.sqrt(core.DIM)
        geometry = G.geometry_cost(boxes, self.xy)
        self.last_geometry = geometry.detach()
        global_f = image.mean(1)[:,None,:].expand(-1,a.shape[1],-1)
        null_cost = self.null(torch.cat((a,global_f),-1)).squeeze(-1)
        plan = core.independent_cap(mass,content+geometry,null_cost,classes,valid)
        return a,plan

def _replace_association(model):
    rng = torch.get_rng_state()
    old = model.association
    new = SharedAssociation()
    new.load_state_dict(old.state_dict(), strict=True)
    assert torch.equal(new.xy, old.xy)
    model.association = new
    torch.set_rng_state(rng)

class CompletionHead342(core.CompletionHead):
    def __init__(self, arm):
        super().__init__(arm)
        _replace_association(self)

class GeometryCalibratedHead342(gcal.GeometryCalibratedHead):
    def __init__(self):
        super().__init__()
        _replace_association(self)

    def generation_plan(self, mass, safe_boxes, safe_classes, valid):
        geometry = G.geometry_cost(safe_boxes, self.association.xy)
        self.last_generation_geometry = geometry.detach()
        return core.independent_cap(mass, geometry, self.class_null[safe_classes],safe_classes,valid)
