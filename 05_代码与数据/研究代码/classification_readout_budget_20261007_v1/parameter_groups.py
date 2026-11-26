"""Metadata grouping only, mirrors run_candidate.py; no optimizer construction."""
def parameter_policy(name,param):
    native=name[5:] if name.startswith('base.') else name
    return {'lr':.0001 if native.startswith('backbone.') else .0002,
            'weight_decay':0. if param.ndim==1 or native.endswith('.bias') else .0001}

def groups_metadata(model):
    return {name:{**parameter_policy(name,p),'parameter_n':p.numel()} for name,p in model.named_parameters() if p.requires_grad}
