"""Metadata only. Exact native run_candidate.py substring policy after base. removal."""
def parameter_policy(name,param):
    native=name[5:] if name.startswith('base.') else name
    return {'lr':.0001 if native.startswith('backbone.') else .0002,
            'weight_decay':0. if any(s in native for s in ('bn','norm','bias')) else .0001}
def groups_metadata(model):
    return {name:{**parameter_policy(name,p),'parameter_n':p.numel()} for name,p in model.named_parameters() if p.requires_grad}
