"""317v4 §3 residual-change metric. Real measurement requires supplied trained mass.

No checkpoint loader, training entry, or implicit uniform-mass fallback exists.
Every window, including empty P, contributes to the mean.
"""
import importlib
_ctx = importlib.import_module('342_context')
import torch
import geom342 as geom
import core

def initial_mass(windows):
    return torch.full((windows, 10, 256), 1./256)

def plans(mass, boxes_norm, classes, valid, kappa):
    assert mass.shape == (boxes_norm.shape[0], 10, 256)
    assert bool(torch.isfinite(mass).all()) and bool((mass >= 0).all())
    assert bool((mass.sum((1, 2)) > 0).all()), 'undefined residual/mass for zero total mass'
    assert kappa.shape == (10,)
    # Padding boxes must already be legal unit boxes; they have zero supply.
    cost = geom.geometry_cost(boxes_norm)
    return core.independent_cap(mass, cost, kappa[classes], classes, valid)

def measure(mass, boxes_norm, classes, valid, kappa, kappa_initial=None):
    p = plans(mass, boxes_norm, classes, valid, kappa)
    p0 = plans(mass, boxes_norm, classes, valid, torch.zeros_like(kappa))
    denom = mass.sum((1, 2))
    ratios = p['residual'].sum((1, 2)) / denom
    ratios0 = p0['residual'].sum((1, 2)) / denom
    diffs = (ratios-ratios0).abs()
    initial = torch.zeros_like(kappa) if kappa_initial is None else kappa_initial
    return {'delta': float(diffs.mean()), 'per_window_delta': diffs.detach().tolist(),
            'residual_over_mass': ratios.detach().tolist(), 'windows': mass.shape[0],
            'kappa_final': kappa.detach().tolist(), 'kappa_initial': initial.detach().tolist(),
            'kappa_displacement': (kappa-initial).detach().tolist()}

def classify(delta):
    assert delta >= 0
    if delta < .01:
        return 'G 未有效校准'
    if delta < .03:
        return '校准能力受限'
    return 'G 可作正式对照'

def measure_initial(boxes_norm, classes, valid, kappa):
    return measure(initial_mass(boxes_norm.shape[0]), boxes_norm, classes, valid, kappa)

def measure_trained_mass(mass, boxes_norm, classes, valid, kappa, *, split, image_ids):
    assert split == 'val548' and len(image_ids) == 548 and len(set(image_ids)) == 548
    assert mass.shape[0] == 548
    result = measure(mass, boxes_norm, classes, valid, kappa)
    result['verdict'] = classify(result['delta'])
    result['image_ids'] = image_ids
    result['mass_source'] = 'caller-supplied trained mass outputs; provenance must accompany calling receipt'
    return result
