"""Phase-B integration helpers. No optimizer construction or training at import."""
from semantic_branch import ClassificationReadout,paired_semantic_loss
from self_capacity import SelfCapacityReadout
from parameter_groups_v2 import groups_metadata

def make_branch(arm):
    if arm=='S80':return SelfCapacityReadout()
    if arm not in ['M','Q']:raise ValueError('Core arms are S80/M/Q.')
    head=ClassificationReadout(mode='mean' if arm=='M' else 'attention')
    if arm=='M':
        head.query.requires_grad_(False);head.key.requires_grad_(False)
    return head

def prepare_frozen_phase_b(detector):
    # Calling detector.train() alone would train frozen base BN; this is explicit.
    detector.eval();detector.base.requires_grad_(False)
    detector.semantic_branch.train();detector.base.decoder.train()
    return detector

def phase_b_loss(detector,outputs,targets,visible,matcher):
    if any(p.requires_grad for p in detector.base.parameters()):raise ValueError('Phase-B native model must be frozen.')
    if len(detector.semantic_scores)!=2:raise ValueError('Exactly global/local semantic scores required.')
    if detector.base.training or not detector.base.decoder.training:raise ValueError('Require base eval and decoder train/DN.')
    semantic={'global':detector.semantic_scores[0],'local':detector.semantic_scores[1]}
    raw,details=paired_semantic_loss(semantic,outputs,targets,visible,matcher)
    return .25*raw,details

def phase_b_parameter_metadata(detector):
    if any(p.requires_grad for p in detector.base.parameters()):raise ValueError('Frozen native parameters required.')
    rows=groups_metadata(detector)
    if any(not name.startswith('semantic_branch.') for name in rows):raise ValueError('Non-head parameter would be optimized.')
    return rows
