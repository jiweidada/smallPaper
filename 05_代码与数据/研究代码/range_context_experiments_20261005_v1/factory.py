"""Model construction and explicit checkpoint migration; no training launcher."""
import copy
import hashlib
import json
from pathlib import Path
import torch
import yaml
import bootstrap
from bootstrap import ROOT
from engine.backbone.hgnetv2 import HGNetv2
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.postprocessor import PostProcessor
from dual_range import DualRangeDEIM


def native_recipe():
    return yaml.safe_load((ROOT / 'native_recipe.yml').read_text(encoding='utf8'))


def load_config(path):
    from range_context import VARIANTS
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf8'))
    if cfg['variant'] not in VARIANTS or cfg['kind'] != 'dual_view':
        raise ValueError('Only the four frozen range-context variants are allowed')
    required = {'epochs':96, 'seed':1, 'train_batch_size':8, 'workers':4,
                'tile_size':640, 'tile_stride':512, 'merge_iou':.5, 'image_topk':300,
                'loss_view_weights':[.5,.5]}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError('Frozen range experiment recipe changed')
    return cfg


def build_model(cfg=None):
    cfg = cfg or {'variant': 'adaptive_range_global_context'}
    recipe = native_recipe()
    backbone = HGNetv2(**copy.deepcopy(recipe['HGNetv2']))
    encoder = HybridEncoder(**copy.deepcopy(recipe['HybridEncoder']), eval_spatial_size=[640, 640])
    decoder = DFINETransformer(**copy.deepcopy(recipe['DFINETransformer']), num_classes=10,
                               eval_spatial_size=[640, 640])
    return DualRangeDEIM(backbone, encoder, decoder, cfg['variant'])


def build_criterion():
    kw = copy.deepcopy(native_recipe()['DEIMCriterion'])
    matcher = kw.pop('matcher')
    matcher.pop('type')
    return DEIMCriterion(matcher=HungarianMatcher(**matcher), num_classes=10, **kw)


def build_postprocessor():
    return PostProcessor(num_classes=10, num_top_queries=300, use_focal_loss=True,
                         remap_mscoco_category=False, score_threshold=0.)


def load_common_init(model, path, expected_sha256):
    """Require the chosen frozen initialization hash; never use a trained EMA.

No initialization is downloaded or changed. Native tensors must all be
present; the two added interaction namespaces alone may be absent.
"""
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Common initialization hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'ema' in payload or payload.get('epoch', -1) not in (-1, None):
        raise ValueError('Use the frozen common initialization, not a trained checkpoint')
    state = payload.get('model', payload)
    expected = model.state_dict()
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if extra or any(not k.startswith(('interaction.', 'range_interaction.')) for k in missing):
        raise ValueError(f'Common initialization mismatch: extra={sorted(extra)}, missing={sorted(missing)}')
    model.load_state_dict(state, strict=False)
    if not all(torch.equal(model.state_dict()[k].cpu(), v.cpu()) for k, v in state.items()):
        raise RuntimeError('Common tensors did not load exactly')
    if torch.count_nonzero(model.range_interaction.restore.weight):
        raise ValueError('Reader residual must start at zero')
    return {'purpose': 'common_initialization_no_training_started', 'sha256': actual,
            'common_tensors_exact': len(state), 'new_tensors': sorted(missing)}


def load_independent_ema_for_cpu_probe(model, path):
    """A trained independent EMA is only a CPU probe, never common training init."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('config', {}).get('arm', 'independent') != 'independent':
        raise ValueError('Probe checkpoint must be an independent dual-view model')
    state = payload['ema']['module']
    expected = model.state_dict()
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if extra or any(not k.startswith('range_interaction.') for k in missing):
        raise ValueError(f'Unexpected checkpoint differences: extra={sorted(extra)}, missing={sorted(missing)}')
    model.load_state_dict(state, strict=False)
    if not all(torch.equal(model.state_dict()[k].cpu(), v.cpu()) for k, v in state.items()):
        raise RuntimeError('Native tensors did not load exactly')
    return {'purpose': 'CPU_probe_not_training_initialization', 'common_tensors_exact': len(state),
            'new_tensors': sorted(missing), 'epoch_zero_based': payload.get('epoch'),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def verify_snapshot():
    manifest = json.loads((ROOT / 'native_source_manifest.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        if hashlib.sha256((ROOT / row['local']).read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError(f'Native snapshot changed: {row["local"]}')
    return manifest
