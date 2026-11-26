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
from engine.deim.deim import DEIM
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.deim.postprocessor import PostProcessor
from dual_view import DualViewDEIM
from query_coverage import DFINECoverageTransformer


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf8'))
    if cfg['train_batch_size'] != 8:
        raise ValueError('Future training is fixed at batch8 by user instruction')
    if cfg['epochs'] != 96 or cfg['seed'] != 1:
        raise ValueError('Prepared candidate configs use96 epochs/seed1; change via a new reviewed recipe')
    if cfg['kind'] == 'dual_view' and (cfg['tile_size'] != 640 or cfg['tile_stride'] != 512 or
            cfg['loss_view_weights'] != [.5, .5] or cfg['image_topk'] != 300 or cfg['merge_iou'] != .5):
        raise ValueError('Prepared dual-view geometry/loss/merge policy changed')
    return cfg


def native_recipe():
    return yaml.safe_load((ROOT / 'native_recipe.yml').read_text(encoding='utf8'))


def build_model(cfg):
    recipe = native_recipe()
    backbone = HGNetv2(**copy.deepcopy(recipe['HGNetv2']))
    encoder = HybridEncoder(**copy.deepcopy(recipe['HybridEncoder']), eval_spatial_size=[640, 640])
    decoder_kw = {**copy.deepcopy(recipe['DFINETransformer']), 'num_classes': 10,
                  'eval_spatial_size': [640, 640]}
    if cfg['kind'] == 'dual_view':
        decoder = DFINETransformer(**decoder_kw)
        return DualViewDEIM(backbone, encoder, decoder, cfg['arm'], cfg['fusion_width'])
    if cfg['kind'] == 'queries':
        decoder = DFINECoverageTransformer(decoder_kw, **cfg['coverage'])
        return DEIM(backbone, encoder, decoder, num_classes=10)
    raise ValueError(cfg['kind'])


def build_criterion():
    kwargs = copy.deepcopy(native_recipe()['DEIMCriterion'])
    matcher = kwargs.pop('matcher')
    matcher.pop('type')
    return DEIMCriterion(matcher=HungarianMatcher(**matcher), num_classes=10, **kwargs)


def build_postprocessor():
    return PostProcessor(num_classes=10, num_top_queries=300, use_focal_loss=True,
                         remap_mscoco_category=False, score_threshold=0.)


def load_common_init(model, path):
    """Strictly load every native key; only the new interaction may be missing."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('model', payload)
    if 'ema' in payload and 'model' not in payload:
        raise ValueError('Use the frozen common initialization, not a trained EMA checkpoint')
    expected = model.state_dict()
    extra = set(state) - set(expected)
    missing = set(expected) - set(state)
    if extra or any(not k.startswith('interaction.') for k in missing):
        raise ValueError(f'Initialization key mismatch: extra={sorted(extra)}, missing={sorted(missing)}')
    for name, value in state.items():
        if value.shape != expected[name].shape:
            raise ValueError(f'Shape mismatch for {name}')
    model.load_state_dict(state, strict=not missing)
    if not all(torch.equal(model.state_dict()[k].cpu(), v.cpu()) for k, v in state.items()):
        raise RuntimeError('Common state did not load exactly')
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'common_tensors_exact': len(state), 'new_tensors': sorted(missing)}


def verify_snapshot():
    manifest = json.loads((ROOT / 'native_source_manifest.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        actual = hashlib.sha256((ROOT / row['local']).read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise RuntimeError(f'Native source changed: {row["local"]}')
    return manifest

