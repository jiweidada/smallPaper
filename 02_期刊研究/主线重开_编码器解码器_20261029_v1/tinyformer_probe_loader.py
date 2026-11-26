"""Model-only official import, with explicit non-numerical compatibility changes.

Official source files remain byte-for-byte unchanged. Package initializers that
import training/data infrastructure are not run. The official registry, YAML
merge, constructors, numerical layers and postprocessor are used unchanged.
"""
from pathlib import Path
import __future__
import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import sys
import types
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
PREFIX = 'tinyformer_official_probe'
IMPORT_RECEIPTS = []


class OfficialLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        data = self.path.read_bytes()
        tree = ast.parse(data, filename=str(self.path))
        changes = ['defer_type_annotations_for_python39']
        if self.path == SOURCE / 'engine/deim/hybrid_encoder.py':
            # DropPath is imported but never referenced anywhere in this file.
            assert not any(isinstance(n, ast.Name) and n.id == 'DropPath'
                           and isinstance(n.ctx, ast.Load) for n in ast.walk(tree))
            before = len(tree.body)
            tree.body = [n for n in tree.body if not (
                isinstance(n, ast.ImportFrom) and n.module == 'timm.models.layers'
                and [a.name for a in n.names] == ['DropPath'])]
            assert len(tree.body) == before - 1
            changes.append('omit_provably_unused_DropPath_import_only')
        code = compile(tree, str(self.path), 'exec',
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)
        module.__file__ = str(self.path)
        IMPORT_RECEIPTS.append({'path': str(self.path.relative_to(SOURCE)),
                                'source_sha256': hashlib.sha256(data).hexdigest(),
                                'compatibility_changes': changes})
        exec(code, module.__dict__)


class OfficialFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(PREFIX + '.'):
            return None
        relative = fullname[len(PREFIX) + 1:].split('.')
        stem = SOURCE.joinpath('engine', *relative)
        if stem.is_dir() and (stem / '__init__.py').is_file():
            p = stem / '__init__.py'
            return importlib.util.spec_from_file_location(fullname, p, loader=OfficialLoader(p),
                                                          submodule_search_locations=[str(stem)])
        p = stem.with_suffix('.py')
        if p.is_file():
            return importlib.util.spec_from_file_location(fullname, p, loader=OfficialLoader(p))
        return None


def namespace(name, directory):
    m = types.ModuleType(name)
    m.__package__ = name
    m.__path__ = [str(directory)]
    sys.modules[name] = m
    return m


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        data = value.detach().cpu().contiguous()
        h.update(name.encode() + b'\0' + str(data.dtype).encode() + b'\0')
        h.update(str(tuple(data.shape)).encode() + b'\0' + data.numpy().tobytes())
    return h.hexdigest()


def load_official_model():
    asset_receipt = json.loads((ROOT / 'receipts/asset_checkpoint_check.json').read_text())
    assert asset_receipt['source_files'] == 187
    assert asset_receipt['weight_sha256'] == '0ba0e997e4bdfae7800f6f3e55fa714b0302efd4aa5327663290c9705eb8a199'
    manifest = json.loads((ROOT / 'assets/probe_package_manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((SOURCE / row['path']).read_bytes()).hexdigest() == row['sha256']
    sys.meta_path.insert(0, OfficialFinder())
    namespace(PREFIX, SOURCE / 'engine')
    core = namespace(PREFIX + '.core', SOURCE / 'engine/core')
    namespace(PREFIX + '.backbone', SOURCE / 'engine/backbone')
    namespace(PREFIX + '.deim', SOURCE / 'engine/deim')
    workspace = importlib.import_module(PREFIX + '.core.workspace')
    core.register = workspace.register
    yu = importlib.import_module(PREFIX + '.core.yaml_utils')
    for name in ['backbone.dinov3_adapter', 'deim.hybrid_encoder', 'deim.deim_decoder',
                 'deim.deim', 'deim.postprocessor']:
        importlib.import_module(PREFIX + '.' + name)
    cfg = yu.load_config(str(SOURCE / 'configs/tinyformer/visdrone/tinyformer_dinov3_s_coco_pbm_visdrone.yml'))
    assert cfg['DEIM']['backbone'] == 'DINOv3SSAs_4Scale'
    assert cfg['DINOv3SSAs_4Scale']['name'] == 'vit_tiny'
    cfg['DINOv3SSAs_4Scale']['weights_path'] = None  # Full strict state loaded below.
    global_cfg = yu.merge_config(cfg, inplace=False, overwrite=False)
    model = workspace.create(cfg['model'], global_cfg)
    pp = workspace.create(cfg['postprocessor'], global_cfg)
    checkpoint = torch.load(ROOT / 'assets/TinyFormer-S-pbm-visdrone.pth',
                            map_location='cpu', weights_only=True)
    state = checkpoint['ema']['module']  # Same choice as official torch_inf.py:122.
    assert state['decoder.enc_score_head.weight'].shape == (10, 192)
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model.eval()
    pp.eval()
    assert model.decoder.num_classes == 10 and model.decoder.num_queries == 300
    assert len(model.decoder.decoder.layers) == 4
    assert model.encoder.out_strides == [8, 16, 32]
    assert cfg['eval_spatial_size'] == [640, 640]
    assert cfg['val_dataloader']['dataset']['transforms']['ops'] == [
        {'type': 'Resize', 'size': [640, 640]},
        {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True},
        {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}]
    receipt = {'status': 'STRICT_AUTHOR_EMA_MODEL_LOADED', 'state_keys': len(state),
               'checkpoint_last_epoch': checkpoint.get('last_epoch'),
               'parameters': sum(p.numel() for p in model.parameters()),
               'trainable_parameters_before_inference_disable': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'state_digest': state_digest(model), 'imports': IMPORT_RECEIPTS,
               'backbone': cfg['DINOv3SSAs_4Scale'], 'encoder': cfg['HybridEncoder_4Scale'],
               'decoder': cfg['DEIMTransformer'], 'eval_spatial_size': cfg['eval_spatial_size'],
               'numerical_source_edits': 0, 'training_updates': 0, 'model_forwards': 0}
    (ROOT / 'receipts/model_load_check.json').write_text(json.dumps(receipt, indent=2))
    model.requires_grad_(False)
    return model, pp, receipt


if __name__ == '__main__':
    _, _, receipt = load_official_model()
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['imports', 'backbone', 'encoder', 'decoder']}))
