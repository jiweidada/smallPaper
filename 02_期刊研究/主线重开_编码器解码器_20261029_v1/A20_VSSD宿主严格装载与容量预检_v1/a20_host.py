"""Pinned native VSSD + native Dome interfaces; construction never calls forward."""
from pathlib import Path
import ast
import __future__
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deps'))
import torch
import torch.nn as nn

WEIGHT = ROOT.parent / 'vssd_tiny_host_assets_20261030_v1/vssd_tiny_mesa.pth'
DOME = ROOT.parent / 'dome_assets_v1'
SOURCE = ROOT / 'vssd_source'
SEED = 20261030


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def state_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        v = v.detach().cpu().contiguous()
        h.update(k.encode() + b'\0' + str(v.dtype).encode() + b'\0')
        h.update(str(tuple(v.shape)).encode() + b'\0' + v.numpy().tobytes())
    return h.hexdigest()


def module_from_source(name, path, omit_profiler=False):
    data = path.read_bytes()
    tree = ast.parse(data, filename=str(path))
    if omit_profiler:
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'fvcore.nn']
        assert len(imports) == 1
        names = {a.asname or a.name for a in imports[0].names}
        active = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VMAMBA2')
        # Profiler names occur only in the optional .flops() method, which is never called.
        outside = ast.Module(body=[n for n in tree.body if n is not imports[0]], type_ignores=[])
        loaded = [(n.id, n.lineno) for n in ast.walk(outside) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names]
        flop = next(n for n in active.body if isinstance(n, ast.FunctionDef) and n.name == 'flops')
        assert all(flop.lineno <= line <= flop.end_lineno for _, line in loaded), loaded
        tree.body.remove(imports[0])
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec', flags=__future__.annotations.compiler_flag, dont_inherit=True), module.__dict__)
    return module


class MetadataCfgNode(dict):
    """Inert unused checkpoint metadata; no custom unpickling behaviour."""
    pass


def strict_backbone():
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    assert sha(WEIGHT) == '853f5719558dffcf8a629f4eb07b18914e397dc33f122525b1e2933d138cd545'
    package = types.ModuleType('a20vssd')
    package.__path__ = [str(SOURCE)]
    sys.modules['a20vssd'] = package
    module_from_source('a20vssd.mamba_util', SOURCE / 'mamba_util.py')
    native = module_from_source('a20vssd.mamba2', SOURCE / 'mamba2.py', omit_profiler=True)
    config = module_from_source('a20vssd.config', SOURCE / 'config.py')._C.clone()
    config.merge_from_file(str(SOURCE / 'vssd_tiny.yaml'))
    tree = ast.parse((SOURCE / 'models_init.py').read_text(encoding='utf-8'))
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_vssd_model')
    scope = {'VMAMBA2': lambda **kwargs: kwargs}
    exec(compile(ast.Module(body=[build], type_ignores=[]), 'author_build_vssd_model', 'exec'), scope)
    kwargs = scope['build_vssd_model'](config)
    torch.manual_seed(SEED)
    classification = native.VMAMBA2(**kwargs)
    with torch.serialization.safe_globals([(MetadataCfgNode, 'yacs.config.CfgNode')]):
        checkpoint = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    ema = checkpoint['model_ema']
    result = classification.load_state_dict(ema, strict=True)
    assert not result.missing_keys and not result.unexpected_keys and len(ema) == 347
    for k, v in classification.state_dict().items():
        assert torch.equal(v, ema[k]), k
    classification_digest = state_digest(classification)
    classification_parameters = sum(p.numel() for p in classification.parameters())
    torch.manual_seed(SEED)
    backbone = native.Backbone_VMAMBA2(out_indices=(0, 1, 2, 3), pretrained=None, **kwargs)
    removed = {'head.weight', 'head.bias', 'norm.weight', 'norm.bias'}
    added = {f'outnorm{i}.{suffix}' for i in range(4) for suffix in ['weight', 'bias']}
    shared = set(ema) - removed
    state = backbone.state_dict()
    assert set(state) == shared | added
    target = {k: (ema[k] if k in shared else v) for k, v in state.items()}
    mapped = backbone.load_state_dict(target, strict=True)
    assert not mapped.missing_keys and not mapped.unexpected_keys
    for k in shared:
        assert torch.equal(backbone.state_dict()[k], ema[k]), k
    receipt = {
        'classification_state_keys': len(ema), 'classification_parameters': classification_parameters,
        'classification_strict_load': True, 'all_classification_tensors_equal_ema': True,
        'classification_state_digest': classification_digest,
        'classifier_to_backbone_removed': sorted(removed),
        'backbone_new_norm_keys': sorted(added), 'backbone_shared_keys_equal_ema': len(shared),
        'mapped_backbone_strict_load': True, 'backbone_state_keys': len(state),
        'constructor_kwargs_from_author_build': kwargs,
        'compatibility': ['Python39 deferred annotations only', 'Unused fvcore profiler import omitted; flops() never called',
                          'Optional mamba_ssm absent; configured noncausal path uses native PyTorch arithmetic'],
        'preprocessing': {'input': 'RGB float32 /255, torchvision v2 Resize([800,800])',
                          'backbone_only_mean': [0.485, 0.456, 0.406], 'backbone_only_std': [0.229, 0.224, 0.225],
                          'neck_img_input': 'original resized RGB /255, no ImageNet normalization'},
    }
    del checkpoint, classification, ema, state, target
    return backbone, receipt


class NormalizedBackbone(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.native((x - self.mean) / self.std)


def build_host():
    backbone, receipt = strict_backbone()
    spec = importlib.util.spec_from_file_location('a20_dome_imports', DOME / 'dome_probe_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)  # Its load_model() is deliberately never called.
    loader.verified_source()
    sys.meta_path.insert(0, loader.AuthorFinder())
    for name, rel in [('src', 'src'), ('src.core', 'src/core'), ('src.nn', 'src/nn'),
                      ('src.zoo', 'src/zoo'), ('src.zoo.dome', 'src/zoo/dome'),
                      ('src.misc', 'src/misc'), ('tools', 'tools')]:
        loader.namespace(name, loader.SOURCE / rel)
    workspace = importlib.import_module('src.core.workspace')
    sys.modules['src.core'].register = workspace.register
    yu = importlib.import_module('src.core.yaml_utils')
    # Only these two unchanged native single-process distributed queries are needed by criterion.
    dist_path = loader.SOURCE / 'src/misc/dist_utils.py'
    dist_tree = ast.parse(dist_path.read_text(encoding='utf-8'))
    functions = [n for n in dist_tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ['is_dist_available_and_initialized', 'get_world_size']]
    assert len(functions) == 2
    dist_module = types.ModuleType('src.misc.dist_utils')
    dist_module.torch = torch
    sys.modules['src.misc.dist_utils'] = dist_module
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(dist_path), 'exec'), dist_module.__dict__)
    for name in ['hybrid_encoder', 'dome_decoder', 'dome', 'matcher', 'dome_criterion']:
        importlib.import_module('src.zoo.dome.' + name)
    cfg = yu.load_config(str(loader.SOURCE / 'configs/dome/Dome-S-VisDrone.yml'))
    cfg['HybridEncoder']['in_channels'] = [96, 192, 384, 768]
    assert cfg['num_classes'] == 12 and cfg['eval_spatial_size'] == [800, 800]
    assert cfg['DomeTransformer']['min_num_select'] == 250 and cfg['DomeTransformer']['max_num_select'] == 500
    global_cfg = yu.merge_config(cfg, inplace=False, overwrite=False)
    torch.manual_seed(SEED)
    encoder = workspace.create('HybridEncoder', global_cfg)
    decoder = workspace.create('DomeTransformer', global_cfg)
    criterion = workspace.create('DomeCriterion', global_cfg)
    model = sys.modules['src.zoo.dome.dome'].DOME(NormalizedBackbone(backbone), encoder, decoder)
    assert model.encoder.out_strides == [4, 8, 16, 32]
    assert len(model.decoder.decoder.layers) == 3
    model.eval()
    receipt.update(utc=datetime.now(timezone.utc).isoformat(), status='STRICT_CPU_HOST_CONSTRUCTION_OK_NO_FORWARD',
                   host_parameters=sum(p.numel() for p in model.parameters()),
                   host_state_digest=state_digest(model), neck_decoder_initialization='fresh seed 20261030; no detector checkpoint load',
                   source_imports=loader.IMPORTS, config=cfg, complete_model_forwards=0, optimizer_updates=0,
                   CUDA_initialized=torch.cuda.is_initialized(), criterion_native=True,
                   dist_utils_scope='two unchanged native query functions extracted; no distributed training claimed')
    assert receipt['CUDA_initialized'] is False
    return model, criterion, loader.input_transform()[0], receipt


if __name__ == '__main__':
    try:
        model, criterion, transform, receipt = build_host()
        write(ROOT / 'strict_host_load.json', receipt)
        print(json.dumps({k: receipt[k] for k in ['status', 'classification_state_keys', 'classification_parameters',
                         'backbone_shared_keys_equal_ema', 'backbone_state_keys', 'host_parameters',
                         'host_state_digest', 'complete_model_forwards', 'optimizer_updates', 'CUDA_initialized']}))
    except Exception:
        import traceback
        write(ROOT / 'strict_host_failure.json', {'utc': datetime.now(timezone.utc).isoformat(),
              'status': 'CPU_CONSTRUCTION_FAILED_BEFORE_ANY_FORWARD', 'traceback': traceback.format_exc(),
              'complete_model_forwards': 0, 'optimizer_updates': 0})
        raise
