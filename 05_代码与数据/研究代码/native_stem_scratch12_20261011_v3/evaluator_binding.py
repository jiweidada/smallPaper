"""Bind actual evaluator sources AND its compiled backend; no import during prepare."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MODULE_FILES = {'pycocoeval': '__init__.py',
                **{'pycocoeval.' + name: name + '.py' for name in
                   ('coco', 'cocoeval', 'mask', 'yoloeval', 'rotated_cocoeval')}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(binding):
    expected = json.loads((HERE / 'evaluator_sources.json').read_text(encoding='utf-8'))
    if binding.get('schema') != 'custom_pycocoeval_complete_package_v1':
        raise ValueError('Missing complete evaluator package binding')
    files = binding.get('files', {})
    if {n: files.get(n) for n in expected} != expected:
        raise ValueError('Evaluator Python sources differ from the verified original')
    extra = set(files) - set(expected)
    if len(extra) != 1:
        raise ValueError('Expected exactly one compiled _mask backend')
    backend = next(iter(extra))
    if (Path(backend).name != backend or not backend.startswith('_mask.')
            or not backend.endswith(('.pyd', '.so')) or backend != binding.get('backend')):
        raise ValueError('Unexpected evaluator backend filename')
    if any(not isinstance(h, str) or len(h) != 64 or
           any(c not in '0123456789abcdef' for c in h) for h in files.values()):
        raise ValueError('Invalid evaluator file digest')
    return binding


def fingerprint(package_root):
    root = Path(package_root).resolve()
    if root.name != 'pycocoeval' or not root.is_dir():
        raise ValueError('Pass the actual pycocoeval package directory')
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts
             and p.suffix in ('.py', '.pyd', '.so', '.dll')}
    backends = [n for n in files if n.startswith('_mask.') and n.endswith(('.pyd', '.so'))]
    if len(backends) != 1:
        raise ValueError('Missing or ambiguous compiled _mask backend')
    return validate({'schema': 'custom_pycocoeval_complete_package_v1',
                     'files': files, 'backend': backends[0]})


def verify_files(package_root, binding):
    validate(binding)
    if fingerprint(package_root) != binding:
        raise RuntimeError('Evaluator package changed after preparation')


def loaded_fingerprint(binding, package_root=None):
    """Inspect modules already used by native code. Never evaluate or import here."""
    validate(binding)
    package = sys.modules.get('pycocoeval')
    if not getattr(package, '__file__', None):
        raise RuntimeError('Bound evaluator package has not been loaded')
    root = Path(package.__file__).resolve().parent
    if package_root is not None and root != Path(package_root).resolve():
        raise RuntimeError('A different evaluator package location was loaded')
    verify_files(root, binding)
    module_files = dict(MODULE_FILES, **{'pycocoeval._mask': binding['backend']})
    for name, filename in module_files.items():
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        if not path or Path(path).resolve() != (root / filename).resolve():
            raise RuntimeError('Evaluator module loaded from unexpected location: ' + name)
        if sha(path) != binding['files'][filename]:
            raise RuntimeError('Loaded evaluator module differs: ' + name)
    mask = sys.modules['pycocoeval.mask']
    backend = sys.modules['pycocoeval._mask']
    cocoeval = sys.modules['pycocoeval.cocoeval']
    if cocoeval.maskUtils is not mask or mask.iou is not backend.iou:
        raise RuntimeError('Native bbox IoU does not use the bound compiled backend')
    return {'binding': binding, 'package_root': str(root),
            'loaded_modules': module_files,
            'historical_numerical_equivalence_verified': False}


def activate(package_root, binding):
    verify_files(package_root, binding)
    if any(n == 'pycocoeval' or n.startswith('pycocoeval.') for n in sys.modules):
        raise RuntimeError('Evaluator activation requires a fresh interpreter')
    sys.path.insert(0, str(Path(package_root).resolve().parent))
    importlib.invalidate_caches()
    for name in (*MODULE_FILES, 'pycocoeval._mask'):
        importlib.import_module(name)
    return loaded_fingerprint(binding, package_root)


def validate_result_source(source, identity):
    binding = validate(identity.get('evaluator_binding', {}))
    if (source.get('binding') != binding or source.get('module') != 'pycocoeval.cocoeval'
            or source.get('class') != 'COCOeval'
            or source.get('sha256') != binding['files']['cocoeval.py']):
        raise ValueError('Endpoint evaluator package differs from its prepared identity')
    return binding


def same_result_evaluator(left, right):
    a = validate_result_source(left['evaluator_source'], left['identity'])
    b = validate_result_source(right['evaluator_source'], right['identity'])
    if a != b:
        raise ValueError('Evaluator source or compiled backend differs between runs')
