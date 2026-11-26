"""TinyPerson v1 official AP evaluator adapter; model/solver sources stay frozen."""
import copy
import importlib.util
import io
import json
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np

IOUS = [0.25, 0.5, 0.75]
AREAS = ['all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable']
RAW_NAMES = [f'{kind}{int(round(iou * 100))}_{area}'
             for kind in ['AP', 'AR'] for iou in IOUS for area in AREAS]
PRIMARY_INDEX = RAW_NAMES.index('AP50_tiny')
ORDER = [PRIMARY_INDEX] + [i for i in range(len(RAW_NAMES)) if i != PRIMARY_INDEX]
METRIC_NAMES = [RAW_NAMES[i] for i in ORDER]


class NumpyCompatibility:
    """Process-local shim for legacy NumPy spellings and integer linspace counts."""
    def __getattr__(self, name):
        if name in ['float', 'int', 'bool']: return {'float': float, 'int': int, 'bool': bool}[name]
        return getattr(np, name)

    @staticmethod
    def linspace(start, stop, num=50, *args, **kwargs):
        if int(num) != num: raise ValueError('Upstream linspace count must be integer-valued')
        return np.linspace(start, stop, int(num), *args, **kwargs)


def official_module(path=None):
    path = Path(path or Path(__file__).with_name('official_tiny_cocoeval.py'))
    spec = importlib.util.spec_from_file_location('pinned_official_tiny_cocoeval', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.np = NumpyCompatibility()
    module.Params.EVAL_STRANDARD = 'tiny'
    return module


def evaluate_records(annotation, predictions, module_path=None, verbose=True):
    from pycocotools.coco import COCO
    module = official_module(module_path)
    with redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = copy.deepcopy(annotation); gt.createIndex()
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO(); dt.dataset = copy.deepcopy(gt.dataset); dt.dataset['annotations'] = []; dt.createIndex()
    evaluator = module.COCOeval(gt, dt, 'bbox', ignore_uncertain=True,
                               use_iod_for_ignore=True, use_ignore_attr=True)
    assert evaluator.params.maxDets == [200]
    assert np.array_equal(evaluator.params.iouThrs, IOUS)
    stream = io.StringIO()
    with redirect_stdout(stream):
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    raw = np.asarray(evaluator.stats, dtype=float)
    assert len(raw) == len(RAW_NAMES) and np.isfinite(raw).all()
    if verbose: print(stream.getvalue(), end='', flush=True)
    return {name: float(raw[i]) for i, name in enumerate(RAW_NAMES)}, raw[ORDER].tolist(), evaluator


class TinyPersonEvaluator:
    """Non-COCO native evaluator interface, preserving official TinyPerson matching."""
    def __init__(self, ann_file, official_module_path=None):
        self.ann_file = str(ann_file); self.annotation = json.loads(Path(ann_file).read_text())
        self.official_module_path = official_module_path
        self.iou_types = ['bbox']; self.cleanup()

    def cleanup(self):
        self.predictions = []; self.seen = set(); self.stats = {}; self.named_metrics = None

    def update(self, predictions, targets=None):
        for image_id, prediction in predictions.items():
            image_id = int(image_id)
            if image_id in self.seen: raise RuntimeError('Duplicate evaluation image: ' + str(image_id))
            self.seen.add(image_id)
            boxes = prediction['boxes'].detach().cpu().tolist()
            scores = prediction['scores'].detach().cpu().tolist()
            labels = prediction['labels'].detach().cpu().tolist()
            for box, score, label in zip(boxes, scores, labels):
                if int(label) != 0: raise RuntimeError('TinyPerson must use the single contiguous person class')
                self.predictions.append({'image_id': image_id, 'category_id': 0,
                    'bbox': [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                    'score': float(score)})

    def synchronize_between_processes(self):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
            raise RuntimeError('This controlled queue uses one GPU process per training; distributed evaluation is not configured')

    def accumulate(self):
        expected = {int(image['id']) for image in self.annotation['images']}
        if self.seen != expected: raise RuntimeError('Evaluation image coverage mismatch')
        named, values, _ = evaluate_records(self.annotation, self.predictions, self.official_module_path)
        self.named_metrics = named; self.stats = {'tiny_eval_bbox': values}

    def summarize(self):
        print(json.dumps({'metric': 'official TinyPerson AP50_tiny',
                          'value': self.named_metrics['AP50_tiny'],
                          'max_detections': 200, 'ignore_uncertain': True, 'use_iod_for_ignore': True}), flush=True)


def install_native_adapter():
    """Patch only this new process's extension points, never the frozen files."""
    from engine.core import YAMLConfig
    from engine.solver.det_solver import DetSolver
    original = YAMLConfig.evaluator.fget
    old_names = DetSolver._coco_metric_names

    def evaluator(self):
        settings = self.yaml_cfg.get('evaluator', {})
        if settings.get('type') != 'TinyPersonEvaluator': return original(self)
        if self._evaluator is None:
            self._evaluator = TinyPersonEvaluator(settings['ann_file'], settings.get('official_module_path'))
        return self._evaluator

    def metric_names(self):
        return METRIC_NAMES if isinstance(getattr(self, 'evaluator', None), TinyPersonEvaluator) else old_names(self)

    YAMLConfig.evaluator = property(evaluator)
    DetSolver._coco_metric_names = metric_names

