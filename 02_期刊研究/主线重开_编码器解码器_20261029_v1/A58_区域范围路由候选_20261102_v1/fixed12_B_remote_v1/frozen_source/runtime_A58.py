"""Common native ECDet-S initialization and data for the bounded A58 pilot."""
from pathlib import Path
import importlib.util
import json
import sys
import torch

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/root/uav_a47_20261101_v1/assets_A43_baseline')
CONSTRUCTOR = Path('/root/uav_a58_20261102_v1/CPU_constructor_stage_v1')
INIT = CONSTRUCTOR / 'common_init_augmented_v1.pth'
sys.path.insert(0, '/root/uav_a47_20261101_v1/pydeps')
spec = importlib.util.spec_from_file_location('runtime_common', ASSETS / 'runtime_common.py')
native = importlib.util.module_from_spec(spec)
sys.modules['runtime_common'] = native
spec.loader.exec_module(native)
from regional_range_attention_v1 import install

DATA = Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
native.TRAIN_IMAGES = DATA / 'VisDrone2019-DET-train/images'
native.TRAIN_ANNS = DATA / 'VisDrone2019-DET-train/annotations'
native.VAL_IMAGES = Path('/root/uav_a48_20261101_v1/val_inputs')

def backend():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plan_check(kind):
    assert kind in ['capacity', 'formal']
    filename = 'capacity_protocol_FROZEN_v1.json' if kind == 'capacity' else 'protocol_FROZEN_v1.json'
    plan = json.loads((ROOT / filename).read_text())
    assert plan['phase'] == 'A58' and plan['arms'] == ['B', 'M', 'Q']
    for name, expected in plan['files_sha256'].items():
        assert native.digest(ROOT / name) == expected, name
    expected = json.loads((ROOT / 'asset_expectations.json').read_text())
    assert native.digest(ASSETS / 'runtime_common.py') == expected['runtime_common_sha256']
    assert native.digest(ASSETS / 'common_init_10class.pth') == expected['common_init_sha256']
    for name, digest in expected['native_source_sha256'].items():
        assert native.digest(native.SOURCE / name) == digest, name
    result = json.loads((CONSTRUCTOR / 'CPU_constructor_result_v1.json').read_text())
    assert result['status'] == 'PASS_CPU_CONSTRUCTOR_NO_FORWARD'
    assert result['counts']['constructors_completed'] == 1 and result['counts']['whole_detector_forwards'] == 0
    assert native.digest(INIT) == plan['augmented_initialization_sha256'] == result['common_init_augmented']['sha256']
    return plan

def components(arm):
    assert arm in ['B', 'M', 'Q']
    model, criterion, post = native.create_components('B')
    info = install(model, arm)
    init = torch.load(INIT, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(init, strict=True)
    state = model.state_dict()
    assert set(state) == set(init) and all(torch.equal(state[k], init[k]) for k in state)
    assert sum(p.numel() for p in model.parameters()) == 9951721
    assert info['extra_parameters'] == 148609
    return model, criterion, post

class ForwardJournal:
    """Journal actual root-model calls; route helpers are not extra model calls."""
    def __init__(self, model, path, limit):
        self.file = Path(path).open('x', buffering=1)
        self.limit = limit
        self.phase = 'not_assigned'
        self.started = self.completed = self.updates = 0
        model.register_forward_pre_hook(self.pre)
        model.register_forward_hook(self.post)

    def write(self, event):
        self.file.write(json.dumps(event, allow_nan=False) + '\n')

    def pre(self, module, args):
        assert self.phase in ['train', 'validation', 'latency', 'capacity']
        assert self.started < self.limit, 'Whole-detector forward cap reached'
        self.started += 1
        self.write(dict(event='forward_start', phase=self.phase, forward=self.started))

    def post(self, module, args, output):
        self.completed += 1
        self.write(dict(event='forward_complete', phase=self.phase, forward=self.completed))

    def optimizer_returned(self):
        self.updates += 1
        self.write(dict(event='optimizer_returned', updates=self.updates))

    def counts(self):
        return dict(full_model_forwards_started=self.started, full_model_forwards_completed=self.completed,
                    optimizer_updates=self.updates, native_counter=native.MODEL_COUNTER['forwards'])
