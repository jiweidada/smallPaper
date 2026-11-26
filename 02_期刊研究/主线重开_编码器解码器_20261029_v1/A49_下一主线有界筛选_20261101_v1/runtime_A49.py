from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import torch
from ignore_query_supervision import disjoint_union

ROOT = Path(__file__).resolve().parent
ASSETS = Path('/root/uav_a47_20261101_v1/assets_A43_baseline')
A48 = Path('/root/uav_a48_20261101_v1')
sys.path.insert(0, '/root/uav_a47_20261101_v1/pydeps')
spec = importlib.util.spec_from_file_location('runtime_common', ASSETS / 'runtime_common.py')
native = importlib.util.module_from_spec(spec)
sys.modules['runtime_common'] = native
spec.loader.exec_module(native)
DATA = Path('/25085400120/DEIM/work/DEIM_1572862469/dataset_visdrone')
native.TRAIN_IMAGES = DATA / 'VisDrone2019-DET-train/images'
native.TRAIN_ANNS = DATA / 'VisDrone2019-DET-train/annotations'
# Reuse the already verified read-only jpg/txt symlink view; original data stays untouched.
native.VAL_IMAGES = A48 / 'val_inputs'
BaseDataset = native.VisDroneDataset


class IgnoreDataset(BaseDataset):
    def __init__(self, split='train'):
        super().__init__(split)
        for row in self.rows:
            row['ignore_disjoint_pixels'] = disjoint_union(row['ignore'])

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        row = self.rows[index]
        tiles = torch.tensor(row['ignore_disjoint_pixels'], dtype=torch.float32).reshape(-1, 4)
        flip = False
        if self.split == 'train':
            raw = hashlib.sha256((str(native.SEED) + ':' + str(self.epoch) + ':' + row['file']).encode()).digest()
            flip = bool(raw[0] & 1)
        if flip:
            tiles[:, [0, 2]] = row['width'] - tiles[:, [2, 0]]
        tiles = tiles / tiles.new_tensor([row['width'], row['height'], row['width'], row['height']])
        assert tiles.numel() == 0 or (tiles.min() >= 0 and tiles.max() <= 1)
        target['ignore_tiles'] = tiles
        return image, target


native.VisDroneDataset = IgnoreDataset


def plan_check(mode='formal'):
    assert mode in ['formal', 'capacity']
    filename = 'capacity_protocol_FROZEN_v2.json' if mode == 'capacity' else 'protocol_FROZEN.json'
    plan = json.loads((ROOT / filename).read_text())
    assert plan['arms'] == ['B', 'M', 'Q']
    if mode == 'formal':
        assert plan['epochs'] == 12
    else:
        assert plan['protocol_type'] == 'A49_CAPACITY_ONLY' and not plan['formal_training_authorized_by_this_file']
    for name, expected in plan['files_sha256'].items():
        assert native.digest(ROOT / name) == expected, name
    expect = json.loads((ROOT / 'asset_expectations.json').read_text())
    assert native.digest(ASSETS / 'common_init_10class.pth') == expect['common_init_sha256']
    assert native.digest(ASSETS / 'runtime_common.py') == expect['runtime_common_sha256']
    for name, expected in expect['native_source_sha256'].items():
        assert native.digest(native.SOURCE / name) == expected, name
    assert native.digest(A48 / 'training_utils_A48.py') == plan['external_training_utils_sha256']
    return plan
