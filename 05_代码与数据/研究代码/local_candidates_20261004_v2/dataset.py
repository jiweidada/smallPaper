"""GT-independent native crops; paired geometry is deterministic by image/epoch."""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import bootstrap
from geometry import windows, make_views, targets_for_global, targets_for_window
from factory import native_recipe


class OriginalCocoImages(Dataset):
    def __init__(self, image_root, annotations):
        self.image_root = Path(image_root)
        self.annotation_path = Path(annotations)
        self.annotation = json.loads(self.annotation_path.read_text(encoding='utf8'))
        cats = sorted(x['id'] for x in self.annotation['categories'])
        if cats != list(range(10)):
            raise ValueError('Expected the frozen 0-based VisDrone ten-class annotation')
        self.images = sorted(self.annotation['images'], key=lambda x: x['id'])
        self.anns = {im['id']: [] for im in self.images}
        for ann in self.annotation['annotations']:
            if not ann.get('iscrowd', 0) and ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                self.anns[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        image = Image.open(self.image_root / info['file_name']).convert('RGB')
        if image.size != (info['width'], info['height']):
            raise ValueError('Image dimensions differ from annotation')
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        anns = self.anns[info['id']]
        xywh = torch.tensor([a['bbox'] for a in anns], dtype=torch.float32).reshape(-1, 4)
        boxes = xywh.clone()
        boxes[:, 2:] += boxes[:, :2]
        boxes = boxes.clamp(min=0).minimum(torch.tensor([image.width, image.height] * 2))
        keep = (boxes[:, 2:] > boxes[:, :2]).all(-1)
        target = {'image_id': torch.tensor([info['id']]), 'boxes': boxes[keep],
                  'labels': torch.tensor([a['category_id'] for a in anns], dtype=torch.long)[keep],
                  'area': torch.tensor([a.get('area', a['bbox'][2] * a['bbox'][3]) for a in anns])[keep],
                  'iscrowd': torch.zeros(int(keep.sum()), dtype=torch.long),
                  'annotation_ids': torch.tensor([a['id'] for a in anns], dtype=torch.long)[keep],
                  'orig_size': torch.tensor([image.height, image.width])}
        return tensor, target


class PairedCocoImages(OriginalCocoImages):
    def __init__(self, image_root, annotations, seed=1, stride=512, augment=True):
        super().__init__(image_root, annotations)
        self.seed, self.stride, self.augment, self.epoch = seed, stride, augment, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        h, w = image.shape[-2:]
        image_id = int(target['image_id'].item())
        material = f'{self.seed}:{self.epoch}:{image_id}'.encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], 'little'))
        # Geometry/photometric transform is applied once before creating both views.
        # Native Mosaic/ZoomOut/MixUp cannot preserve this simple one-image mapping.
        if self.augment and rng.random() < .5:
            image = image.flip(-1)
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target = {**target, 'boxes': boxes}
        if self.augment and self.epoch < 84:
            image = (image * rng.uniform(.8, 1.2)).clamp(0, 1)
        choices = windows(w, h, stride=self.stride)
        window = choices[rng.randrange(len(choices))]
        full, local, crop = make_views(image, window)
        return {'global': full, 'local': local, 'crop_norm': crop}, {
            'global': targets_for_global(target, (h, w)),
            'local': targets_for_window(target, window, (h, w)),
            'window': torch.tensor(window), 'image_id': target['image_id'],
        }


def paired_collate(items):
    samples = {k: torch.stack([x[0][k] for x in items]) for k in ('global', 'local', 'crop_norm')}
    targets = {k: [x[1][k] for x in items] for k in ('global', 'local')}
    targets['windows'] = torch.stack([x[1]['window'] for x in items])
    return samples, targets


def native_query_loader(image_root, annotations, batch_size=8, workers=4):
    # Use byte-identical native augmentations/collate for the query-only pair.
    from engine.data.transforms._transforms import (EmptyTransform, RandomPhotometricDistort,
        RandomZoomOut, RandomIoUCrop, RandomHorizontalFlip, Resize, SanitizeBoundingBoxes,
        ConvertPILImage, ConvertBoxes)
    from engine.data.transforms.mosaic import Mosaic
    from engine.data.transforms.container import Compose
    from engine.data.dataset.coco_dataset import CocoDetection
    from engine.data.dataloader import BatchImageCollateFunction
    kwargs = copy.deepcopy(native_recipe()['train_dataloader'])
    ds = kwargs['dataset']
    transform = ds['transforms']; transform.pop('type')
    dataset = CocoDetection(str(image_root), str(annotations), Compose(**transform),
                            num_classes=10, cache_imgsz=None, ram_cache=False)
    collate = kwargs['collate_fn']; collate.pop('type')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=workers, collate_fn=BatchImageCollateFunction(**collate),
                      persistent_workers=False, pin_memory=True)


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value

