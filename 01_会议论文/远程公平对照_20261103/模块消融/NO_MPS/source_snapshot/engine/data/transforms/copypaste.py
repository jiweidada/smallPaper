"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class CopyPaste(T.Transform):
    """
    Implements Copy-Paste augmentation for object detection.
    Randomly copies objects from source images and pastes them onto the current image.
    """

    def __init__(self, p=0.5, max_objects=6, iou_thresh=0.3, area_thresh=0.01,
                 blend_ratio=0.5, use_cache=True, max_cached_objects=100) -> None:
        """
        Args:
            p (float): Probability of applying CopyPaste augmentation
            max_objects (int): Maximum number of objects to paste
            iou_thresh (float): IoU threshold for filtering overlapping objects
            area_thresh (float): Minimum area threshold (relative to image area) for objects
            blend_ratio (float): Blending ratio for pasted objects (0=transparent, 1=opaque)
            use_cache (bool): Whether to cache objects from previous images
            max_cached_objects (int): Maximum number of cached objects
        """
        super().__init__()
        self.p = p
        self.max_objects = max_objects
        self.iou_thresh = iou_thresh
        self.area_thresh = area_thresh
        self.blend_ratio = blend_ratio
        self.use_cache = use_cache
        self.max_cached_objects = max_cached_objects
        self.object_cache = []

    def forward(self, *inputs):
        """
        Apply CopyPaste augmentation

        Args:
            inputs: (image, target, dataset) tuple

        Returns:
            tuple: Augmented (image, target, dataset)
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip with probability 1 - p
        if random.random() > self.p:
            return image, target, dataset

        # Convert PIL image to tensor for processing if needed
        if isinstance(image, Image.Image):
            return self._copypaste_pil(image, target, dataset)
        else:
            return self._copypaste_tensor(image, target, dataset)

    def _copypaste_pil(self, image, target, dataset):
        """Handle CopyPaste for PIL images"""
        # Get image dimensions
        w, h = image.size
        image_area = w * h

        # Extract current image objects for potential caching
        if self.use_cache and 'boxes' in target and len(target['boxes']) > 0:
            self._cache_objects(image, target, image_area)

        # Get objects to paste
        paste_objects = self._get_paste_objects(dataset, image_area, num_objects=self.max_objects)

        if not paste_objects:
            return image, target, dataset

        # Create copy of image and target
        augmented_image = image.copy()
        augmented_target = self._clone_target(target)

        # Paste objects
        for obj in paste_objects:
            augmented_image, augmented_target = self._paste_object_pil(
                augmented_image, augmented_target, obj, w, h
            )

        return augmented_image, augmented_target, dataset

    def _copypaste_tensor(self, image, target, dataset):
        """Handle CopyPaste for tensor images"""
        # Get image dimensions
        c, h, w = image.shape
        image_area = w * h

        # For tensor implementation, convert to PIL, process, then convert back
        # This maintains compatibility with existing pipeline
        pil_image = F.to_pil_image(image)
        augmented_image, augmented_target, _ = self._copypaste_pil(pil_image, target, dataset)

        # Convert back to tensor
        augmented_image = F.pil_to_tensor(augmented_image).float() / 255.0

        return augmented_image, augmented_target, dataset

    def _cache_objects(self, image, target, image_area):
        """Cache objects from current image for future use"""
        boxes = target['boxes']
        labels = target['labels']

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            # Check if object meets area threshold
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area / image_area < self.area_thresh:
                continue

            # Crop object from image
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped_obj = image.crop((x1, y1, x2, y2))

            # Create mask for the object (simple rectangular mask)
            mask = Image.new('L', cropped_obj.size, 255)

            # Add to cache
            self.object_cache.append({
                'image': cropped_obj,
                'mask': mask,
                'label': label.item() if torch.is_tensor(label) else label,
                'relative_area': box_area / image_area
            })

        # Limit cache size
        if len(self.object_cache) > self.max_cached_objects:
            # Randomly remove excess objects
            num_to_remove = len(self.object_cache) - self.max_cached_objects
            indices_to_remove = random.sample(range(len(self.object_cache)), num_to_remove)
            self.object_cache = [obj for i, obj in enumerate(self.object_cache)
                                 if i not in indices_to_remove]

    def _get_paste_objects(self, dataset, image_area, num_objects):
        """Get objects to paste from cache or dataset"""
        paste_objects = []

        if self.use_cache and self.object_cache:
            # Select from cache
            num_available = min(len(self.object_cache), num_objects)
            if num_available > 0:
                indices = random.sample(range(len(self.object_cache)), num_available)
                paste_objects = [self.object_cache[i].copy() for i in indices]
        else:
            # Load from dataset
            for _ in range(min(3, num_objects)):  # Limit dataset loading
                try:
                    idx = random.randint(0, len(dataset) - 1)
                    sample_image, sample_target = dataset.load_item(idx)

                    if 'boxes' not in sample_target or len(sample_target['boxes']) == 0:
                        continue

                    # Extract objects from sample
                    self._extract_objects_from_sample(
                        sample_image, sample_target, image_area, paste_objects
                    )

                    if len(paste_objects) >= num_objects:
                        break

                except Exception:
                    continue

        return paste_objects[:num_objects]

    def _extract_objects_from_sample(self, sample_image, sample_target, target_area, paste_objects):
        """Extract objects from a sample image"""
        boxes = sample_target['boxes']
        labels = sample_target['labels']

        # Get sample image dimensions
        if isinstance(sample_image, Image.Image):
            sw, sh = sample_image.size
        else:
            _, sh, sw = sample_image.shape
            sample_image = F.to_pil_image(sample_image)

        sample_area = sw * sh

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            # Check area threshold
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            relative_area = box_area / sample_area

            if relative_area < self.area_thresh:
                continue

            # Crop object
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(sw, x2), min(sh, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cropped_obj = sample_image.crop((x1, y1, x2, y2))
            mask = Image.new('L', cropped_obj.size, 255)

            paste_objects.append({
                'image': cropped_obj,
                'mask': mask,
                'label': label.item() if torch.is_tensor(label) else label,
                'relative_area': relative_area
            })

    def _paste_object_pil(self, image, target, obj, img_w, img_h):
        """Paste a single object onto the image"""
        obj_w, obj_h = obj['image'].size

        # Random position for pasting
        max_x = img_w - obj_w
        max_y = img_h - obj_h

        if max_x <= 0 or max_y <= 0:
            # Object too large, resize it
            scale = min(img_w / obj_w, img_h / obj_h) * 0.8
            new_w = int(obj_w * scale)
            new_h = int(obj_h * scale)
            obj['image'] = obj['image'].resize((new_w, new_h), Image.BILINEAR)
            obj['mask'] = obj['mask'].resize((new_w, new_h), Image.NEAREST)
            obj_w, obj_h = new_w, new_h
            max_x = img_w - obj_w
            max_y = img_h - obj_h

        if max_x > 0 and max_y > 0:
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Create new box
            new_box = torch.tensor([paste_x, paste_y, paste_x + obj_w, paste_y + obj_h],
                                   dtype=torch.float32)

            # Check IoU with existing boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                ious = self._compute_iou(new_box.unsqueeze(0), target['boxes'])
                if ious.max() > self.iou_thresh:
                    return image, target  # Skip this object

            # Paste object with alpha blending
            if self.blend_ratio < 1.0:
                # Create blended version
                paste_region = image.crop((paste_x, paste_y, paste_x + obj_w, paste_y + obj_h))
                blended = Image.blend(paste_region, obj['image'], self.blend_ratio)
                image.paste(blended, (paste_x, paste_y))
            else:
                # Direct paste with mask
                image.paste(obj['image'], (paste_x, paste_y), obj['mask'])

            # Update target
            if 'boxes' in target:
                target['boxes'] = torch.cat([target['boxes'], new_box.unsqueeze(0)], dim=0)
            else:
                target['boxes'] = new_box.unsqueeze(0)

            if 'labels' in target:
                new_label = torch.tensor([obj['label']], dtype=target['labels'].dtype)
                target['labels'] = torch.cat([target['labels'], new_label], dim=0)
            else:
                target['labels'] = torch.tensor([obj['label']])

            # Update other fields
            if 'area' in target:
                new_area = torch.tensor([obj_w * obj_h], dtype=target['area'].dtype)
                target['area'] = torch.cat([target['area'], new_area], dim=0)

            if 'iscrowd' in target:
                new_iscrowd = torch.tensor([0], dtype=target['iscrowd'].dtype)
                target['iscrowd'] = torch.cat([target['iscrowd'], new_iscrowd], dim=0)

        return image, target

    def _compute_iou(self, boxes1, boxes2):
        """Compute IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        # Union
        union = area1[:, None] + area2[None, :] - inter

        iou = inter / union
        return iou

    def _clone_target(self, target):
        """Clone target dictionary"""
        cloned = {}
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = value.copy()
            else:
                cloned[key] = value
        return cloned