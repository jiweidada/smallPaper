"""Training-only, full-source substitution at an existing Mosaic boundary.

The native Mosaic still executes: donor/affine random draws are consumed before
its result is conditionally replaced. This preserves data-side RNG draws,
not an entire detector training trajectory. No inference or model edits.
"""
import hashlib
import torch

ARMS = ('identity', 'target_protect', 'random_protect')
CONTRACT = dict(schema='scale_protection12_v1', reference_input=640,
    risk_shortside_threshold=4., minimum_risky_fraction=.25,
    substitution='complete original source; no additional crop or bbox filtering',
    downstream='native Mosaic branch downstream transforms and collate unchanged',
    random_control='same number of protected image identities per epoch; SHA-ranked independent image set',
    random_control_realized_event_counts='not forced equal; explicitly record',
    global_rng='native Mosaic is executed then its output is substituted; no new random draws',
    model_changes=False, inference_changes=False, validation_labels_used=False)


def training_profile(dataset):
    """Native raw-box clipping, from the train COCO object only."""
    by_image = {int(i): [] for i in dataset.ids}
    for a in dataset.coco.anns.values():
        iid = int(a['image_id'])
        if iid not in by_image or a.get('iscrowd', 0):
            continue
        im = dataset.coco.imgs[iid]; w, h = float(im['width']), float(im['height'])
        x, y, bw, bh = map(float, a['bbox'])
        # Match raw target FP32 xywh->xyxy and native clamping.
        b = torch.tensor([x, y, bw, bh], dtype=torch.float32)
        b[2:] += b[:2]; b[0::2].clamp_(0, w); b[1::2].clamp_(0, h)
        size = b[2:] - b[:2]
        if bool((size > 0).all()):
            side = float((size * size.new_tensor([640/w, 640/h])).min())
            by_image[iid].append(side)
    result = {}
    for iid, sides in by_image.items():
        fraction = sum(s < 4. for s in sides) / len(sides) if sides else 0.
        result[iid] = dict(raw_valid_GT=len(sides), risky_fraction=fraction,
                           protect=fraction >= .25)
    assert 0 < sum(r['protect'] for r in result.values()) < len(result)
    return result


class ScaleGuard:
    def __init__(self, dataset, seed, arm, profile, observer=None):
        if arm not in ARMS:
            raise ValueError(arm)
        self.dataset, self.seed, self.arm, self.profile = dataset, int(seed), arm, profile
        self.observer, self.pending, self.handles = observer, None, []
        self.occurrence, self.random_epoch, self.random_members = 0, None, set()
        self.protected_images = sum(p['protect'] for p in profile.values())

    def reset(self):
        assert self.pending is None
        self.occurrence = 0

    def _random_set(self, epoch):
        if self.random_epoch != epoch:
            order = sorted(self.profile, key=lambda iid: hashlib.sha256(
                f'scale_protection_control:{self.seed}:{epoch}:{iid}'.encode()).digest())
            self.random_members = set(order[:self.protected_images])
            self.random_epoch = epoch
        return self.random_members

    def before(self, module, inputs):
        assert self.pending is None
        sample = inputs[0] if len(inputs) == 1 else inputs
        assert len(sample) == 3 and sample[2] is self.dataset
        image, target, dataset = sample
        iid = int(target['image_id'].reshape(-1)[0]); epoch = int(dataset.epoch)
        assert iid in self.profile and image.mode == 'RGB'
        risk = self.profile[iid]
        assert len(target['boxes']) == risk['raw_valid_GT']
        apply = self.arm == 'target_protect' and risk['protect']
        if self.arm == 'random_protect':
            apply = iid in self._random_set(epoch)
        raw = None
        if apply:
            assert not any(k in target for k in ('masks', 'keypoints'))
            copied = {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}
            copied['mosaic_source_image_ids'] = torch.tensor([iid], dtype=torch.int64)
            raw = (image.copy(), copied, dataset)
        self.pending = dict(image_id=iid, epoch=epoch, occurrence=self.occurrence,
                            risky_fraction=risk['risky_fraction'], applied=bool(apply),
                            source_GT=len(target['boxes']), original=raw)
        self.occurrence += 1

    def after(self, module, inputs, output):
        assert self.pending is not None and len(output) == 3 and output[2] is self.dataset
        row = self.pending; self.pending = None
        original = row.pop('original')
        row['native_mosaic_GT'] = len(output[1]['boxes'])
        row['used_GT'] = len(original[1]['boxes']) if original is not None else row['native_mosaic_GT']
        row['arm'] = self.arm
        if self.observer:
            self.observer(row)
        return original

    def close(self):
        assert self.pending is None
        for h in self.handles:
            h.remove()
        self.handles.clear()


def install(dataset, seed, arm, observer=None, profile=None):
    ops = [t for t in dataset._transforms.transforms if type(t).__name__ == 'Mosaic']
    assert len(ops) == 1 and not ops[0].use_cache and ops[0].probability == 1.
    assert dataset._transforms.policy['epoch'] == [4, 64, 120]
    assert dataset._transforms.mosaic_prob == .5
    profile = training_profile(dataset) if profile is None else profile
    guard = ScaleGuard(dataset, seed, arm, profile, observer)
    guard.handles = [ops[0].register_forward_pre_hook(guard.before),
                     ops[0].register_forward_hook(guard.after)]
    return guard
