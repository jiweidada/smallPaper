"""Training-only box-weighted BN moments; weighted normalization is prior work.

M uses summed box-pixel mass, Q gives each valid box unit foreground mass.
Both mix this with uniform mass. At evaluation these are ordinary BatchNorm2d.
Box footprints include background and overlap multiplicity, not object masks.
"""
import contextlib
import torch
from torch import nn

SITES = (
    'encoder.fpn_blocks.0.cv4.norm',
    'encoder.fpn_blocks.1.cv4.norm',
    'encoder.pan_blocks.0.cv4.norm',
    'encoder.pan_blocks.1.cv4.norm',
)


class MomentContext:
    def __init__(self, mode, steps_per_epoch=809):
        assert mode in ('M', 'Q')
        self.mode = mode
        self.steps_per_epoch = steps_per_epoch
        self.targets = None
        self.mixture = None
        self.cache = {}

    def set(self, targets, global_step):
        assert self.targets is None
        self.targets = targets
        self.mixture = 0.5 * min(1.0, global_step / (2 * self.steps_per_epoch))
        self.cache.clear()

    def clear(self):
        self.targets = None
        self.mixture = None
        self.cache.clear()

    @torch.no_grad()
    def weights(self, x):
        assert self.targets is not None, 'Training context must be explicitly set'
        batch, _, height, width = x.shape
        assert len(self.targets) == batch
        key = (height, width, x.device)
        if key in self.cache:
            return self.cache[key]
        base = torch.full((batch, 1, height, width), 1.0/(batch*height*width), device=x.device)
        if self.mixture == 0:
            self.cache[key] = base
            return base
        masks = []
        count = 0
        xs = torch.arange(width, device=x.device, dtype=torch.float32)
        ys = torch.arange(height, device=x.device, dtype=torch.float32)
        for target in self.targets:
            boxes = target['boxes'].detach().float()
            if len(boxes) == 0:
                masks.append(torch.zeros((height, width), device=x.device))
                continue
            edges = torch.cat((boxes[:, :2]-boxes[:, 2:]/2,
                               boxes[:, :2]+boxes[:, 2:]/2), -1).clamp(0, 1)
            dx = (torch.minimum(xs[None]+1, edges[:, 2:3]*width)
                  - torch.maximum(xs[None], edges[:, 0:1]*width)).clamp(min=0)
            dy = (torch.minimum(ys[None]+1, edges[:, 3:4]*height)
                  - torch.maximum(ys[None], edges[:, 1:2]*height)).clamp(min=0)
            area = dx.sum(-1)*dy.sum(-1)
            assert (area > 0).all(), 'Nonpositive projected GT box'
            if self.mode == 'Q':
                dx = dx / area[:, None]
            # Explicit products avoid TF32 approximation in label-derived mass.
            masks.append((dy[:, :, None]*dx[:, None, :]).sum(0))
            count += len(boxes)
        if count:
            foreground = torch.stack(masks)[:, None]
            foreground = foreground / foreground.sum()
            result = (1-self.mixture)*base + self.mixture*foreground
        else:
            result = base
        self.cache[key] = result
        return result


class BoxMomentBN(nn.BatchNorm2d):
    def __init__(self, original, context):
        super().__init__(original.num_features, eps=original.eps, momentum=original.momentum,
                         affine=original.affine, track_running_stats=original.track_running_stats)
        assert self.affine and self.track_running_stats
        self.to(device=original.weight.device, dtype=original.weight.dtype)
        self.load_state_dict(original.state_dict(), strict=True)
        self.context = context
        self.weighted_training_calls = 0
        self.standard_eval_calls = 0

    def forward(self, x):
        if not self.training:
            assert self.context.targets is None, 'Evaluation must not carry training labels'
            self.standard_eval_calls += 1
            return super().forward(x)
        assert self.context.targets is not None
        if self.context.mixture == 0:
            return super().forward(x)
        self.weighted_training_calls += 1
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            weights = self.context.weights(value)
            mean = (weights*value).sum((0, 2, 3))
            residual = value-mean[None, :, None, None]
            variance = (weights*residual.square()).sum((0, 2, 3))
            result = residual * torch.rsqrt(variance[None, :, None, None]+self.eps)
            result = result*self.weight[None, :, None, None]+self.bias[None, :, None, None]
            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                momentum = self.momentum if self.momentum is not None else 1.0/float(self.num_batches_tracked)
                correction = 1.0/(1.0-weights.square().sum())
                self.running_mean.lerp_(mean.detach(), momentum)
                self.running_var.lerp_(variance.detach()*correction, momentum)
        return result.to(x.dtype)


def install(model, mode):
    assert mode in ('B', 'M', 'Q')
    if mode == 'B':
        return None
    context = MomentContext(mode)
    state_names = tuple(model.state_dict())
    parameters = sum(p.numel() for p in model.parameters())
    for name in SITES:
        original = model.get_submodule(name)
        assert type(original) is nn.BatchNorm2d, (name, type(original))
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, BoxMomentBN(original, context))
    assert tuple(model.state_dict()) == state_names
    assert sum(p.numel() for p in model.parameters()) == parameters
    return context


def report_calls(model):
    return {name:{'weighted_training_calls':layer.weighted_training_calls,
                  'standard_eval_calls':layer.standard_eval_calls}
            for name,layer in model.named_modules() if isinstance(layer,BoxMomentBN)}


@contextlib.contextmanager
def ordinary_inference(model):
    """Use the original BN class and graph for all inference and timing."""
    saved = []
    try:
        for name in SITES:
            original = model.get_submodule(name)
            if not isinstance(original, BoxMomentBN):
                continue
            assert original.context.targets is None
            plain = nn.BatchNorm2d(original.num_features, eps=original.eps,
                                   momentum=original.momentum, affine=True, track_running_stats=True)
            plain.to(device=original.weight.device, dtype=original.weight.dtype)
            plain.load_state_dict(original.state_dict(), strict=True)
            plain.eval()
            parent, attr = name.rsplit('.', 1)
            saved.append((model.get_submodule(parent), attr, original))
            setattr(model.get_submodule(parent), attr, plain)
        assert not any(isinstance(m, BoxMomentBN) for m in model.modules())
        yield
    finally:
        for parent, attr, original in saved:
            setattr(parent, attr, original)
