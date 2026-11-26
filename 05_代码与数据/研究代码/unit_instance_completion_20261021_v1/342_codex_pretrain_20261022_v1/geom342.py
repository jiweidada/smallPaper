"""342 shared CPU geometry. All outputs retain unclipped xyxy edges.

The enclosing 342 directory is a new version; the pre-existing geom342.py is
preserved. geometry_cost returns [B,256,J], including B=1 for [J,4] input.
"""
import importlib
_ctx = importlib.import_module('342_context')
import torch
core311 = importlib.import_module('core')

def _boxes(boxes):
    b = torch.as_tensor(boxes)
    if not b.is_floating_point():
        b = b.float()
    assert b.device.type == 'cpu' and b.ndim in (2, 3) and b.shape[-1] == 4
    assert bool(torch.isfinite(b).all()), 'non-finite box'
    assert bool((b[..., 2:] > b[..., :2]).all()), 'expected positive xyxy edges'
    return b

def assert_window(window):
    w = torch.as_tensor(window, dtype=torch.float32)
    assert w.device.type == 'cpu' and w.shape == (4,) and bool(torch.isfinite(w).all())
    assert bool((w[2:] > w[:2]).all()), 'invalid xyxy window'
    assert bool(((w >= 0) & (w <= 640)).all()), 'window outside 640 plane'
    return w

def win_norm_xyxy(boxes_640_xyxy, window_xyxy_640):
    b = _boxes(boxes_640_xyxy)
    w = assert_window(window_xyxy_640).to(b.dtype)
    origin = w[:2].repeat(2)
    extent = (w[2:] - w[:2]).repeat(2)
    out = (b - origin) / extent
    _boxes(out)
    return out

def xyxy_to_cxcywh(boxes):
    b = _boxes(boxes)
    return torch.cat(((b[..., :2] + b[..., 2:]) * .5, b[..., 2:] - b[..., :2]), -1)

def win_norm_cxcywh(boxes_640_xyxy, window_xyxy_640):
    return xyxy_to_cxcywh(win_norm_xyxy(boxes_640_xyxy, window_xyxy_640))

def win_norm_to_window_pixels(boxes_win_norm_xyxy, window_xyxy_640, output_size=None):
    b = _boxes(boxes_win_norm_xyxy)
    w = assert_window(window_xyxy_640).to(b.dtype)
    extent = w[2:] - w[:2] if output_size is None else b.new_tensor(output_size)
    assert extent.shape == (2,) and bool((extent > 0).all())
    return b * extent.repeat(2)

def grid_xy(dtype=torch.float32):
    return core311.grid_positions(dtype=dtype)

def geometry_cost(boxes_win_norm_xyxy, grid_xy=None):
    b = _boxes(boxes_win_norm_xyxy)
    # A project-specific sanity guard, not clipping or candidate filtering.
    # Native sigmoid boxes in a 64px window are within +/-15 normalized units.
    # Out-of-window boxes remain legal. Extremely small custom windows may need
    # a separately reviewed bound; they are not silently accepted here.
    assert bool((b.abs() < 32).all()), 'pixel-space input or unreviewed coordinate range'
    if b.ndim == 2:
        b = b.unsqueeze(0)
    xy = core311.grid_positions(dtype=b.dtype) if grid_xy is None else grid_xy.to(b.dtype)
    assert xy.device.type == 'cpu' and xy.shape == (256, 2)
    assert bool(torch.isfinite(xy).all()) and bool(((xy > 0) & (xy < 1)).all())
    assert torch.equal(xy, core311.grid_positions(dtype=b.dtype)), 'grid must be frozen 311_v2 grid'
    lower, upper = b[..., :2], b[..., 2:]
    outside = (lower[:, None] - xy[None, :, None]).clamp_min(0)
    outside = outside + (xy[None, :, None] - upper[:, None]).clamp_min(0)
    result = (outside / (upper - lower)[:, None]).sum(-1)
    assert bool(torch.isfinite(result).all())
    return result

def intersecting_candidates(fullp_boxes_640, window_xyxy_640):
    b = _boxes(fullp_boxes_640)
    assert b.ndim == 2
    w = assert_window(window_xyxy_640).to(b.dtype)
    lower = torch.maximum(b[:, :2], w[:2])
    upper = torch.minimum(b[:, 2:], w[2:])
    return (upper > lower).all(-1)
