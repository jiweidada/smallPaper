"""342 stage-1: the single shared geometry entry for the whole project.

ALL box-coordinate conversion and d_out geometry must go through this module (R3a).
Coordinate spaces used here:
  * 640 plane      : xyxy in [0,640] pixels (window coordinates live here too)
  * window-normal  : xyxy with origin at the window top-left, divided by the window
                     width/height -> roughly [0,1] (values may exceed 1 for boxes
                     outside the window; never clipped)
  * grid           : 311_v2 grid_positions() -> 16x16 cell centres in [0,1]
"""
import os, sys
import torch
sys.dont_write_bytecode = True
BASE = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2"
sys.path.insert(0, BASE + chr(92) + '311_unit_instance_core_v2')
import core as core311

__all__ = ['win_norm_xyxy', 'win_norm_cxcywh', 'win_norm_xyxy_to_cxcywh',
           'geometry_cost', 'intersecting_candidates', 'grid_xy',
           'assert_window', 'bbox_area']

def assert_window(window_xyxy_640):
    """Window must be a valid 640-plane xyxy box."""
    w = torch.as_tensor(window_xyxy_640, dtype=torch.float32).reshape(-1)
    assert w.numel() == 4, 'window must have 4 numbers'
    assert torch.isfinite(w).all(), 'window has non-finite values'
    assert w[2] > w[0] and w[3] > w[1], 'window must have positive width/height'
    return w

def win_norm_xyxy(boxes_640_xyxy, window_xyxy_640):
    """640-plane xyxy -> window-normalised xyxy (no clipping).
    x_n = (x - W0)/(W2-W0); y_n = (y - W1)/(W3-W1)."""
    b = torch.as_tensor(boxes_640_xyxy, dtype=torch.float32)
    assert b.dim() == 2 and b.shape[1] == 4, 'boxes must be [N,4]'
    assert torch.isfinite(b).all(), 'boxes have non-finite values'
    W = assert_window(window_xyxy_640)
    ww = W[2] - W[0]; wh = W[3] - W[1]
    out = torch.stack([(b[:, 0] - W[0]) / ww, (b[:, 1] - W[1]) / wh,
                       (b[:, 2] - W[0]) / ww, (b[:, 3] - W[1]) / wh], dim=-1)
    return out

def win_norm_xyxy_to_cxcywh(boxes_win_norm_xyxy):
    """window-normalised xyxy -> window-normalised cxcywh (derived, not a second implementation)."""
    b = torch.as_tensor(boxes_win_norm_xyxy, dtype=torch.float32)
    assert b.dim() == 2 and b.shape[1] == 4
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)

def win_norm_cxcywh(boxes_640_xyxy, window_xyxy_640):
    """640-plane xyxy -> window-normalised cxcywh (for targets). One conversion path only."""
    return win_norm_xyxy_to_cxcywh(win_norm_xyxy(boxes_640_xyxy, window_xyxy_640))

def grid_xy():
    """16x16 cell centres in [0,1] from the frozen 311_v2 implementation."""
    return core311.grid_positions()

def bbox_area(b):
    return (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)

def geometry_cost(boxes_win_norm_xyxy, grid_xy_tensor=None):
    """d_out(i,j): sum over axes of the non-negative distance from the cell centre to the
    box exterior, divided by the box width/height. Input MUST be window-normalised xyxy."""
    b = torch.as_tensor(boxes_win_norm_xyxy, dtype=torch.float32)
    assert b.dim() == 2 and b.shape[1] == 4, 'geometry_cost expects [N,4]'
    assert torch.isfinite(b).all(), 'boxes have non-finite values'
    wh = b[:, 2:] - b[:, :2]
    assert (wh > 0).all(), 'geometry_cost requires positive width/height (xyxy, not cxcywh)'
    # guard against the classic px-vs-norm mistake: a valid window-normal xyxy box
    # may legitimately exceed 1 (box outside window) but cannot be enormous.
    assert (b.abs() < 50).all(), 'boxes look like pixel coordinates, not window-normalised'
    if grid_xy_tensor is None:
        grid_xy_tensor = grid_xy()
    xy = grid_xy_tensor.to(b.dtype)[None, :, None, :]        # [1,L,1,2]
    bb = b[:, None, :]                                      # [N,1,4]
    outside = (bb[..., :2] - xy).clamp_min(0.) + (xy - bb[..., 2:]).clamp_min(0.)
    geom = (outside / wh[:, None]).sum(dim=-1)              # [N,L]
    assert torch.isfinite(geom).all(), 'geometry produced non-finite values'
    return geom

def intersecting_candidates(fullp_boxes_640, window_xyxy_640):
    """317 P_W: candidates whose box has positive-area intersection with the window."""
    b = torch.as_tensor(fullp_boxes_640, dtype=torch.float32)
    W = assert_window(window_xyxy_640)
    ix = torch.clamp(torch.minimum(b[:, 2], W[2]) - torch.maximum(b[:, 0], W[0]), min=0)
    iy = torch.clamp(torch.minimum(b[:, 3], W[3]) - torch.maximum(b[:, 1], W[1]), min=0)
    return (ix * iy) > 0