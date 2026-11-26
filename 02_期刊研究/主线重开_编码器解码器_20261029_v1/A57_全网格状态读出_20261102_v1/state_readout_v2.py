"""Spatial readout from a context-only selective scan; no detector or data loading.

The selective recurrence and query-generated C are existing mechanisms. The
prototype decouples spatial context length from sparse, geometric read events.
An event has delta=u=B=0, so it only observes the current context state.
"""
import torch
import torch.nn.functional as F


def stencil(reference_xy, hw):
    h, w = hw
    xy = reference_xy * reference_xy.new_tensor([w, h]) - .5
    x, y = xy.unbind(-1)
    x, y = x.clamp(0, w-1), y.clamp(0, h-1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
    dx, dy = x-x0, y-y0
    ids = torch.stack([y0*w+x0, y0*w+x1, y1*w+x0, y1*w+x1], -1)
    weights = torch.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], -1)
    return ids, weights


def _select(x, indices):
    return x.gather(2, indices.flatten(1).unsqueeze(1).expand(-1, x.shape[1], -1))


def _prepare(u, delta, A, B, z, reference_xy, hw, reverse):
    n = hw[0]*hw[1]
    assert u.shape[-1] == delta.shape[-1] == B.shape[-1] == z.shape[-1] == n
    assert A.shape == (u.shape[1], B.shape[1])
    assert delta.shape == z.shape == u.shape
    ids, weights = stencil(reference_xy, hw)
    if reverse:
        u, delta, B, z = [x.flip(-1) for x in [u, delta, B, z]]
        ids = n-1-ids
    return u, delta, B, z, ids, weights


def reference_states(u, delta, A, B):
    """Small synthetic gold only; deliberately materializes the hidden field."""
    state = u.new_zeros(u.shape[0], u.shape[1], B.shape[1])
    history = []
    for t in range(u.shape[-1]):
        d = delta[:, :, t, None]
        state = torch.exp(d*A[None])*state + d*u[:, :, t, None]*B[:, None, :, t]
        history.append(state)
    return torch.stack(history, -1)


def reference_scan(u, delta, A, B, C):
    """SSM output reference, with explicit nonnegative steps and no bias/gate."""
    state = u.new_zeros(u.shape[0], u.shape[1], B.shape[1])
    ys = []
    for t in range(u.shape[-1]):
        d = delta[:, :, t, None]
        state = torch.exp(d*A[None])*state + d*u[:, :, t, None]*B[:, None, :, t]
        ys.append((state*C[:, None, :, t]).sum(-1))
    return torch.stack(ys, -1)


def _finish(values, u, z, D, ids, weights):
    b, q, four = ids.shape
    skip = _select(u, ids)*D[None, :, None]
    gated = (values+skip)*F.silu(_select(z, ids))
    return (gated.reshape(b, u.shape[1], q, four)*weights[:, None]).sum(-1).transpose(1, 2)


def materialized_read(u, delta, A, B, z, D, Cquery, reference_xy, hw,
                      Ccontext=None, reverse=False):
    u, delta, B, z, ids, weights = _prepare(u, delta, A, B, z, reference_xy, hw, reverse)
    history = reference_states(u, delta, A, B)
    b, d, s, _ = history.shape
    picked = history.gather(3, ids.flatten(1)[:, None, None].expand(-1, d, s, -1))
    if Ccontext is None:
        c = Cquery.transpose(1, 2).repeat_interleave(4, dim=-1)
    else:
        c = _select(Ccontext.flip(-1) if reverse else Ccontext, ids)
    values = (picked*c[:, None]).sum(2)
    return _finish(values, u, z, D, ids, weights)


def encoded_feature_read(u, delta, A, B, z, D, Ccontext, reference_xy, hw,
                         scan=reference_scan, reverse=False):
    """Strong simple control: ordinary full-grid SSM encoding, then sampling."""
    u, delta, B, z, ids, weights = _prepare(u, delta, A, B, z, reference_xy, hw, reverse)
    c = Ccontext.flip(-1) if reverse else Ccontext
    values = scan(u.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(), c.contiguous())
    return _finish(_select(values, ids), u, z, D, ids, weights)


def packed_query_read(u, delta, A, B, z, D, Cquery, reference_xy, hw,
                       scan=reference_scan, reverse=False):
    """O(N+4Q) scan storage; never constructs a full D-by-state hidden field.

    Context projections/convolution must be computed BEFORE packing, preventing
    inserted read events from changing context convolution or its scan clock.
    Bilinear corner readings use the query-specific C and local context gate.
    """
    u, delta, B, z, ids, weights = _prepare(u, delta, A, B, z, reference_xy, hw, reverse)
    batch, channels, n = u.shape
    q = reference_xy.shape[1]
    events = 4*q
    ctx_keys = torch.arange(n, device=u.device)[None].expand(batch, -1)*2
    keys = torch.cat([ctx_keys, ids.flatten(1)*2+1], dim=1)
    order = keys.argsort(dim=1, stable=True)
    inverse = order.argsort(1)

    def pack(context, read):
        x = torch.cat([context, read], dim=-1)
        return x.gather(2, order[:, None].expand(-1, x.shape[1], -1)).contiguous()

    up = pack(u, u.new_zeros(batch, channels, events))
    dp = pack(delta, delta.new_zeros(batch, channels, events))
    bp = pack(B, B.new_zeros(batch, B.shape[1], events))
    cp = pack(B.new_zeros(B.shape), Cquery.transpose(1, 2).repeat_interleave(4, dim=-1))
    out = scan(up, dp, A.contiguous(), bp, cp)
    values = out.gather(2, inverse[:, None, n:].expand(-1, channels, -1))
    return _finish(values, u, z, D, ids, weights)
