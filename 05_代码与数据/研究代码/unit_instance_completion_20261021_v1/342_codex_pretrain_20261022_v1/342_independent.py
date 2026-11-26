"""Independent oracle: NumPy/Python, no geom342 or 311 imports.

d_out is evaluated in the raw 640 plane by closest-point projection, with
scalar row/column loops. Thus it does not reuse normalized broadcasting or
the main implementation's grid ordering assumptions.
"""
import numpy as np

def intersection(boxes, window):
    selected = []
    left, top, right, bottom = map(float, window)
    for box in boxes:
        x0, y0, x1, y1 = map(float, box)
        selected.append(not (x1 <= left or x0 >= right or y1 <= top or y0 >= bottom))
    return np.asarray(selected, dtype=bool)

def normalized_edges(boxes, window):
    out = np.empty((len(boxes), 4), dtype=np.float64)
    left, top, right, bottom = map(float, window)
    for j, (x0, y0, x1, y1) in enumerate(boxes):
        out[j] = [(float(x0)-left)/(right-left), (float(y0)-top)/(bottom-top),
                  (float(x1)-left)/(right-left), (float(y1)-top)/(bottom-top)]
    return out

def geometry_raw(boxes, window):
    left, top, right, bottom = map(float, window)
    out = np.zeros((256, len(boxes)), dtype=np.float64)
    for row in range(16):
        y = top + (row + .5) * (bottom-top) / 16
        for col in range(16):
            x = left + (col + .5) * (right-left) / 16
            for j, (x0, y0, x1, y1) in enumerate(boxes):
                x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
                closest_x = min(max(x, x0), x1)
                closest_y = min(max(y, y0), y1)
                out[row*16+col, j] = abs(x-closest_x)/(x1-x0) + abs(y-closest_y)/(y1-y0)
    return out

def plan(mass, geometry, classes, kappa):
    C, I = mass.shape
    J = len(classes)
    p = np.zeros((I, J), dtype=np.float64)
    p0 = np.zeros(J, dtype=np.float64)
    for j, c in enumerate(classes):
        kernel = mass[c] * np.exp(-geometry[:, j])
        null = np.exp(-kappa[c])
        p[:, j] = kernel / (kernel.sum() + null)
        p0[j] = null / (kernel.sum() + null)
    u = np.zeros_like(mass, dtype=np.float64)
    for j, c in enumerate(classes):
        u[c] += p[:, j]
    a = np.ones_like(u)
    for c in range(C):
        for i in range(I):
            if u[c, i] > 0:
                a[c, i] = min(1., mass[c, i] / u[c, i])
    T = p.copy()
    explained = np.zeros_like(u)
    z = p0.copy()
    for j, c in enumerate(classes):
        T[:, j] *= a[c]
        explained[c] += T[:, j]
        z[j] += (p[:, j] - T[:, j]).sum()
    return dict(T=T, z=z, residual=np.maximum(0., mass-explained))
