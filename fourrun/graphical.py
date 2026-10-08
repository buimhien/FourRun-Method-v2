"""
Graphical (circle-intersection) solution, reproduced numerically.

Circle C_i has centre O_i = V0 (cos alpha_i, sin alpha_i) and radius V_i.
For noise-free data the three circles pass through the common point
P = T (cos theta, sin theta), where theta is the correction angle, so |P| = T
and m_c = m_t (r_t/r_c) V0 / |P|.

Algorithm (matches the web tool used for the case studies):
 1. For each pair (i, j) compute the intersection points. When the pair does
    not intersect, the foot of the radical axis on the line of centres is
    used as the single candidate (this is continuous at tangency).
 2. Of the two candidates keep the one with the smaller residual
    | |P - O_k| - V_k | against the third circle k.
 3. The solution is the UNWEIGHTED mean of the three retained points.
 4. The spread Delta_max = max_k |P_k - P_mean| is reported.

Pairwise intersection condition (centres are sqrt(3) V0 apart):
      |V_i - V_j| <= sqrt(3) V0 <= V_i + V_j
"""
import numpy as np
from .core import ALPHA, SQ3

PAIRS = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]   # (i, j, k) with radii index 0..2


def pair_intersects(V0, Vi, Vj):
    d = SQ3 * np.asarray(V0, float)
    return (np.abs(Vi - Vj) <= d) & (d <= Vi + Vj)


def graphical_solve(V0, V1, V2, V3, mt=1.0, rt=1.0, rc=None):
    V0, V1, V2, V3 = [np.asarray(a, float) for a in (V0, V1, V2, V3)]
    shape = np.broadcast(V0, V1, V2, V3).shape
    V0 = np.broadcast_to(V0, shape)
    R = [np.broadcast_to(v, shape) for v in (V1, V2, V3)]
    O = [np.stack([V0 * np.cos(a), V0 * np.sin(a)], axis=-1) for a in ALPHA]
    pts, n_int = [], np.zeros(shape, dtype=int)
    for i, j, k in PAIRS:
        c1, c2, r1, r2 = O[i], O[j], R[i], R[j]
        dvec = c2 - c1
        d = np.linalg.norm(dvec, axis=-1)
        a = (d**2 + r1**2 - r2**2) / (2.0 * d)
        inter = pair_intersects(V0, r1, r2)
        n_int = n_int + inter.astype(int)
        h = np.sqrt(np.clip(r1**2 - a**2, 0.0, None)) * inter
        u = dvec / d[..., None]
        perp = np.stack([-u[..., 1], u[..., 0]], axis=-1)
        base = c1 + a[..., None] * u
        pA = base + h[..., None] * perp
        pB = base - h[..., None] * perp
        resA = np.abs(np.linalg.norm(pA - O[k], axis=-1) - R[k])
        resB = np.abs(np.linalg.norm(pB - O[k], axis=-1) - R[k])
        pts.append(np.where((resA <= resB)[..., None], pA, pB))
    P = np.stack(pts, axis=0)                 # (3, ..., 2)
    Pm = P.mean(axis=0)
    spread = np.max(np.linalg.norm(P - Pm[None], axis=-1), axis=0)
    T = np.linalg.norm(Pm, axis=-1)
    rc = rt if rc is None else rc
    with np.errstate(divide="ignore", invalid="ignore"):
        m = mt * (rt / rc) * V0 / T
    theta = np.mod(np.rad2deg(np.arctan2(Pm[..., 1], Pm[..., 0])), 360.0)
    return dict(T=T, m=m, theta=theta, spread=spread, n_intersecting=n_int,
                points=P, P=Pm)
