"""
Reference implementation of the single-plane Four-run (phase-free) balancing
computations used in the revised manuscript.

Conventions
-----------
* Trial weight positions alpha_i = 0, 120, 240 deg, measured in the direction
  in which the trial weight is moved (the web tool calls this CCW).
* Measured amplitudes V0 (no trial weight) and V1, V2, V3 (trial weight at
  alpha_1..3). The noise-free model is
      V_i^2 = V0^2 + T^2 + 2 V0 T cos(phi - alpha_i),       i = 1, 2, 3
  where T is the trial-weight effect and phi the angle of the unbalance
  response expressed in the trial-weight frame.
* Correction angle theta = (phi + 180) mod 360 deg.
* Correction mass m_c = m_t (r_t / r_c) V0 / T.

All functions accept numpy arrays (vectorised) so that the same code is used
for the case studies and for the Monte Carlo simulations.
"""
import numpy as np

ALPHA_DEG = np.array([0.0, 120.0, 240.0])
ALPHA = np.deg2rad(ALPHA_DEG)
SQ3 = np.sqrt(3.0)

# status codes
OK = 0
T2SUM_NEGATIVE = 1      # rho undefined (T_sum not real)
T2SUM_ZERO = 2          # degenerate, rho = 0
TXY_TOO_SMALL = 3       # correction mass not computable
V0_INVALID = 4


def _arr(*v):
    return [np.asarray(a, dtype=float) for a in v]


def estimators(V0, V1, V2, V3, txy_eps=1e-9):
    """Return the two trial-effect estimators, the closure residual g and rho.

    T2sum = (V1^2 + V2^2 + V3^2)/3 - V0^2                       (Eq. 3)
    x     = (2 V1^2 - V2^2 - V3^2) / (6 V0)                      (Eq. 4)
    y     = (V2^2 - V3^2) / (2 sqrt(3) V0)
    Txy   = sqrt(x^2 + y^2)
    g     = T2sum - Txy^2   (closure residual; zero for noise-free data)
    rho   = sqrt(T2sum) / Txy   (0 when T2sum = 0 and Txy > 0; undefined when
            T2sum < 0, Txy ~ 0 or V0 <= 0)
    """
    V0, V1, V2, V3 = _arr(V0, V1, V2, V3)
    with np.errstate(divide="ignore", invalid="ignore"):
        T2sum = (V1**2 + V2**2 + V3**2) / 3.0 - V0**2
        x = (2.0 * V1**2 - V2**2 - V3**2) / (6.0 * V0)
        y = (V2**2 - V3**2) / (2.0 * SQ3 * V0)
        Txy = np.hypot(x, y)
        g = T2sum - Txy**2
        Tsum = np.where(T2sum > 0, np.sqrt(np.where(T2sum > 0, T2sum, 0.0)), np.nan)
        Tsum = np.where(T2sum == 0, 0.0, Tsum)
        rho = Tsum / Txy
    status = np.full(np.broadcast(V0, V1, V2, V3).shape, OK, dtype=int)
    status = np.where(T2sum < 0, T2SUM_NEGATIVE, status)
    status = np.where(T2sum == 0, T2SUM_ZERO, status)
    status = np.where(Txy <= txy_eps * np.maximum(V0, 1.0), TXY_TOO_SMALL, status)
    status = np.where(~(V0 > 0), V0_INVALID, status)
    # rho needs a valid denominator: undefined when Txy ~ 0 or V0 <= 0
    rho = np.where((status == TXY_TOO_SMALL) | (status == V0_INVALID), np.nan, rho)
    return dict(T2sum=T2sum, Tsum=Tsum, x=x, y=y, Txy=Txy, g=g, rho=rho,
                status=status)


def correction(V0, V1, V2, V3, mt, rt=1.0, rc=None, estimator="xy"):
    """Correction mass and angle.

    estimator = "xy"  : T = Txy (recommended, see Section 4.1.4)
    estimator = "sum" : T = Tsum (angle still taken from (x, y), because
                        Tsum carries no angular information)
    """
    e = estimators(V0, V1, V2, V3)
    rc = rt if rc is None else rc
    T = e["Txy"] if estimator == "xy" else e["Tsum"]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = mt * (rt / rc) * np.asarray(V0, float) / T
    phi = np.rad2deg(np.arctan2(e["y"], e["x"]))
    theta = np.mod(phi + 180.0, 360.0)
    e.update(m=m, theta=theta, phi=np.mod(phi, 360.0))
    return e


# ---------------------------------------------------------------------------
# First-order (delta-method) uncertainty of the closure residual and of rho
# ---------------------------------------------------------------------------
def _gradients(V0, V1, V2, V3):
    """Gradients of T2sum, Txy^2, Tsum, Txy with respect to (V0, V1, V2, V3).
    Returned arrays have a leading axis of length 4 (V0, V1, V2, V3)."""
    V0, V1, V2, V3 = _arr(V0, V1, V2, V3)
    x = (2.0 * V1**2 - V2**2 - V3**2) / (6.0 * V0)
    y = (V2**2 - V3**2) / (2.0 * SQ3 * V0)
    z = np.zeros_like(V0 + V1 + V2 + V3)
    dx = np.stack([-x / V0 + z, 2.0 * V1 / (3.0 * V0) + z,
                   -V2 / (3.0 * V0) + z, -V3 / (3.0 * V0) + z])
    dy = np.stack([-y / V0 + z, z, V2 / (SQ3 * V0) + z, -V3 / (SQ3 * V0) + z])
    dT2sum = np.stack([-2.0 * V0 + z, 2.0 * V1 / 3.0 + z,
                       2.0 * V2 / 3.0 + z, 2.0 * V3 / 3.0 + z])
    dTxy2 = 2.0 * (x * dx + y * dy)
    return dT2sum, dTxy2, x, y


def _sig_vec(sigma, shape_like):
    """sigma may be: a scalar; a length-4 sequence (sigma_0..sigma_3); an
    array with the sample shape (one common sigma per sample, e.g. a pooled
    estimate); or an array of shape (4, *sample_shape)."""
    s = np.asarray(sigma, dtype=float)
    nd = np.ndim(shape_like)
    if s.ndim == 0:
        s = np.full(4, float(s))
    if s.ndim == 1 and s.shape[0] == 4 and nd != 1:
        return s.reshape((4,) + (1,) * nd)
    if s.ndim == 1 and s.shape[0] == 4 and nd == 1 and np.shape(shape_like)[0] != 4:
        return s.reshape(4, 1)
    if s.shape == np.shape(shape_like):
        return np.broadcast_to(s, (4,) + s.shape)
    return s


def sd_closure(V0, V1, V2, V3, sigma):
    """First-order standard deviation of g = T2sum - Txy^2.
    sigma: scalar or length-4 sequence (sigma_0..sigma_3) of the amplitude
    noise applied to the values actually used (i.e. after averaging)."""
    dT2sum, dTxy2, _, _ = _gradients(V0, V1, V2, V3)
    dg = dT2sum - dTxy2
    s = _sig_vec(sigma, dg[0])
    return np.sqrt(np.sum((s * dg) ** 2, axis=0))


def z_closure(V0, V1, V2, V3, sigma):
    """Standardised closure residual z = g / SD(g). Defined even when
    T2sum <= 0. For rho close to one, z ~ (rho - 1) / SD(rho)."""
    e = estimators(V0, V1, V2, V3)
    return e["g"] / sd_closure(V0, V1, V2, V3, sigma)


def sd_rho(V0, V1, V2, V3, sigma):
    """First-order standard deviation of rho (only where T2sum > 0)."""
    e = estimators(V0, V1, V2, V3)
    dT2sum, dTxy2, _, _ = _gradients(V0, V1, V2, V3)
    with np.errstate(divide="ignore", invalid="ignore"):
        # d ln rho = 0.5 d ln T2sum - 0.5 d ln Txy^2
        dl = 0.5 * dT2sum / e["T2sum"] - 0.5 * dTxy2 / e["Txy"]**2
        s = _sig_vec(sigma, dl[0])
        out = e["rho"] * np.sqrt(np.sum((s * dl) ** 2, axis=0))
    return np.where(e["T2sum"] > 0, out, np.nan)


def predicted_uncertainty(V0, V1, V2, V3, sigma):
    """First-order standard uncertainty of the correction (Txy estimator).
    Returns (u_m_rel, u_theta_deg): relative standard uncertainty of m_c and
    standard uncertainty of theta. Noise in V0 does not enter theta."""
    V0, V1, V2, V3 = _arr(V0, V1, V2, V3)
    _, _, x, y = _gradients(V0, V1, V2, V3)
    z = np.zeros_like(V0 + V1 + V2 + V3)
    dx = np.stack([-x / V0 + z, 2.0 * V1 / (3.0 * V0) + z,
                   -V2 / (3.0 * V0) + z, -V3 / (3.0 * V0) + z])
    dy = np.stack([-y / V0 + z, z, V2 / (SQ3 * V0) + z, -V3 / (SQ3 * V0) + z])
    T2 = x**2 + y**2
    dlnT = (x * dx + y * dy) / T2
    dlnV0 = np.stack([1.0 / V0 + z, z, z, z])
    dlnm = dlnV0 - dlnT
    dphi = (x * dy - y * dx) / T2
    s = _sig_vec(sigma, dlnm[0])
    u_m = np.sqrt(np.sum((s * dlnm) ** 2, axis=0))
    u_t = np.rad2deg(np.sqrt(np.sum((s * dphi) ** 2, axis=0)))
    return u_m, u_t


# ---------------------------------------------------------------------------
# Closed-form first-order results quoted in Section 4.1 (equal sigma)
# ---------------------------------------------------------------------------
def theory(V0, T, phi_deg, sigma):
    """Bias and variance expressions derived in Section 4.1 / Appendix A
    for equal, independent Gaussian noise of standard deviation sigma on
    V0..V3 (first order in sigma^2)."""
    r = V0 / T
    c3 = np.cos(np.deg2rad(3.0 * phi_deg))
    return dict(
        bias_T2sum=0.0,
        bias_T2xy=sigma**2 / 3.0 * (4.0 + 13.0 / r**2),
        var_Tsum=sigma**2 * (4.0 * r**2 + 1.0) / 3.0,
        var_Txy=sigma**2 * (2.0 / 3.0 + 2.0 / 3.0 * c3 / r + 5.0 / 3.0 / r**2),
        # Eq. (11): standard deviation of rho for consistent data
        sd_rho=(sigma / T) * np.sqrt(4.0 / 3.0 * r**2 - 7.0 / 3.0 + 5.0 / (3.0 * r**2) + 2.0 / 3.0 * c3 / r),
        # Eq. (13): relative standard uncertainty of m_c and of theta (rad)
        u_m_rel=(sigma / T) * np.sqrt(2.0 / 3.0 + 2.0 / 3.0 * c3 / r + 14.0 / (3.0 * r**2)),
        u_theta_rad=(sigma / T) * np.sqrt(2.0 / 3.0 * (1.0 + 1.0 / r**2 - c3 / r)),
    )


def ideal_amplitudes(V0, T, phi_deg):
    """Noise-free V1..V3 from the model for given V0, T, phi."""
    phi = np.deg2rad(phi_deg)
    V0 = np.asarray(V0, float)
    return [np.sqrt(V0**2 + T**2 + 2.0 * V0 * T * np.cos(phi - a)) for a in ALPHA]


# ---------------------------------------------------------------------------
# Noise estimate from repeated readings and the screening decision
# ---------------------------------------------------------------------------
def pooled_sigma(readings):
    """readings: array-like of shape (4, k[, ...]) holding the k repeated
    readings of each run. Returns (means, sigma_mean_hat, nu) where
    sigma_mean_hat = s_p / sqrt(k) is the estimated standard deviation of the
    averaged amplitudes and nu = 4 (k - 1) its degrees of freedom."""
    R = np.asarray(readings, dtype=float)
    k = R.shape[1]
    m = R.mean(axis=1)
    sp = np.sqrt(((R - m[:, None]) ** 2).sum(axis=(0, 1)) / (4 * (k - 1)))
    return m, sp / np.sqrt(k), 4 * (k - 1)


def critical_value(nu=None, level=0.95):
    """2 for a known sigma (as used in the paper), Student t for an
    estimated sigma with nu degrees of freedom."""
    if nu is None:
        return 2.0
    from scipy import stats
    return float(stats.t.ppf(0.5 + level / 2.0, nu))


def screen(V0, V1, V2, V3, sigma, nu=None, minV_factor=5.0):
    """Closure check of Section 4.3 (scalar inputs).
    Returns a dict with z, the critical value and one of the decisions
    'not computable', 'flagged', 'not assessable', 'no inconsistency detected'.
    'not assessable': |z| below the critical value while some amplitude is
    smaller than minV_factor * sigma; the first-order check has little power
    there, so the absence of a flag is not evidence of consistency."""
    e = estimators(V0, V1, V2, V3)
    crit = critical_value(nu)
    if int(e["status"]) in (TXY_TOO_SMALL, V0_INVALID):
        return dict(decision="not computable", z=float("nan"), crit=crit, status=int(e["status"]))
    ug = float(sd_closure(V0, V1, V2, V3, sigma))
    if not ug > 0:
        return dict(decision="not assessable", z=float("nan"), crit=crit, status=int(e["status"]))
    z = float(e["g"]) / ug
    if abs(z) > crit:
        d = "flagged"
    elif min(V0, V1, V2, V3) < minV_factor * float(np.max(sigma)):
        d = "not assessable"
    else:
        d = "no inconsistency detected"
    um, ut = predicted_uncertainty(V0, V1, V2, V3, sigma)
    k95 = 1.96 if nu is None else crit
    return dict(decision=d, z=z, crit=crit, status=int(e["status"]), rho=float(e["rho"]),
                U95_m_rel=float(k95 * um), U95_theta_deg=float(k95 * ut))
