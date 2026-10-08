/*
 * fourrun.js - reference implementation (JavaScript) of the single-plane
 * Four-run computations, mirroring fourrun/core.py and fourrun/graphical.py.
 * Intended as a drop-in computational core for the Digital Balancing
 * Assistant web tool. MIT licence.
 */
(function (root) {
  "use strict";
  const SQ3 = Math.sqrt(3);
  const ALPHA = [0, 120, 240].map((d) => (d * Math.PI) / 180);
  const STATUS = { OK: 0, T2SUM_NEGATIVE: 1, T2SUM_ZERO: 2, TXY_TOO_SMALL: 3, V0_INVALID: 4 };

  function estimators(V0, V1, V2, V3, txyEps = 1e-9) {
    const T2sum = (V1 * V1 + V2 * V2 + V3 * V3) / 3 - V0 * V0;
    const x = (2 * V1 * V1 - V2 * V2 - V3 * V3) / (6 * V0);
    const y = (V2 * V2 - V3 * V3) / (2 * SQ3 * V0);
    const Txy = Math.hypot(x, y);
    const g = T2sum - Txy * Txy;
    let Tsum = NaN, rho = NaN, status = STATUS.OK;
    if (T2sum > 0) { Tsum = Math.sqrt(T2sum); rho = Tsum / Txy; }
    else if (T2sum === 0) { Tsum = 0; rho = 0; status = STATUS.T2SUM_ZERO; }
    else { status = STATUS.T2SUM_NEGATIVE; }
    if (Txy <= txyEps * Math.max(V0, 1)) status = STATUS.TXY_TOO_SMALL;
    if (!(V0 > 0)) status = STATUS.V0_INVALID;
    // rho needs a valid denominator: undefined when Txy ~ 0 or V0 <= 0
    if (status === STATUS.TXY_TOO_SMALL || status === STATUS.V0_INVALID) rho = NaN;
    return { T2sum, Tsum, x, y, Txy, g, rho, status };
  }

  function correction(V0, V1, V2, V3, mt, rt = 1, rc = null, estimator = "xy") {
    const e = estimators(V0, V1, V2, V3);
    rc = rc === null ? rt : rc;
    const T = estimator === "xy" ? e.Txy : e.Tsum;
    const m = (mt * (rt / rc) * V0) / T;
    const phi = (Math.atan2(e.y, e.x) * 180) / Math.PI;
    const theta = (((phi + 180) % 360) + 360) % 360;
    return Object.assign(e, { m, theta, phi: ((phi % 360) + 360) % 360 });
  }

  function gradients(V0, V1, V2, V3) {
    const x = (2 * V1 * V1 - V2 * V2 - V3 * V3) / (6 * V0);
    const y = (V2 * V2 - V3 * V3) / (2 * SQ3 * V0);
    const dx = [-x / V0, (2 * V1) / (3 * V0), -V2 / (3 * V0), -V3 / (3 * V0)];
    const dy = [-y / V0, 0, V2 / (SQ3 * V0), -V3 / (SQ3 * V0)];
    const dT2sum = [-2 * V0, (2 * V1) / 3, (2 * V2) / 3, (2 * V3) / 3];
    const dTxy2 = dx.map((_, i) => 2 * (x * dx[i] + y * dy[i]));
    return { dT2sum, dTxy2 };
  }

  function sigVec(sigma) { return Array.isArray(sigma) ? sigma : [sigma, sigma, sigma, sigma]; }

  /* First-order SD of the closure residual g = T2sum - Txy^2. */
  function sdClosure(V0, V1, V2, V3, sigma) {
    const { dT2sum, dTxy2 } = gradients(V0, V1, V2, V3);
    const s = sigVec(sigma);
    let acc = 0;
    for (let i = 0; i < 4; i++) acc += Math.pow(s[i] * (dT2sum[i] - dTxy2[i]), 2);
    return Math.sqrt(acc);
  }

  /* Standardised closure residual z = g / SD(g). */
  function zClosure(V0, V1, V2, V3, sigma) {
    return estimators(V0, V1, V2, V3).g / sdClosure(V0, V1, V2, V3, sigma);
  }

  function sdRho(V0, V1, V2, V3, sigma) {
    const e = estimators(V0, V1, V2, V3);
    if (!(e.T2sum > 0)) return NaN;
    const { dT2sum, dTxy2 } = gradients(V0, V1, V2, V3);
    const s = sigVec(sigma);
    let acc = 0;
    for (let i = 0; i < 4; i++) {
      const dl = (0.5 * dT2sum[i]) / e.T2sum - (0.5 * dTxy2[i]) / (e.Txy * e.Txy);
      acc += Math.pow(s[i] * dl, 2);
    }
    return e.rho * Math.sqrt(acc);
  }

  function pairIntersects(V0, Vi, Vj) {
    const d = SQ3 * V0;
    return Math.abs(Vi - Vj) <= d && d <= Vi + Vj;
  }

  /* Graphical solution: unweighted mean of the three retained pairwise points. */
  function graphicalSolve(V0, V1, V2, V3, mt = 1, rt = 1, rc = null) {
    const R = [V1, V2, V3];
    const O = ALPHA.map((a) => [V0 * Math.cos(a), V0 * Math.sin(a)]);
    const pairs = [[0, 1, 2], [0, 2, 1], [1, 2, 0]];
    const pts = []; let nInt = 0;
    for (const [i, j, k] of pairs) {
      const dvx = O[j][0] - O[i][0], dvy = O[j][1] - O[i][1];
      const d = Math.hypot(dvx, dvy);
      const a = (d * d + R[i] * R[i] - R[j] * R[j]) / (2 * d);
      const inter = pairIntersects(V0, R[i], R[j]);
      if (inter) nInt++;
      const h = inter ? Math.sqrt(Math.max(R[i] * R[i] - a * a, 0)) : 0;
      const ux = dvx / d, uy = dvy / d;
      const bx = O[i][0] + a * ux, by = O[i][1] + a * uy;
      const pA = [bx - h * uy, by + h * ux], pB = [bx + h * uy, by - h * ux];
      const res = (p) => Math.abs(Math.hypot(p[0] - O[k][0], p[1] - O[k][1]) - R[k]);
      pts.push(res(pA) <= res(pB) ? pA : pB);
    }
    const P = [(pts[0][0] + pts[1][0] + pts[2][0]) / 3, (pts[0][1] + pts[1][1] + pts[2][1]) / 3];
    const spread = Math.max(...pts.map((p) => Math.hypot(p[0] - P[0], p[1] - P[1])));
    const T = Math.hypot(P[0], P[1]);
    rc = rc === null ? rt : rc;
    const theta = ((((Math.atan2(P[1], P[0]) * 180) / Math.PI) % 360) + 360) % 360;
    return { T, m: (mt * (rt / rc) * V0) / T, theta, spread, nIntersecting: nInt, points: pts, P };
  }

  /* First-order standard uncertainty of the correction (Eq. 13, general
     gradient form): returns {uMRel, uThetaDeg}. */
  function predictedUncertainty(V0, V1, V2, V3, sigma) {
    const x = (2 * V1 * V1 - V2 * V2 - V3 * V3) / (6 * V0);
    const y = (V2 * V2 - V3 * V3) / (2 * SQ3 * V0);
    const dx = [-x / V0, (2 * V1) / (3 * V0), -V2 / (3 * V0), -V3 / (3 * V0)];
    const dy = [-y / V0, 0, V2 / (SQ3 * V0), -V3 / (SQ3 * V0)];
    const T2 = x * x + y * y, s = sigVec(sigma);
    let am = 0, at = 0;
    for (let i = 0; i < 4; i++) {
      const dlnT = (x * dx[i] + y * dy[i]) / T2;
      const dlnm = (i === 0 ? 1 / V0 : 0) - dlnT;
      const dphi = (x * dy[i] - y * dx[i]) / T2;
      am += Math.pow(s[i] * dlnm, 2); at += Math.pow(s[i] * dphi, 2);
    }
    return { uMRel: Math.sqrt(am), uThetaDeg: (Math.sqrt(at) * 180) / Math.PI };
  }

  /* Pooled noise estimate from k repeated readings per run:
     readings = [[V0 readings], [V1 readings], [V2 ...], [V3 ...]]. */
  function pooledSigma(readings) {
    const k = readings[0].length;
    const means = readings.map((r) => r.reduce((a, b) => a + b, 0) / k);
    let ss = 0;
    readings.forEach((r, i) => r.forEach((v) => { ss += (v - means[i]) * (v - means[i]); }));
    const nu = 4 * (k - 1);
    return { means, sigmaMean: Math.sqrt(ss / nu) / Math.sqrt(k), nu };
  }

  /* Student t quantile, computed from the regularised incomplete beta
     function (continued fraction) and bisection; valid for any nu > 0. */
  function logGamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
      0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x; const tmp = x + 5.5 - (x + 0.5) * Math.log(x + 5.5);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }
  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 1e-15, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d; let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function betai(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
  }
  function tCdf(t, nu) {
    const p = 0.5 * betai(nu / 2, 0.5, nu / (nu + t * t));
    return t >= 0 ? 1 - p : p;
  }
  function tQuantile(p, nu) {
    let lo = 0, hi = 1;
    while (tCdf(hi, nu) < p) hi *= 2;
    for (let i = 0; i < 200; i++) {
      const mid = 0.5 * (lo + hi);
      if (tCdf(mid, nu) < p) lo = mid; else hi = mid;
      if (hi - lo < 1e-13) break;
    }
    return 0.5 * (lo + hi);
  }
  function criticalValue(nu, level = 0.95) { return nu == null ? 2 : tQuantile(0.5 + level / 2, nu); }

  /* Closure check of Section 4.3. sigma: noise of the averaged amplitudes;
     nu: degrees of freedom if sigma was estimated (null if known). */
  function screen(V0, V1, V2, V3, sigma, nu = null, minVFactor = 5) {
    const e = estimators(V0, V1, V2, V3);
    const crit = criticalValue(nu);
    if (e.status === STATUS.TXY_TOO_SMALL || e.status === STATUS.V0_INVALID)
      return { decision: "not computable", z: NaN, crit, status: e.status };
    const ug = sdClosure(V0, V1, V2, V3, sigma);
    if (!(ug > 0)) return { decision: "not assessable", z: NaN, crit, status: e.status };
    const z = e.g / ug;
    const sMax = Array.isArray(sigma) ? Math.max(...sigma) : sigma;
    let decision;
    if (Math.abs(z) > crit) decision = "flagged";
    else if (Math.min(V0, V1, V2, V3) < minVFactor * sMax) decision = "not assessable";
    else decision = "no inconsistency detected";
    const u = predictedUncertainty(V0, V1, V2, V3, sigma);
    const k95 = nu == null ? 1.96 : crit;
    return { decision, z, crit, status: e.status, rho: e.rho, U95MRel: k95 * u.uMRel, U95ThetaDeg: k95 * u.uThetaDeg };
  }

  const api = { STATUS, estimators, correction, sdClosure, zClosure, sdRho, predictedUncertainty, pooledSigma,
    criticalValue, tQuantile, pairIntersects, graphicalSolve, screen };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FourRun = api;
})(this);
