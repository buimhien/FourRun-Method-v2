"""Additional checks of the closure test and of the predicted uncertainty:
 A. noise estimated from repeated readings (pooled, nu = 4(k-1) = 8)
 B. dependence on the phase phi (fixed-phase grid)
 C. mis-stated noise level
 D. departures from the noise model (unequal variance, correlated runs, common gain)
 E. coverage of the U95 intervals (marginal and joint) against known reference values
 F. z of the three cases as a function of the assumed device noise
Output: results/robustness.json"""
import json, pathlib, numpy as np, fourrun as fr
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "results/cases.json").read_text())
SEED, N, K = 20261005, 50_000, 3
NU = 4 * (K - 1)
T_NU = float(stats.t.ppf(0.975, NU))
R_GRID = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
S_GRID = [10, 20, 35, 50, 100]
MINV_FACTOR = 5.0
GATE_FACTORS = [5.0, 8.0, 10.0, 15.0]
rng = np.random.default_rng(SEED)


def circ(a):
    return (a + 180.0) % 360.0 - 180.0


def readings(Vt, sig_dev, n, k=K, sig_run=None, corr=0.0):
    """k readings per run; optional per-run sigma and correlation between runs."""
    sr = np.full(4, sig_dev) if sig_run is None else np.asarray(sig_run)
    out = []
    common = rng.normal(0, 1, (k, n))
    for i, v in enumerate(Vt):
        e = np.sqrt(1 - corr) * rng.normal(0, 1, (k, n)) + np.sqrt(corr) * common
        out.append(v[None, :] + sr[i] * e)
    return np.stack(out)                     # (4, k, n)


def pooled(reads):
    m = reads.mean(1)
    sp = np.sqrt(((reads - m[:, None, :]) ** 2).sum((0, 1)) / NU)
    return m, sp / np.sqrt(reads.shape[1])


def case_truth(key):
    c = CASES[key]
    return c["V"][0], c["Txy"], (c["theta"] - 180.0) % 360.0, c["theta"]


def fault_amplitudes(V0, T, phi, dang=(0, 0, 0), v0_scale=1.0):
    al = np.deg2rad(np.array([0, 120, 240]) + np.array(dang)); p = np.deg2rad(phi)
    return [V0 * v0_scale] + [np.sqrt(V0**2 + T**2 + 2 * V0 * T * np.cos(p - a)) for a in al]


out = dict(seed=SEED, N=N, k=K, nu=NU, t_nu=T_NU, minV_factor=MINV_FACTOR)

# ---------------------------------------------------------------- A + E (grid, phi uniform)
gridA = []
for r in R_GRID:
    for s in S_GRID:
        phi = rng.uniform(0, 360, N)
        Vt = [np.full(N, r)] + fr.ideal_amplitudes(r, 1.0, phi)
        sig_dev = np.sqrt(K) / s
        reads = readings(Vt, sig_dev, N)
        Vm, se_hat = pooled(reads)
        se = sig_dev / np.sqrt(K)
        z_known = fr.z_closure(*Vm, se)
        z_est = fr.z_closure(*Vm, se_hat)
        assess = np.min(np.stack(Vm), axis=0) >= MINV_FACTOR * se_hat
        a = fr.correction(*Vm, mt=1.0)
        em = np.abs(a["m"] - r) / r
        et = np.abs(circ(a["theta"] - (phi + 180)))
        um, ut = fr.predicted_uncertainty(*Vm, se)
        um_h, ut_h = fr.predicted_uncertainty(*Vm, se_hat)
        # interval for the mass is m_hat +/- k * m_hat * u_rel(m_hat); the reference value r is inside when
        # |m_hat - r| <= k * m_hat * u_rel(m_hat)
        dm_abs = np.abs(a["m"] - r)
        in_m, in_t = dm_abs <= 1.96 * um * a["m"], et <= 1.96 * ut
        in_mh, in_th = dm_abs <= T_NU * um_h * a["m"], et <= T_NU * ut_h
        minV = np.min(np.stack(Vm), axis=0)
        flag_est, flag_known = np.abs(z_est) > T_NU, np.abs(z_known) > 2
        gate = {}
        for cfac in GATE_FACTORS:
            ass_e, ass_k = minV >= cfac * se_hat, minV >= cfac * se
            gate[str(cfac)] = dict(
                est=dict(flagged=float(flag_est.mean()), not_assessable=float((~flag_est & ~ass_e).mean()),
                         no_inconsistency=float((~flag_est & ass_e).mean()),
                         fa_given_assessable=float(flag_est[ass_e].mean()) if ass_e.any() else float("nan"),
                         share_assessable=float(ass_e.mean())),
                known=dict(flagged=float(flag_known.mean()), not_assessable=float((~flag_known & ~ass_k).mean()),
                           no_inconsistency=float((~flag_known & ass_k).mean()),
                           fa_given_assessable=float(flag_known[ass_k].mean()) if ass_k.any() else float("nan"),
                           share_assessable=float(ass_k.mean())))
        gridA.append(dict(r=r, s=s, gate=gate,
            fa_known_2=float(np.mean(np.abs(z_known) > 2)),
            fa_est_2=float(np.mean(np.abs(z_est) > 2)),
            fa_est_t=float(np.mean(np.abs(z_est) > T_NU)),
            fa_est_t_assessable=float(np.mean(np.abs(z_est[assess]) > T_NU)) if assess.any() else float("nan"),
            frac_not_assessable=float(1 - assess.mean()),
            cov_m=float(in_m.mean()), cov_t=float(in_t.mean()), cov_joint=float((in_m & in_t).mean()),
            cov_m_est=float(in_mh.mean()), cov_t_est=float(in_th.mean()), cov_joint_est=float((in_mh & in_th).mean())))
out["gridA"] = gridA

# ---------------------------------------------------------------- B (fixed phase grid, known sigma, s = 35)
phase = {}
PHI = np.arange(0.0, 120.0, 2.5)          # results repeat every 120 deg
NB = 20_000
for r in (0.75, 1.0, 1.25, 1.5, 2.0, 4.0):
    fa, mv = [], []
    for ph in PHI:
        Vt = [np.full(NB, r)] + fr.ideal_amplitudes(r, 1.0, np.full(NB, ph))
        se = 1.0 / 35
        Vm = [v + rng.normal(0, se, NB) for v in Vt]
        fa.append(float(np.mean(np.abs(fr.z_closure(*Vm, se)) > 2)))
        mv.append(float(min(v[0] for v in Vt) / se))
    phase[str(r)] = dict(phi=PHI.tolist(), fa=fa, minV_over_sigma=mv)
out["phase"] = phase

# ---------------------------------------------------------------- C (mis-stated sigma at the case configurations)
mis = {}
RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]
for key in ("BRT", "IBR", "ICC"):
    V0, T, phi, th = case_truth(key)
    se = 0.10 / np.sqrt(K)
    res = []
    for q in RATIOS:
        row = dict(ratio=q)
        for lab, kw in (("none", {}), ("v0_drift_5pct", dict(v0_scale=1.05)), ("angle_one_15", dict(dang=(0, 15, 0)))):
            Vt = [np.full(N, v) for v in fault_amplitudes(V0, T, phi, **kw)]
            Vm = [v + rng.normal(0, se, N) for v in Vt]
            row[lab] = float(np.mean(np.abs(fr.z_closure(*Vm, q * se)) > 2))
        res.append(row)
    mis[key] = res
out["misstated"] = mis

# ---------------------------------------------------------------- D (departures from the noise model, case configurations)
dep = {}
for key in ("BRT", "IBR", "ICC"):
    V0, T, phi, th = case_truth(key)
    Vt = [np.full(N, v) for v in fault_amplitudes(V0, T, phi)]
    sig_dev = 0.10; se = sig_dev / np.sqrt(K)
    m_true = V0 / T
    rows = {}
    scen = {
        "nominal": dict(),
        "unequal_variance": dict(sig_run=sig_dev * np.array([v[0] for v in Vt]) / np.sqrt(np.mean([v[0] ** 2 for v in Vt]))),
        "correlated_runs_0.5": dict(corr=0.5),
    }
    for name, kw in scen.items():
        reads = readings(Vt, sig_dev, N, **kw)
        Vm = list(reads.mean(1))
        z = fr.z_closure(*Vm, se)
        a = fr.correction(*Vm, mt=1.0)
        um, ut = fr.predicted_uncertainty(*Vm, se)
        et = np.abs(circ(a["theta"] - th))
        rows[name] = dict(fa=float(np.mean(np.abs(z) > 2)), cov_m=float(np.mean(np.abs(a["m"] - m_true) <= 1.96 * um * a["m"])),
                          cov_t=float(np.mean(et <= 1.96 * ut)))
    # common gain error of +5 % on every reading
    reads = readings([v * 1.05 for v in Vt], sig_dev, N)
    Vm = list(reads.mean(1))
    a = fr.correction(*Vm, mt=1.0)
    rows["common_gain_5pct"] = dict(fa=float(np.mean(np.abs(fr.z_closure(*Vm, se)) > 2)),
                                    mean_em=float(np.mean(100 * (a["m"] - m_true) / m_true)),
                                    mean_et=float(np.mean(circ(a["theta"] - th))))
    dep[key] = rows
out["departures"] = dep

# ---------------------------------------------------------------- A' (estimated sigma at case configurations, faults)
est = {}
for key in ("BRT", "IBR", "ICC"):
    V0, T, phi, th = case_truth(key)
    r_ = {}
    for lab, kw in (("none", {}), ("v0_drift_5pct", dict(v0_scale=1.05)), ("angle_one_15", dict(dang=(0, 15, 0)))):
        Vt = [np.full(N, v) for v in fault_amplitudes(V0, T, phi, **kw)]
        reads = readings(Vt, 0.10, N)
        Vm, se_hat = pooled(reads)
        r_[lab] = dict(known_2=float(np.mean(np.abs(fr.z_closure(*Vm, 0.10 / np.sqrt(K))) > 2)),
                       est_t=float(np.mean(np.abs(fr.z_closure(*Vm, se_hat)) > T_NU)))
    est[key] = r_
out["estimated_cases"] = est

# ---------------------------------------------------------------- F (z of the published cases vs assumed sigma_dev)
sd = np.round(np.geomspace(0.02, 0.5, 60), 5)
out["z_vs_sigma"] = dict(sigma_dev=sd.tolist(), **{k: (CASES[k]["g"] / (fr.sd_closure(*CASES[k]["V"], 1.0) * sd / np.sqrt(K))).tolist() for k in ("BRT", "IBR", "ICC")})

(ROOT / "results/robustness.json").write_text(json.dumps(out, indent=1))

# ---------------------------------------------------------------- summary
g = gridA
gA_ = gridA
print(f"t(0.975,{NU}) = {T_NU:.3f}")
print("A: FA known/2 {:.1f}-{:.1f}% | est/2 {:.1f}-{:.1f}% | est/t {:.1f}-{:.1f}% | est/t assessable {:.1f}-{:.1f}% | not assessable up to {:.1f}%".format(
    *[100 * f(x[k] for x in g) for k in ("fa_known_2", "fa_est_2", "fa_est_t") for f in (min, max)],
    100 * np.nanmin([x["fa_est_t_assessable"] for x in g]), 100 * np.nanmax([x["fa_est_t_assessable"] for x in g]),
    100 * max(x["frac_not_assessable"] for x in g)))
print("E: coverage known m {:.1f}-{:.1f} t {:.1f}-{:.1f} joint {:.1f}-{:.1f} | est m {:.1f}-{:.1f} t {:.1f}-{:.1f} joint {:.1f}-{:.1f}".format(
    *[100 * f(x[k] for x in g) for k in ("cov_m", "cov_t", "cov_joint", "cov_m_est", "cov_t_est", "cov_joint_est") for f in (min, max)]))
for x in g:
    if x["s"] in (10, 35) and x["r"] in (0.75, 1.0, 2.0, 4.0):
        print(f"   r={x['r']} s={x['s']}: cov m/t/joint {100*x['cov_m']:.1f}/{100*x['cov_t']:.1f}/{100*x['cov_joint']:.1f}  est {100*x['cov_m_est']:.1f}/{100*x['cov_t_est']:.1f}/{100*x['cov_joint_est']:.1f}  notassess {100*x['frac_not_assessable']:.1f}%")
for r, d in phase.items():
    fa = np.array(d["fa"]); mv = np.array(d["minV_over_sigma"]); ok = mv >= MINV_FACTOR
    print(f"B r={r}: FA over phi {100*fa.min():.1f}-{100*fa.max():.1f}%; where min V >= {MINV_FACTOR:g} sigma: {100*fa[ok].min():.1f}-{100*fa[ok].max():.1f}%; phi share below: {100*(1-ok.mean()):.0f}%")
for k, rows in mis.items():
    print("C", k, " | ".join(f"q={r['ratio']}: FA {100*r['none']:.1f} V0 {100*r['v0_drift_5pct']:.0f} ang {100*r['angle_one_15']:.0f}" for r in rows))
for k, rows in dep.items():
    print("D", k, rows)
for k, rows in est.items():
    print("A'", k, rows)
for k in ("BRT", "IBR", "ICC"):
    z = np.array(out["z_vs_sigma"][k]); print("F", k, "z at 0.05/0.10/0.20:", [round(float(np.interp(v, sd, z)), 2) for v in (0.05, 0.10, 0.20)])

print("gate sensitivity (estimated sigma, t rule): factor -> max share not assessable | FA given assessable over cells with >=80% assessable | over all cells")
for cfac in GATE_FACTORS:
    k = str(cfac)
    na = [x["gate"][k]["est"]["not_assessable"] for x in gA_]
    fa_ok = [x["gate"][k]["est"]["fa_given_assessable"] for x in gA_ if x["gate"][k]["est"]["share_assessable"] >= 0.8]
    fa_all = [x["gate"][k]["est"]["fa_given_assessable"] for x in gA_ if np.isfinite(x["gate"][k]["est"]["fa_given_assessable"])]
    print(f"  c={cfac:>4}: not assessable up to {100*max(na):.1f}% | FA|assessable {100*min(fa_ok):.1f}-{100*max(fa_ok):.1f}% (n={len(fa_ok)}) | all {100*min(fa_all):.1f}-{100*max(fa_all):.1f}%")
