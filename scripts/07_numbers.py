"""Collect every number quoted in the manuscript text into results/numbers.json
so that the text is filled from computed results only (no hand-typed values)."""
import json, pathlib, numpy as np, fourrun as fr
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = lambda f: json.loads((ROOT / "results" / f).read_text())
cases, A, B, F, PM = R("cases.json"), R("partA.json"), R("partB.json"), R("faults.json"), R("passmap.json")
N = {}


def f(x, d=1):
    return f"{x:,.{d}f}"


# ---- crossover of the first-order variances (worst and best phase)
N["r_star_worst"] = f(brentq(lambda r: 4 * r**4 - r**2 - 2 * r - 5, 1.0, 2.0), 2)
N["r_star_best"] = f(brentq(lambda r: 4 * r**4 - r**2 + 2 * r - 5, 0.5, 1.5), 2)

# ---- cases
for k, c in cases.items():
    p = k.lower()
    N[f"{p}_rho"] = f(c["rho"], 3); N[f"{p}_absrho"] = f(abs(c["rho"] - 1), 3)
    N[f"{p}_Tsum"] = f(c["Tsum"], 3); N[f"{p}_Txy"] = f(c["Txy"], 3)
    N[f"{p}_r"] = f(c["V0_over_T"], 2)
    N[f"{p}_s010"] = f(c["Txy"] / (0.10 / np.sqrt(3)), 0); N[f"{p}_s020"] = f(c["Txy"] / (0.20 / np.sqrt(3)), 0)
    dm = {"BRT": 3, "IBR": 2, "ICC": 1}[k]
    N[f"{p}_m"] = f(c["m_xy"], dm); N[f"{p}_th"] = f(c["theta"], 2)
    N[f"{p}_msum"] = f(c["m_sum"], dm); N[f"{p}_estdiff"] = f(c["estimator_diff_pct"], 1)
    N[f"{p}_mg"] = f(c["m_graph"], dm); N[f"{p}_thg"] = f(c["theta_graph"], 2)
    N[f"{p}_dm"] = f(c["dm_methods_pct"], 2); N[f"{p}_dth"] = f(c["dtheta_methods"], 2)
    N[f"{p}_spread"] = f(c["spread"], 2)
    N[f"{p}_z010"] = f(c["z@0.10"], 2); N[f"{p}_z020"] = f(c["z@0.20"], 2)
    N[f"{p}_sdrho010"] = f(c["sd_rho@0.10"], 3); N[f"{p}_sdrho020"] = f(c["sd_rho@0.20"], 3)
    N[f"{p}_sigz2"] = f(c["sigma_dev_for_z2"], 3)
    N[f"{p}_red"] = f(c["reduction_pct"], 1)
    N[f"{p}_uper"] = f(c["uper_G63"], 1 if c["uper_G63"] < 1000 else 0)
    N[f"{p}_mtuper"] = f(c["mt_from_uper"], 2 if c["mt_from_uper"] < 10 else 1)
    N[f"{p}_usedU"] = f(c["used_trial_unbalance"], 0); N[f"{p}_ratioU"] = f(c["used_over_uper"], 2)
    N[f"{p}_Geq"] = f(c["equivalent_G"], 1)
    N[f"{p}_V0"] = f(c["V"][0], 2 if k == "IBR" else 1); N[f"{p}_Va"] = f(c["V_after"], 2)
    rowsB = B["cases"][k]["rows"]
    for r in rowsB:
        t = f"{int(round(r['sig_dev']*100)):03d}"
        N[f"{p}_B{t}_p95m"] = f(r["p95_abs_m"], 1); N[f"{p}_B{t}_p95t"] = f(r["p95_abs_t"], 1)
        N[f"{p}_B{t}_U95m"] = f(r["pred_U95_m"], 1); N[f"{p}_B{t}_U95t"] = f(r["pred_U95_t"], 1)
        N[f"{p}_B{t}_pass"] = f(100 * r["pass_rate"], 1); N[f"{p}_B{t}_t2neg"] = f(100 * r["T2sum_nonpos"], 1)
        N[f"{p}_B{t}_skew"] = f(r["skew_m"], 2)
        N[f"{p}_B{t}_gp95m"] = f(r["p95_abs_m_graph"], 1); N[f"{p}_B{t}_gp95t"] = f(r["p95_abs_t_graph"], 1)
    for sd in ("0.10", "0.20"):
        t = sd.replace(".", "")
        for fn, r in F["results"][k][sd].items():
            N[f"{p}_F{t}_{fn}_det"] = f(100 * r["detect"], 0)
            N[f"{p}_F{t}_{fn}_bm"] = f(r["mean_em"], 1); N[f"{p}_F{t}_{fn}_bt"] = f(r["mean_et"], 1)
N["icc_installed"] = f(cases["ICC"]["installed_angle_deg"], 1)
N["icc_installed_diff"] = f(cases["ICC"]["installed_minus_computed"], 2)
N["icc_inst_m"] = f(cases["ICC"]["installed_mass_g"], 0)
N["icc_inst_vs_an"] = f(abs(cases["ICC"]["installed_vs_analytical_pct"]), 1)
N["icc_inst_vs_gr"] = f(abs(cases["ICC"]["installed_vs_graph_pct"]), 1)
N["icc_inst_gr_ang"] = f(cases["ICC"]["installed_minus_graph_angle"], 2)

# ---- Part A
cells = A["cells"]
z = [100 * c["screening"]["z2"]["flag_rate"] for c in cells]
fx = [100 * c["screening"]["fixed_0.08"]["flag_rate"] for c in cells]
N["A_z_min"], N["A_z_max"] = f(min(z), 1), f(max(z), 1)
N["A_f08_min"], N["A_f08_max"] = f(min(fx), 1), f(max(fx), 1)
N["A_N"] = f(A["N"], 0); N["A_ncells"] = str(len(cells)); N["A_seed"] = str(A["seed"])
get = lambda r, s: next(c for c in cells if c["r"] == r and c["s"] == s)
for (r, s) in [(0.75, 35), (1.0, 35), (1.5, 35), (2.0, 35), (4.0, 35), (4.0, 20), (1.5, 20), (0.75, 10), (2.0, 10), (4.0, 10)]:
    c = get(r, s); key = f"A_{str(r).replace('.', 'p')}_{s}"
    for e in ("xy", "sum", "graph"):
        N[f"{key}_{e}"] = f(c[e]["rmse"], 1)
    N[f"{key}_pass"] = f(100 * c["pass_xy"], 1)
    N[f"{key}_t2neg"] = f(100 * c["T2sum_nonpos"], 1)
    N[f"{key}_failacc_z"] = f(100 * c["screening"]["z2"]["fail_given_accepted"], 0)
    N[f"{key}_f08"] = f(100 * c["screening"]["fixed_0.08"]["flag_rate"], 1)
N["A_max_t2neg"] = f(100 * max(c["T2sum_nonpos"] for c in cells), 1)
# s95 from pass map
Rr, Ss, P = np.array(PM["r"]), np.array(PM["s"]), np.array(PM["pass_rate"])
def s95(r):
    j = int(np.argmin(abs(Rr - r))); col = P[:, j]; i = int(np.argmax(col >= 0.95))
    return float(np.exp(np.interp(0.95, [col[i - 1], col[i]], np.log([Ss[i - 1], Ss[i]]))))
for r in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
    N[f"s95_{str(r).replace('.', 'p')}"] = f(s95(r), 0)
N["B_N"] = f(B["N"], 0); N["B_seed"] = str(B["seed"]); N["F_seed"] = str(F["seed"])

# ---- robustness (Section 6.2.5)
Rb = R("robustness.json")
gA = Rb["gridA"]
rng_ = lambda key, d=1: (f(100 * min(x[key] for x in gA), d), f(100 * max(x[key] for x in gA), d))
N["R_fa_known_min"], N["R_fa_known_max"] = rng_("fa_known_2")
N["R_fa_est2_min"], N["R_fa_est2_max"] = rng_("fa_est_2")
N["R_fa_estt_min"], N["R_fa_estt_max"] = rng_("fa_est_t")
N["R_cov_m_min"], N["R_cov_m_max"] = rng_("cov_m")
N["R_cov_t_min"], N["R_cov_t_max"] = rng_("cov_t")
N["R_cov_j_min"], N["R_cov_j_max"] = rng_("cov_joint")
N["R_covest_m_min"], N["R_covest_m_max"] = rng_("cov_m_est")
N["R_covest_j_min"], N["R_covest_j_max"] = rng_("cov_joint_est")
N["R_notass_max"] = f(100 * max(x["frac_not_assessable"] for x in gA), 0)
N["R_t_nu"] = f(Rb["t_nu"], 3); N["R_seed"] = str(Rb["seed"])
ph = Rb["phase"]
ok = [v for r, d in ph.items() if float(r) != 1.0 for v in d["fa"]]
N["R_phase_min"], N["R_phase_max"] = f(100 * min(ok), 1), f(100 * max(ok), 1)
d1 = ph["1.0"]; N["R_phase_r1_min"] = f(100 * min(d1["fa"]), 1)
mis = Rb["misstated"]["BRT"]
N["R_mis05_fa"] = f(100 * next(r for r in mis if r["ratio"] == 0.5)["none"], 0)
N["R_mis2_fa"] = f(100 * next(r for r in mis if r["ratio"] == 2.0)["none"], 1)
for k in ("BRT", "IBR", "ICC"):
    p_ = k.lower()
    N[f"{p_}_mis2_v0"] = f(100 * next(r for r in Rb["misstated"][k] if r["ratio"] == 2.0)["v0_drift_5pct"], 0)
    N[f"{p_}_est_v0"] = f(100 * Rb["estimated_cases"][k]["v0_drift_5pct"]["est_t"], 0)
    N[f"{p_}_est_ang"] = f(100 * Rb["estimated_cases"][k]["angle_one_15"]["est_t"], 0)
    dep = Rb["departures"][k]
    N[f"{p_}_uv_fa"] = f(100 * dep["unequal_variance"]["fa"], 1); N[f"{p_}_uv_covt"] = f(100 * dep["unequal_variance"]["cov_t"], 1)
    N[f"{p_}_uv_covm"] = f(100 * dep["unequal_variance"]["cov_m"], 1)
    N[f"{p_}_cr_fa"] = f(100 * dep["correlated_runs_0.5"]["fa"], 1); N[f"{p_}_cr_covm"] = f(100 * dep["correlated_runs_0.5"]["cov_m"], 1)
    N[f"{p_}_cg_em"] = f(dep["common_gain_5pct"]["mean_em"], 2)
    # Eq. (11) at the fitted (T, phi) of each case vs gradient form at the measured amplitudes
    c = cases[k]; phi_ = (c["theta"] - 180.0) % 360.0
    N[f"{p_}_sdrho_eq11"] = f(float(fr.theory(c["V"][0], c["Txy"], phi_, 0.10 / np.sqrt(3))["sd_rho"]), 3)
# sigma_dev below which the ICC readings would be flagged with an estimated sigma (t threshold)
N["icc_sigt"] = f(cases["ICC"]["sigma_dev_for_z2"] * 2.0 / Rb["t_nu"], 3)

# ---- amplitude condition (final procedure, estimated sigma, t rule)
G = lambda c: [x["gate"][c]["est"] for x in gA]
g5 = G("5.0")
N["R_na_max"] = f(100 * max(x["not_assessable"] for x in g5), 1)
N["R_na_r1_s35"] = f(100 * next(x["gate"]["5.0"]["est"]["not_assessable"] for x in gA if x["r"] == 1.0 and x["s"] == 35), 1)
N["R_fl_min"], N["R_fl_max"] = f(100 * min(x["flagged"] for x in g5), 1), f(100 * max(x["flagged"] for x in g5), 1)
fa5 = [x["fa_given_assessable"] for x in g5]
N["R_fa_ass5_min"], N["R_fa_ass5_max"] = f(100 * min(fa5), 1), f(100 * max(fa5), 1)
for c in ("8.0", "10.0", "15.0"):
    N[f"R_fa_ass{c.split('.')[0]}_max"] = f(100 * max(x["fa_given_assessable"] for x in G(c) if np.isfinite(x["fa_given_assessable"])), 1)
fa80 = [x["fa_given_assessable"] for c in ("5.0", "8.0", "10.0", "15.0") for x in G(c) if x["share_assessable"] >= 0.8]
N["R_fa80_min"], N["R_fa80_max"] = f(100 * min(fa80), 1), f(100 * max(fa80), 1)
N["icc_inst_vs_gr2"] = f(abs(cases["ICC"]["installed_vs_graph_pct"]), 2)
N["R_minV_brt"] = f(min(cases["BRT"]["V"]) / (0.10 / np.sqrt(3)), 0)

(ROOT / "results/numbers.json").write_text(json.dumps(N, indent=0, ensure_ascii=False))
print(len(N), "numbers;", {k: N[k] for k in ["r_star_worst", "r_star_best", "A_z_min", "A_z_max", "A_f08_max", "s95_1p5", "s95_4p0", "icc_estdiff", "icc_z010", "icc_sigz2"]})
