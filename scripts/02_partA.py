"""Part A: accuracy with a known reference (simulation-based verification).

Because relative mass error and angle error depend only on the dimensionless
pair r = V0/T and s = T/sigma (plus the phase phi), a grid over (r, s) with phi
drawn uniformly covers every single-plane configuration. T = 1 is used.
sigma here is the noise of the amplitude values actually used (after
averaging), i.e. sigma_eff.
Output: results/partA.json
"""
import json, pathlib, numpy as np, fourrun as fr

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20261001
N = 50_000
R_GRID = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
S_GRID = [10, 20, 35, 50, 100]
FIXED = [0.08, 0.15, 0.25]
ZCRIT = 2.0
TOL_M, TOL_T = 10.0, 10.0


def circ(a):
    return (a + 180.0) % 360.0 - 180.0


def run_cell(r, s, rng, n=N):
    T, V0, sig = 1.0, r, 1.0 / s
    phi = rng.uniform(0.0, 360.0, n)
    Vt = [np.full(n, V0)] + fr.ideal_amplitudes(V0, T, phi)
    Vn = [v + rng.normal(0.0, sig, n) for v in Vt]
    m_true, th_true = V0 / T, np.mod(phi + 180.0, 360.0)
    a = fr.correction(*Vn, mt=1.0)
    gsol = fr.graphical_solve(*Vn, mt=1.0)
    res = {"r": r, "s": s}
    eth_xy = circ(a["theta"] - th_true)
    eth_g = circ(gsol["theta"] - th_true)
    for lab, m_hat, eth in (("xy", a["m"], eth_xy),
                            ("sum", Vn[0] / a["Tsum"], eth_xy),
                            ("graph", gsol["m"], eth_g)):
        em = 100.0 * (m_hat - m_true) / m_true
        valid = np.isfinite(em)
        emv = em[valid]
        ok = valid & (np.abs(np.nan_to_num(em, nan=1e9)) <= TOL_M) & (np.abs(eth) <= TOL_T)
        res[lab] = dict(bias=float(emv.mean()), sd=float(emv.std()),
                        rmse=float(np.sqrt(np.mean(emv**2))),
                        p95=float(np.percentile(np.abs(emv), 95)),
                        rmse_theta=float(np.sqrt(np.mean(eth[valid]**2))),
                        p95_theta=float(np.percentile(np.abs(eth[valid]), 95)),
                        invalid=float(1.0 - valid.mean()), pass_rate=float(ok.mean()))
    passed = (np.abs(100.0 * (a["m"] - m_true) / m_true) <= TOL_M) & (np.abs(eth_xy) <= TOL_T)
    res["pass_xy"] = float(passed.mean())
    res["T2sum_nonpos"] = float(np.mean(a["T2sum"] <= 0))
    res["graph_nonintersect"] = float(np.mean(gsol["n_intersecting"] < 3))
    rho = a["rho"]
    rules = {f"fixed_{t:.2f}": ~(np.abs(rho - 1.0) <= t) for t in FIXED}   # NaN rho -> flagged
    z = fr.z_closure(*Vn, sig)
    rules["z2"] = np.abs(z) > ZCRIT
    scr = {}
    for name, flag in rules.items():
        acc = ~flag
        scr[name] = dict(flag_rate=float(flag.mean()),
                         fail_given_accepted=float(np.mean(~passed[acc])) if acc.any() else float("nan"),
                         flag_given_pass=float(np.mean(flag[passed])) if passed.any() else float("nan"))
    res["screening"] = scr
    res["sd_rho_mc"] = float(np.nanstd(rho))
    return res


def main():
    rng = np.random.default_rng(SEED)
    cells = [run_cell(r, s, rng) for r in R_GRID for s in S_GRID]
    out = dict(seed=SEED, N=N, r_grid=R_GRID, s_grid=S_GRID, tol=[TOL_M, TOL_T], z_crit=ZCRIT, cells=cells)
    (ROOT / "results/partA.json").write_text(json.dumps(out, indent=1))
    print(f"{'r':>5}{'s':>5} | RMSE m: xy  sum  graph | pass xy | T2s<=0 | false alarm: 0.08  0.15  0.25   z2 | P(fail|acc) 0.08  z2")
    for c in cells:
        sc = c["screening"]
        print(f"{c['r']:5.2f}{c['s']:5d} | {c['xy']['rmse']:6.2f}{c['sum']['rmse']:7.2f}{c['graph']['rmse']:7.2f} |"
              f" {100*c['pass_xy']:6.1f} | {100*c['T2sum_nonpos']:5.2f} |"
              f" {100*sc['fixed_0.08']['flag_rate']:6.1f}{100*sc['fixed_0.15']['flag_rate']:6.1f}"
              f"{100*sc['fixed_0.25']['flag_rate']:6.1f}{100*sc['z2']['flag_rate']:6.1f} |"
              f" {100*sc['fixed_0.08']['fail_given_accepted']:6.1f}{100*sc['z2']['fail_given_accepted']:6.1f}")


if __name__ == "__main__":
    main()
