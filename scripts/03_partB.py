"""Part B: sensitivity of the published case results to an assumed amplitude
noise model (not an accuracy assessment against a true value).
Reference values are the analytical results computed from the published
mean amplitudes. Output: results/partB.json"""
import json, pathlib, numpy as np, fourrun as fr

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = {k: v for k, v in json.loads((ROOT / "data/cases.json").read_text()).items() if not k.startswith("_")}
SEED, N, K = 20261002, 50_000, 3
SIG_DEV = [0.10, 0.20, 0.50, 1.00]


def circ(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    rng = np.random.default_rng(SEED)
    out = dict(seed=SEED, N=N, k=K, sig_dev=SIG_DEV, cases={})
    for key, c in C.items():
        V = np.array(c["V"]); ref = fr.correction(*V, mt=c["mt_g"], rt=c["rt_mm"])
        m0, t0 = float(ref["m"]), float(ref["theta"])
        rows = []
        for sd in SIG_DEV:
            se = sd / np.sqrt(K)
            Vn = [V[i] + rng.normal(0.0, se, N) for i in range(4)]
            a = fr.correction(*Vn, mt=c["mt_g"], rt=c["rt_mm"])
            g = fr.graphical_solve(*Vn, mt=c["mt_g"], rt=c["rt_mm"])
            em = 100.0 * (a["m"] - m0) / m0; et = circ(a["theta"] - t0)
            emg = 100.0 * (g["m"] - m0) / m0; etg = circ(g["theta"] - t0)
            ok = (np.abs(em) <= 10) & (np.abs(et) <= 10)
            z = fr.z_closure(*Vn, se)
            um, ut = fr.predicted_uncertainty(*V, se)
            rows.append(dict(
                sig_dev=sd, sig_eff=se,
                abs_m_mean=float(np.abs(em).mean()), abs_m_sd=float(np.abs(em).std()),
                abs_t_mean=float(np.abs(et).mean()), abs_t_sd=float(np.abs(et).std()),
                m_q025=float(np.percentile(em, 2.5)), m_q975=float(np.percentile(em, 97.5)),
                t_q025=float(np.percentile(et, 2.5)), t_q975=float(np.percentile(et, 97.5)),
                p95_abs_m=float(np.percentile(np.abs(em), 95)), p95_abs_t=float(np.percentile(np.abs(et), 95)),
                p95_abs_m_graph=float(np.percentile(np.abs(emg), 95)), p95_abs_t_graph=float(np.percentile(np.abs(etg), 95)),
                pass_rate=float(ok.mean()),
                T2sum_nonpos=float(np.mean(a["T2sum"] <= 0)),
                graph_nonintersect=float(np.mean(g["n_intersecting"] < 3)),
                flag_rate_z2=float(np.mean(np.abs(z) > 2)),
                rho_sd=float(np.nanstd(a["rho"])),
                pred_u_m=float(100 * um), pred_u_t=float(ut),
                pred_U95_m=float(196 * um), pred_U95_t=float(1.96 * ut),
                skew_m=float(((em - em.mean())**3).mean() / em.std()**3),
            ))
        out["cases"][key] = dict(m_ref=m0, theta_ref=t0, rows=rows)
    (ROOT / "results/partB.json").write_text(json.dumps(out, indent=1))
    for k, v in out["cases"].items():
        for r in v["rows"]:
            print(f"{k} {r['sig_dev']:.2f}: |dm| {r['abs_m_mean']:5.1f}±{r['abs_m_sd']:5.1f}  |dth| {r['abs_t_mean']:5.1f}±{r['abs_t_sd']:5.1f}"
                  f"  P95 {r['p95_abs_m']:5.1f}%/{r['p95_abs_t']:4.1f}°  pred U95 {r['pred_U95_m']:5.1f}%/{r['pred_U95_t']:4.1f}°"
                  f"  pass {100*r['pass_rate']:5.1f}%  T2s<=0 {100*r['T2sum_nonpos']:4.1f}%  flag {100*r['flag_rate_z2']:4.1f}%"
                  f"  graphP95 {r['p95_abs_m_graph']:5.1f}%/{r['p95_abs_t_graph']:4.1f}°  q[{r['m_q025']:6.1f},{r['m_q975']:6.1f}] skew {r['skew_m']:.2f}")


if __name__ == "__main__":
    main()
