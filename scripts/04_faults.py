"""Detectability of systematic measurement faults by the closure test |z| > 2.
Noise-free amplitudes are generated from each case's analytical solution
(V0, T, phi); a systematic fault is injected; random noise is added.
Output: results/faults.json"""
import json, pathlib, numpy as np, fourrun as fr

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "results/cases.json").read_text())
SEED, N, K = 20261003, 50_000, 3
SIG_DEV = [0.10, 0.20]


def circ(a):
    return (a + 180.0) % 360.0 - 180.0


def amplitudes(V0, T, phi, dang=(0, 0, 0), gain=(1, 1, 1), v0_scale=1.0):
    """Model with faults: trial weight actually at alpha_i + dang_i; trial
    effect multiplied by gain_i in run i; V0 reading scaled by v0_scale
    (e.g. a drift between the reference run and the trial runs)."""
    al = np.deg2rad(np.array([0, 120, 240]) + np.array(dang))
    p = np.deg2rad(phi)
    Vt = [np.sqrt(V0**2 + (gi * T)**2 + 2 * V0 * gi * T * np.cos(p - a)) for a, gi in zip(al, gain)]
    return [V0 * v0_scale] + Vt


FAULTS = {
    "none": {},
    "angle_one_5": dict(dang=(0, 5, 0)),
    "angle_one_10": dict(dang=(0, 10, 0)),
    "angle_one_15": dict(dang=(0, 15, 0)),
    "angle_all_20": dict(dang=(20, 20, 20)),
    "gain_one_5pct": dict(gain=(1, 1, 1.05)),
    "gain_one_10pct": dict(gain=(1, 1, 1.10)),
    "v0_drift_5pct": dict(v0_scale=1.05),
}


def main():
    rng = np.random.default_rng(SEED)
    out = dict(seed=SEED, N=N, k=K, faults=list(FAULTS), results={})
    for key, c in CASES.items():
        V0, T = c["V"][0], c["Txy"]
        phi = (c["theta"] - 180.0) % 360.0
        m_true = V0 / T          # relative scale (mt = 1)
        th_true = c["theta"]
        out["results"][key] = {}
        for sd in SIG_DEV:
            se = sd / np.sqrt(K)
            res = {}
            for fname, kw in FAULTS.items():
                Vt = amplitudes(V0, T, phi, **kw)
                Vn = [v + rng.normal(0, se, N) for v in Vt]
                a = fr.correction(*Vn, mt=1.0)
                z = fr.z_closure(*Vn, se)
                em = 100 * (a["m"] - m_true) / m_true
                et = circ(a["theta"] - th_true)
                ok = (np.abs(em) <= 10) & (np.abs(et) <= 10)
                flag = np.abs(z) > 2
                res[fname] = dict(detect=float(flag.mean()),
                                  mean_em=float(em.mean()), mean_et=float(et.mean()),
                                  pass_rate=float(ok.mean()),
                                  pass_given_accepted=float(ok[~flag].mean()) if (~flag).any() else float("nan"))
            out["results"][key][f"{sd:.2f}"] = res
    (ROOT / "results/faults.json").write_text(json.dumps(out, indent=1))
    for key, d in out["results"].items():
        for sd, res in d.items():
            print(f"{key} sd={sd}: " + " | ".join(
                f"{f}: det {100*r['detect']:4.0f}% bias {r['mean_em']:+5.1f}%/{r['mean_et']:+5.1f}° pass {100*r['pass_rate']:4.0f}%"
                for f, r in res.items()))


if __name__ == "__main__":
    main()
