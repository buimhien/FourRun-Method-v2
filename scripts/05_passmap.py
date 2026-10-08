"""Fine (r, s) map of P(|dm|<=10% and |dtheta|<=10 deg) for the Txy estimator,
used for the planning chart (Fig. 4b). Output: results/passmap.json"""
import json, pathlib, numpy as np, fourrun as fr
ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED, N = 20261004, 20_000
R = np.round(np.geomspace(0.75, 5.0, 17), 4).tolist()
S = np.round(np.geomspace(8, 120, 17), 3).tolist()
rng = np.random.default_rng(SEED)
P = np.zeros((len(S), len(R)))
for j, r in enumerate(R):
    for i, s in enumerate(S):
        phi = rng.uniform(0, 360, N)
        Vt = [np.full(N, r)] + fr.ideal_amplitudes(r, 1.0, phi)
        Vn = [v + rng.normal(0, 1.0 / s, N) for v in Vt]
        a = fr.correction(*Vn, mt=1.0)
        em = 100 * (a["m"] - r) / r
        et = (a["theta"] - (phi + 180) + 180) % 360 - 180
        P[i, j] = np.mean((np.abs(em) <= 10) & (np.abs(et) <= 10))
json.dump(dict(seed=SEED, N=N, r=R, s=S, pass_rate=P.tolist()), open(ROOT / "results/passmap.json", "w"))
# s needed for 95 % at a few r values (linear interpolation in log s)
for j, r in enumerate(R):
    col = P[:, j]; idx = np.argmax(col >= 0.95)
    if col[idx] >= 0.95 and idx > 0:
        s95 = np.exp(np.interp(0.95, [col[idx-1], col[idx]], np.log([S[idx-1], S[idx]])))
        print(f"r={r:.2f}: s95 ≈ {s95:.1f}")
