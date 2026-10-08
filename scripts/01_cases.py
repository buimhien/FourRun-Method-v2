"""Recompute every quantity reported for the three application case studies.
Output: results/cases.json"""
import json, pathlib, numpy as np, fourrun as fr

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "data/cases.json").read_text())
K = 3
SIG_DEV = [0.10, 0.20]


def g6p3_uper(m_kg, n_rpm, G=6.3):
    # U_per [g mm] = 1000 * G[mm/s] * m[kg] / omega[rad/s]
    return 1000.0 * G * m_kg * 60.0 / (2.0 * np.pi * n_rpm)


def circ_diff(a, b):
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


out = {}
for key, c in C.items():
    if key.startswith("_"):
        continue
    V = c["V"]
    a = fr.correction(*V, mt=c["mt_g"], rt=c["rt_mm"])
    s = fr.correction(*V, mt=c["mt_g"], rt=c["rt_mm"], estimator="sum")
    g = fr.graphical_solve(*V, mt=c["mt_g"], rt=c["rt_mm"])
    uper = g6p3_uper(c["mass_kg"], c["speed_rpm"])
    used_U = c["mt_g"] * c["rt_mm"]
    rec = dict(
        V=V, V_after=c["V_after"],
        T2sum=float(a["T2sum"]), Tsum=float(a["Tsum"]), Txy=float(a["Txy"]),
        g=float(a["g"]), rho=float(a["rho"]), V0_over_T=V[0] / float(a["Txy"]),
        m_xy=float(a["m"]), theta=float(a["theta"]), m_sum=float(s["m"]),
        estimator_diff_pct=100.0 * (float(s["m"]) - float(a["m"])) / float(a["m"]),
        m_graph=float(g["m"]), theta_graph=float(g["theta"]), T_graph=float(g["T"]),
        spread=float(g["spread"]), n_intersecting=int(g["n_intersecting"]),
        dm_methods_pct=100.0 * abs(float(g["m"]) - float(a["m"])) / float(a["m"]),
        dtheta_methods=circ_diff(float(g["theta"]), float(a["theta"])),
        reduction_pct=100.0 * (V[0] - c["V_after"]) / V[0],
        uper_G63=uper, mt_from_uper=uper / c["rt_mm"], used_trial_unbalance=used_U,
        used_over_uper=used_U / uper,
        equivalent_G=6.3 * used_U / uper,
        pair_intersects=[bool(fr.pair_intersects(V[0], V[i], V[j])) for i, j in ((1, 2), (1, 3), (2, 3))],
    )
    for sd in SIG_DEV:
        se = sd / np.sqrt(K)
        rec[f"sd_rho@{sd:.2f}"] = float(fr.sd_rho(*V, se))
        rec[f"z@{sd:.2f}"] = float(fr.z_closure(*V, se))
        rec[f"z_rho@{sd:.2f}"] = (rec["rho"] - 1.0) / rec[f"sd_rho@{sd:.2f}"]
    # sigma_dev at which |z| would reach 2
    rec["sigma_dev_for_z2"] = float(abs(a["g"]) / (2.0 * fr.sd_closure(*V, 1.0)) * np.sqrt(K))
    if "installed_angle_deg" in c:
        rec["installed_angle_deg"] = c["installed_angle_deg"]
        rec["installed_minus_computed"] = circ_diff(c["installed_angle_deg"], rec["theta"])
        rec["installed_minus_graph_angle"] = circ_diff(c["installed_angle_deg"], rec["theta_graph"])
    if "installed_mass_g" in c:
        rec["installed_mass_g"] = c["installed_mass_g"]
        rec["installed_vs_analytical_pct"] = 100.0 * (c["installed_mass_g"] - rec["m_xy"]) / rec["m_xy"]
        rec["installed_vs_graph_pct"] = 100.0 * (c["installed_mass_g"] - rec["m_graph"]) / rec["m_graph"]
    out[key] = rec

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/cases.json").write_text(json.dumps(out, indent=2))
for k, r in out.items():
    print(f"{k}: rho={r['rho']:.4f} Tsum={r['Tsum']:.4f} Txy={r['Txy']:.4f} V0/T={r['V0_over_T']:.3f} "
          f"m_xy={r['m_xy']:.4f} th={r['theta']:.2f} | m_sum={r['m_sum']:.2f} ({r['estimator_diff_pct']:+.2f}%) | "
          f"graph m={r['m_graph']:.4f} th={r['theta_graph']:.2f} dm={r['dm_methods_pct']:.2f}% dth={r['dtheta_methods']:.2f} "
          f"spread={r['spread']:.3f} nint={r['n_intersecting']}")
    print(f"     z@0.10={r['z@0.10']:+.2f} z_rho@0.10={r['z_rho@0.10']:+.2f} sd_rho@0.10={r['sd_rho@0.10']:.4f} "
          f"z@0.20={r['z@0.20']:+.2f} sd_rho@0.20={r['sd_rho@0.20']:.4f} sigma_dev(|z|=2)={r['sigma_dev_for_z2']:.3f} "
          f"red={r['reduction_pct']:.2f}% Uper={r['uper_G63']:.1f} mt_uper={r['mt_from_uper']:.3f} "
          f"used/uper={r['used_over_uper']:.3f} Geq={r['equivalent_G']:.2f}")
