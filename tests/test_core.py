"""Tests for the reference implementation. Run: python -m pytest -q tests"""
import numpy as np
import fourrun as fr


def test_noise_free_recovery():
    rng = np.random.default_rng(0)
    for _ in range(200):
        V0 = rng.uniform(0.5, 20); T = rng.uniform(0.2, 15); phi = rng.uniform(0, 360)
        V1, V2, V3 = fr.ideal_amplitudes(V0, T, phi)
        c = fr.correction(V0, V1, V2, V3, mt=1.0)
        assert np.isclose(c["Txy"], T) and np.isclose(c["Tsum"], T)
        assert np.isclose(c["rho"], 1.0) and abs(c["g"]) < 1e-9 * max(1, V0**2)
        assert np.isclose((c["theta"] - (phi + 180)) % 360, 0, atol=1e-7) or \
            np.isclose((c["theta"] - (phi + 180)) % 360, 360, atol=1e-7)
        gsol = fr.graphical_solve(V0, V1, V2, V3)
        assert gsol["n_intersecting"] == 3
        assert np.isclose(gsol["T"], T, rtol=1e-6)
        assert np.isclose(np.cos(np.deg2rad(gsol["theta"] - c["theta"])), 1.0)


def test_t2sum_cancels_in_xy():
    V = dict(V0=10.4, V1=8.7, V2=9.8, V3=12.9)
    e = fr.estimators(**V)
    D = [v**2 - V["V0"]**2 - e["T2sum"] for v in (V["V1"], V["V2"], V["V3"])]
    assert np.isclose(D[0] / (2 * V["V0"]), e["x"])
    assert np.isclose((D[1] - D[2]) / (2 * V["V0"] * np.sqrt(3)), e["y"])


def test_delta_vs_numeric_gradient():
    V = np.array([10.4, 8.7, 9.8, 12.9]); s = 0.1
    h = 1e-6
    def rho(v): return fr.estimators(*v)["rho"]
    g = np.array([(rho(V + h * e) - rho(V - h * e)) / (2 * h) for e in np.eye(4)])
    assert np.isclose(fr.sd_rho(*V, s), s * np.linalg.norm(g), rtol=1e-5)


def test_bias_and_variance_formulas_against_mc():
    rng = np.random.default_rng(1)
    V0, T, phi, s, N = 5.0, 2.0, 40.0, 0.05, 1_000_000
    Vt = [V0] + fr.ideal_amplitudes(V0, T, phi)
    Vn = [v + rng.normal(0, s, N) for v in Vt]
    e = fr.estimators(*Vn)
    th = fr.theory(V0, T, phi, s)
    assert abs(e["T2sum"].mean() - T**2) < 4 * e["T2sum"].std() / np.sqrt(N)
    assert np.isclose((e["Txy"]**2).mean() - T**2, th["bias_T2xy"], rtol=0.05)
    assert np.isclose(np.nanvar(e["Tsum"]), th["var_Tsum"], rtol=0.05)
    assert np.isclose(e["Txy"].var(), th["var_Txy"], rtol=0.05)


def test_status_codes():
    e = fr.estimators(10.0, 1.0, 2.0, 1.5)
    assert e["status"] == fr.T2SUM_NEGATIVE and np.isnan(e["rho"])
    assert np.isfinite(fr.z_closure(10.0, 1.0, 2.0, 1.5, 0.1))
    assert fr.estimators(10.0, 3.0, 3.0, 3.0)["status"] == fr.TXY_TOO_SMALL


def test_closed_forms_match_gradient_forms():
    rng = np.random.default_rng(5)
    for _ in range(100):
        V0 = rng.uniform(1, 12); T = rng.uniform(0.5, 6); phi = rng.uniform(0, 360); s = 0.05
        V = [V0] + [float(v) for v in fr.ideal_amplitudes(V0, T, phi)]
        th = fr.theory(V0, T, phi, s)
        um, ut = fr.predicted_uncertainty(*V, s)
        assert np.isclose(um, th["u_m_rel"], rtol=1e-6)
        assert np.isclose(np.deg2rad(ut), th["u_theta_rad"], rtol=1e-6)
        assert np.isclose(fr.sd_rho(*V, s), th["sd_rho"], rtol=1e-6)
        assert np.isclose(np.sqrt(fr.theory(V0, T, phi, s)["var_Txy"]), np.sqrt(th["var_Txy"]))


def test_pooled_sigma_and_screen():
    rng = np.random.default_rng(3)
    V = np.array([10.4, 8.7, 9.8, 12.9])
    R = V[:, None] + rng.normal(0, 0.1, (4, 3))
    m, s, nu = fr.pooled_sigma(R)
    assert nu == 8 and m.shape == (4,) and 0 < s < 0.2
    assert abs(fr.critical_value(8) - 2.306) < 1e-3 and fr.critical_value() == 2.0
    d = fr.screen(*V, 0.10 / np.sqrt(3))
    assert d["decision"] == "no inconsistency detected" and abs(d["z"] + 1.86) < 0.01
    assert fr.screen(*V, 0.04 / np.sqrt(3))["decision"] == "flagged"
    assert fr.screen(1.0, 1.0, 0.05, 1.7, 0.02)["decision"] in ("not assessable", "flagged")


def test_rho_degenerate_cases():
    # T2sum = 0 exactly with Txy > 0 -> rho = 0 (27/3 - 9 = 0, Txy = 8/3)
    e = fr.estimators(3.0, 5.0, 1.0, 1.0)
    assert float(e["T2sum"]) == 0.0 and float(e["Txy"]) > 0
    assert float(e["rho"]) == 0.0 and int(e["status"]) == fr.T2SUM_ZERO
    # T2sum = 0 and Txy = 0 -> rho undefined, not computable
    e = fr.estimators(1.0, 1.0, 1.0, 1.0)
    assert np.isnan(float(e["rho"])) and int(e["status"]) == fr.TXY_TOO_SMALL
    assert fr.screen(1.0, 1.0, 1.0, 1.0, 0.05)["decision"] == "not computable"
    # V0 <= 0 -> rho undefined
    e = fr.estimators(0.0, 1.0, 2.0, 3.0)
    assert np.isnan(float(e["rho"])) and int(e["status"]) == fr.V0_INVALID
