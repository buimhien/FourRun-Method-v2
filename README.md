# Four-run Method: reference implementation and reproducibility package

Code and data for the paper *Uncertainty and Data Consistency Assessment for Phase-Free
Single-Plane Rigid Rotor Balancing: Monte Carlo Analysis and Three Application Case Studies*
(M.-H. Bui, D.-H. Dinh, P.-V. Dang; The University of Danang – University of Science and Technology).

## Contents

| Path | What it is |
|---|---|
| `index.html` | Web tool v3.0 (Digital Balancing Assistant, served at https://buimhien.github.io/Four-run-Method/); all computations are done by `js/fourrun.js` |
| `fourrun/` | Python package: estimators (Eqs. 3–6), closure residual and z test (Eqs. 10–12) with known or pooled noise estimate, screening decision incl. 'not assessable', predicted uncertainty (Eq. 13), graphical solution (Section 4.5) |
| `js/fourrun.js` | JavaScript module with the same functions, the computational core of the web tool |
| `data/cases.json` | Published mean amplitudes and machine data of the three application cases |
| `scripts/01_cases.py` | Recomputes every case quantity (Tables 2, 10, 11) |
| `scripts/02_partA.py` | Part A Monte Carlo, known reference values (Tables 4–5) |
| `scripts/03_partB.py` | Part B Monte Carlo, sensitivity of the published results (Table 9, Fig. 7) |
| `scripts/04_faults.py` | Detectability of systematic faults (Table 6) |
| `scripts/05_passmap.py` | Accuracy map (Fig. 4b) |
| `scripts/08_robustness.py` | Estimated noise, fixed phase, mis-stated noise, model departures, amplitude condition, coverage (Section 6.2.5, Tables 7–8, Fig. 6) |
| `scripts/06_figures.py` | All computed figures |
| `scripts/07_numbers.py` | Collects every number quoted in the text into `results/numbers.json` |
| `tests/` | Unit tests of the derived expressions; JavaScript/Python parity checks (results, screening decisions, Student quantiles) |
| `results/` | Outputs of the scripts (JSON) |
| `figures/` | Figures (PNG). File names follow the order in which the figures were produced; paper numbers: Fig1 → 1, Fig2 → 2, Fig3 → 3, Fig4 → 4, Fig5 → 5, Fig6_robustness → 6, Fig6_partB_sensitivity → 7, Fig9_case_constructions → 8, Fig7_BRT_rig → 9, Fig8_ICC_rotor → 10 |

## Reproduce

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib, pytest
./run_all.sh                          # about 1 minute on a laptop
```

Random seeds are fixed in each script (Part A 20261001, Part B 20261002, faults 20261003, accuracy map 20261004, robustness 20261005).

## Web tool

`index.html` runs in any current desktop or mobile browser without installation; open it locally or use the GitHub Pages address above. Enter the four 1× amplitudes (one averaged value per run, or k ≥ 2 individual readings per run separated by commas or spaces), the trial mass and radii, and either the noise of one reading σ_dev with the number k of readings averaged, or the readings themselves. The tool reports the analytical correction, the graphical construction, ρ, the closure statistic z with the decision (no inconsistency detected / flagged / not assessable / not computable) and the predicted expanded uncertainties U95 of the correction mass and angle. See `CHANGELOG.md` for the differences from earlier versions.

## Minimal use

```python
import fourrun as fr
V = [10.4, 8.7, 9.8, 12.9]                      # V0, V1, V2, V3 in mm/s (ICC case)
c = fr.correction(*V, mt=378, rt=250)           # m_c (g), theta (deg), Txy, rho, g ...
z = fr.z_closure(*V, sigma=0.10 / 3**0.5)       # standardised closure residual
u_m, u_t = fr.predicted_uncertainty(*V, 0.10 / 3**0.5)   # relative u(m_c), u(theta) in degrees
print(c["m"], c["theta"], c["rho"], z, 1.96 * u_m, 1.96 * u_t)

# with k repeated readings per run (shape 4 x k): pooled noise estimate and Student critical value
# means, sigma_hat, nu = fr.pooled_sigma(readings)
# fr.screen(*means, sigma_hat, nu=nu)   # -> 'flagged' / 'no inconsistency detected' / 'not assessable' / 'not computable'
```

`sigma` is the standard deviation of the averaged amplitudes (sigma_dev / sqrt(k)).
Status codes: 0 OK, 1 T2sum < 0 (rho undefined), 2 T2sum = 0 (rho = 0), 3 Txy too small and 4 V0 invalid (rho and the correction undefined).

## Licence

MIT (see `LICENSE`).
