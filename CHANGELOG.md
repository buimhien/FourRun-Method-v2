# Changelog

## v3.0

Web tool (`index.html`)
- All computations moved to `js/fourrun.js`, the JavaScript reference implementation that agrees with the Python package `fourrun` to within 1e-9 (`tests/parity.py`).
- Closure check: standardised closure residual z = g/u(g) with a known noise level (limit ±2) or with the noise estimated from k ≥ 2 readings per run (limit t(0.975, ν), ν = 4(k − 1)).
- Amplitude condition: data sets that are not flagged but have an amplitude below 5σ are reported as "not assessable".
- Predicted expanded uncertainties U95 of the correction mass and angle (amplitude noise only).
- ρ is reported as undefined when T²sum < 0 instead of stopping the calculation; T²sum = 0 gives ρ = 0 when T_xy > 0. When T_xy ≈ 0 or V0 ≤ 0, ρ and the correction are undefined and a status code is returned.
- Graphical construction: for a circle pair that does not intersect, the foot of the radical axis is used; the solution is the unweighted mean of the three retained points.
- Inputs for the trial-weight radius and the correction radius.
- The ISO zone verdict was removed from the report, because the zones of ISO 20816-3 apply to broadband r.m.s. velocity on machines above 15 kW and the tool works with 1× amplitudes.

Repository
- Python package `fourrun`, unit tests, JavaScript/Python parity tests, the input data of the three application cases, and the scripts (with fixed random seeds) that generate every table and figure of the paper.
- MIT licence.

## v2.0

Version used for the application cases of the paper: analytical and graphical solutions side by side, circle construction and ρ.

## v1.0

First public version of the web tool.
