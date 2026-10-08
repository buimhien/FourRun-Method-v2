"""Check that js/fourrun.js and the Python package give identical results."""
import json, subprocess, numpy as np, fourrun as fr
rng = np.random.default_rng(11)
cases = [[6.8,8.7,9.6,3.2],[2.9,3.65,4.58,1.4],[10.4,8.7,9.8,12.9]]
cases += rng.uniform(0.5, 15, size=(300, 4)).tolist()
js = subprocess.run(["node", "-e", """
const F=require('./js/fourrun.js'); const C=JSON.parse(process.argv[1]);
console.log(JSON.stringify(C.map(v=>{const c=F.correction(...v,1);const g=F.graphicalSolve(...v,1);
const pu=F.predictedUncertainty(...v,0.1); const sc=F.screen(...v,0.1,8); const sc32=F.screen(...v,0.1,32);
return [c.Txy,c.T2sum,c.m,c.theta,F.zClosure(...v,0.1),F.sdRho(...v,0.1),g.m,g.theta,g.nIntersecting,pu.uMRel,pu.uThetaDeg,sc.crit,['not computable','flagged','not assessable','no inconsistency detected'].indexOf(sc.decision),sc32.crit,['not computable','flagged','not assessable','no inconsistency detected'].indexOf(sc32.decision)]})));
""", json.dumps(cases)], capture_output=True, text=True, check=True).stdout
J = np.array(json.loads(js), dtype=float)
P = []
for v in cases:
    c = fr.correction(*v, mt=1); g = fr.graphical_solve(*v)
    um, ut = fr.predicted_uncertainty(*v, 0.1); sc = fr.screen(*v, 0.1, nu=8); sc32 = fr.screen(*v, 0.1, nu=32)
    P.append([c["Txy"], c["T2sum"], c["m"], c["theta"], fr.z_closure(*v, 0.1), fr.sd_rho(*v, 0.1),
              g["m"], g["theta"], g["n_intersecting"], um, ut, sc["crit"],
              ["not computable", "flagged", "not assessable", "no inconsistency detected"].index(sc["decision"]), sc32["crit"],
              ["not computable", "flagged", "not assessable", "no inconsistency detected"].index(sc32["decision"])])
P = np.array(P, dtype=float)
ok = np.allclose(np.nan_to_num(J, nan=-1), np.nan_to_num(P, nan=-1), rtol=1e-9, atol=1e-9)
print("JS/Python parity on", len(cases), "inputs:", "PASS" if ok else "FAIL")
if not ok:
    i = np.argwhere(~np.isclose(np.nan_to_num(J, nan=-1), np.nan_to_num(P, nan=-1)))[:5]; print(i, J[i[:,0]], P[i[:,0]])

# degenerate inputs: rho and status codes (T2sum = 0 with Txy > 0; Txy = 0; V0 = 0)
deg = [[3.0, 5.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]]
jd = subprocess.run(["node", "-e", """
const F=require('./js/fourrun.js'); const C=JSON.parse(process.argv[1]);
console.log(JSON.stringify(C.map(v=>{const e=F.estimators(...v); const s=F.screen(...v,0.05);
return [Number.isNaN(e.rho)?null:e.rho, e.status, s.decision]})));
""", json.dumps(deg)], capture_output=True, text=True, check=True).stdout
JD = json.loads(jd)
PD = []
for v in deg:
    with np.errstate(all="ignore"):
        e = fr.estimators(*v); s = fr.screen(*v, 0.05)
    PD.append([None if np.isnan(float(e["rho"])) else float(e["rho"]), int(e["status"]), s["decision"]])
okd = JD == PD and PD[0][0] == 0.0 and PD[1][0] is None and PD[2][0] is None
print("JS/Python parity on degenerate inputs:", "PASS" if okd else "FAIL", PD)
