"""Student t critical values: JavaScript (incomplete-beta implementation) against scipy."""
import json, subprocess, numpy as np
from scipy import stats
nus = list(range(1, 201)) + [250, 500, 1000, 10000]
js = subprocess.run(["node", "-e", "const F=require('./js/fourrun.js');console.log(JSON.stringify(JSON.parse(process.argv[1]).map(n=>F.criticalValue(n))))",
                     json.dumps(nus)], capture_output=True, text=True, check=True).stdout
J = np.array(json.loads(js)); P = np.array([stats.t.ppf(0.975, n) for n in nus])
err = np.max(np.abs(J - P))
print(f"Student t(0.975, nu) for nu = 1..200, 250, 500, 1000, 10000: max |JS - scipy| = {err:.2e}", "PASS" if err < 1e-9 else "FAIL")
