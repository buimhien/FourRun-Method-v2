#!/usr/bin/env bash
set -e
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
cd "$PYTHONPATH"
python3 -m pytest -q tests
python3 tests/parity.py
python3 tests/parity_t.py
for s in 01_cases 02_partA 03_partB 04_faults 05_passmap 08_robustness 06_figures 07_numbers; do
  echo "== $s"; python scripts/$s.py > /dev/null
done
echo "done: results/ and figures/ regenerated"
