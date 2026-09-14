#!/usr/bin/env python
"""Run the ensemble: one-time data prep (if missing) then model_runn.py."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PREP_STEPS = [
    (ROOT / "data" / "hierarchical_agents.csv", "auxillary", "create_hierarchical_agents.py"),
    (ROOT / "data" / "demographic_pmfs.pkl", "auxillary", "create_pmf_tables.py"),
]


def run(cwd, script):
    subprocess.run([sys.executable, script], cwd=ROOT / cwd, check=True)


if __name__ == "__main__":
    for output, cwd, script in PREP_STEPS:
        if output.exists():
            print(f"WARNING: skipping {script}, {output.relative_to(ROOT)} already exists")
            continue
        run(cwd, script)

    run("model_src", "model_runn.py")
