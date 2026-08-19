#!/usr/bin/env python3
"""Smoketest: how much does the veg-only credit fix move the top-agent multiplier?

Paired design: identical seeds, two arms.
  post  = current code (credit only if partner is veg)
  pre   = pre-fix behaviour (credit whichever neighbour was sampled)
Everything else identical, so the difference is the attribution fix alone.
"""
import os, sys, random, pickle
import numpy as np
from multiprocessing import Pool

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) if False else None
SRC = "/home/jpoveralls/Documents/Projects_code/foodfriends/model_src"
os.chdir(SRC)
sys.path.insert(0, SRC)
import scipy.stats as st
import model_main
from model_runn import DEFAULT_PARAMS, load_pmf_tables

DIRECT_KG = 664
SEEDS = list(range(42, 48))          # 6 paired runs
STEPS = 30000
N = 385


def prefix_step(self, G, agents, t):
    """Pre-fix Agent.step: credits the sampled neighbour regardless of its diet."""
    neighbours = [agents[n] for n in G.neighbors(self.i)]
    if not neighbours:
        return
    other_agent = random.choice(neighbours)
    self.memory.append((other_agent.diet, other_agent.i))

    if not self.immune and self.flip(self.prob_calc(other_agent)):
        old_diet = self.diet
        self.diet = "meat" if self.diet == "veg" else "veg"
        self.C = self.diet_emissions(self.diet)
        delta = self.C_base["meat"] - self.C_base["veg"]

        if old_diet == "meat" and self.diet == "veg":
            self.change_time = t
            self.influence_parent = other_agent.i
            other_agent.influenced_agents.add(self.i)
            self._cascade_attribute(delta, other_agent, agents, t)

        elif old_diet == "veg" and self.diet == "meat":
            if self.influence_parent is not None:
                agents[self.influence_parent].influenced_agents.discard(self.i)
                self.influence_parent = None
                self.change_time = None
    else:
        self.C = st.lognorm.rvs(s=0.20, scale=self.C_base[self.diet])


def one_run(job):
    arm, seed = job
    if arm == "pre":
        model_main.Agent.step = prefix_step
    np.random.seed(seed); random.seed(seed)
    p = DEFAULT_PARAMS.copy()
    p.update({"N": N, "steps": STEPS, "agent_ini": "sample-max", "seed": seed})
    m = model_main.Model(p, pmf_tables=None)
    m.run()
    snap = m.snapshots['final']
    reds = np.array(snap['reductions'])
    pos = reds[reds > 1e-3] / DIRECT_KG
    return {
        'arm': arm, 'seed': seed,
        'final_veg_f': m.fraction_veg[-1],
        'top': float(pos.max()), 'mean': float(pos.mean()), 'median': float(np.median(pos)),
        'p99': float(np.percentile(pos, 99)), 'p95': float(np.percentile(pos, 95)),
        'total_credit': float(reds.sum()), 'n_pos': int(len(pos)),
        'n_parents': int(sum(x is not None for x in snap['influence_parents'])),
    }


if __name__ == "__main__":
    jobs = [(a, s) for a in ("post", "pre") for s in SEEDS]
    with Pool(6) as pool:
        res = pool.map(one_run, jobs)
    with open('/tmp/claude-1000/-home-jpoveralls-Documents-Projects-code-foodfriends/52421b30-0366-4fe3-aae1-80d758999dfe/scratchpad/smoke_res.pkl', 'wb') as f:
        pickle.dump(res, f)

    import pandas as pd
    df = pd.DataFrame(res)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))
    print()
    piv = df.pivot(index='seed', columns='arm')
    for k in ['top', 'mean', 'median', 'p95', 'total_credit', 'final_veg_f']:
        pre, post = piv[k]['pre'], piv[k]['post']
        print(f"{k:14s} pre median={np.median(pre):9.3f}  post median={np.median(post):9.3f}  "
              f"ratio(post/pre) per-seed median={np.median(post/pre):.3f}")
