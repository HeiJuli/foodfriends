#!/usr/bin/env python3
"""Third arm: credit the most recent VEG source in the memory buffer (the thing h_soc
is actually built from), falling back to spontaneous only if the buffer holds no veg
source at all. Reduces to the current fix whenever the sampled partner is veg."""
import os, sys, random, pickle
import numpy as np
from multiprocessing import Pool

SRC = "/home/jpoveralls/Documents/Projects_code/foodfriends/model_src"
os.chdir(SRC); sys.path.insert(0, SRC)
import scipy.stats as st
import model_main
from model_runn import DEFAULT_PARAMS

DIRECT_KG = 664
SEEDS = list(range(42, 48))
STEPS, N = 30000, 385


def mem_step(self, G, agents, t):
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
            M = self.params["M"]
            src = next((sid for d, sid in reversed(self.memory[-M:])
                        if d == "veg" and sid != self.i), None)
            if src is not None:
                self.influence_parent = src
                agents[src].influenced_agents.add(self.i)
                self._cascade_attribute(delta, agents[src], agents, t)

        elif old_diet == "veg" and self.diet == "meat":
            if self.influence_parent is not None:
                agents[self.influence_parent].influenced_agents.discard(self.i)
                self.influence_parent = None
                self.change_time = None
    else:
        self.C = st.lognorm.rvs(s=0.20, scale=self.C_base[self.diet])


def one_run(seed):
    model_main.Agent.step = mem_step
    np.random.seed(seed); random.seed(seed)
    p = DEFAULT_PARAMS.copy()
    p.update({"N": N, "steps": STEPS, "agent_ini": "sample-max", "seed": seed})
    m = model_main.Model(p, pmf_tables=None)
    m.run()
    snap = m.snapshots['final']
    reds = np.array(snap['reductions'])
    pos = reds[reds > 1e-3] / DIRECT_KG
    # who ends up holding credit: is the parent currently veg?
    diets = np.array(snap['diets'])
    par = [x for x in snap['influence_parents'] if x is not None]
    return {'arm': 'mem', 'seed': seed, 'final_veg_f': m.fraction_veg[-1],
            'top': float(pos.max()), 'mean': float(pos.mean()),
            'median': float(np.median(pos)), 'p95': float(np.percentile(pos, 95)),
            'p99': float(np.percentile(pos, 99)),
            'total_credit': float(reds.sum()), 'n_pos': int(len(pos)),
            'n_parents': len(par),
            'parent_now_veg': float(np.mean([diets[i] == 'veg' for i in par])) if par else np.nan}


if __name__ == "__main__":
    with Pool(6) as pool:
        res = pool.map(one_run, SEEDS)
    pickle.dump(res, open(os.path.join(os.path.dirname(__file__), 'smoke_arm3.pkl'), 'wb'))
    import pandas as pd
    print(pd.DataFrame(res).to_string(index=False))
