"""
Cumulative exposure round 2:
- Bug-fixed version of cum_replaces_memory
- Longer run (400k steps) to see full S-curve
- Decay sweep to find optimal timescale
"""
import sys, os, math, copy
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, ".."))
os.chdir(_dir)
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_main_single import Model, Agent, params

_original_prob_calc = Agent.prob_calc
_original_calc_utility = Agent.calc_utility
_original_step = Agent.step

def _make_cum_step(decay):
    def step_fn(self, G, agents, t):
        if not hasattr(self, 'cum_exp'):
            self.cum_exp = 0.0
        self.neighbours = [agents[nb] for nb in G.neighbors(self.i)]
        if not self.neighbours:
            return
        other_agent = random.choice(self.neighbours)
        self.memory.append(other_agent.diet)
        obs = 1.0 if other_agent.diet == "veg" else 0.0
        self.cum_exp = decay * self.cum_exp + (1 - decay) * obs
        prob_switch = self.prob_calc(other_agent)
        if not self.immune and np.random.random() < prob_switch:
            old_C, old_diet = self.C, self.diet
            self.diet = "meat" if self.diet == "veg" else "veg"
            self.C = self.diet_emissions(self.diet)
            if old_diet == "meat" and self.diet == "veg":
                self.influence_parent = other_agent.i
                self.change_time = t
                other_agent.influenced_agents.append(self.i)
                self.reduction_tracker(old_C, other_agent, agents, cascade_depth=1)
        else:
            self.C = np.random.normal(self.C, 0.1 * self.C)
    return step_fn

def _make_cum_replaces_memory_fixed():
    """Bug-fixed: social_ratio correctly maps cum_exp to the queried diet.
    cum_exp = fraction of veg observed cumulatively.
    - querying veg: social_ratio = cum_exp
    - querying meat: social_ratio = 1 - cum_exp
    """
    def calc_util(self, other_agent, mode):
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"
        if len(self.memory) == 0:
            return 0.0
        cum = getattr(self, 'cum_exp', 0.0)
        social_ratio = cum if diet == "veg" else (1.0 - cum)
        social = self.beta * (3 * social_ratio - 1.5)
        diss = self.alpha * self.dissonance_new("simple", mode)
        return 0.6 * social + 0.4 * diss
    return calc_util

def _make_cum_replaces_memory_buggy():
    """Original (buggy) version for comparison - inverts veg agent social feedback."""
    def calc_util(self, other_agent, mode):
        if mode == "same":
            diet = self.diet
        else:
            diet = "meat" if self.diet == "veg" else "veg"
        if len(self.memory) == 0:
            return 0.0
        cum = getattr(self, 'cum_exp', 0.0)
        social_ratio = cum if diet != self.diet else (1.0 - cum)
        social = self.beta * (3 * social_ratio - 1.5)
        diss = self.alpha * self.dissonance_new("simple", mode)
        return 0.6 * social + 0.4 * diss
    return calc_util

def _std_prob():
    def prob_fn(self, other_agent):
        u_i = self.calc_utility(other_agent, mode="same")
        u_s = self.calc_utility(other_agent, mode="diff")
        return 1 / (1 + math.exp(-6 * (u_s - u_i)))
    return prob_fn

def run_variant(label, p, decay, calc_util_fn, n_traj=3):
    trajs = []
    Agent.step = _make_cum_step(decay)
    Agent.calc_utility = calc_util_fn
    Agent.prob_calc = _std_prob()
    for t in range(n_traj):
        model = Model(p)
        model.run()
        trajs.append(model.fraction_veg)
        print(f"  {label} traj {t+1}/{n_traj}: final={model.fraction_veg[-1]:.3f}", flush=True)
    return trajs

def plot_variants(variants, filename, title, steps):
    n = len(variants)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows), sharey=True, squeeze=False)
    xticks = np.linspace(0, steps, 5, dtype=int)
    for idx, (label, trajs) in enumerate(variants):
        ax = axes[idx // cols][idx % cols]
        for tr in trajs:
            ax.plot(tr, alpha=0.5, linewidth=0.7)
        ax.set_xlabel("t (steps)")
        ax.set_title(label, fontsize=9)
        ax.set_ylim(-0.02, 0.8)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x//1000}k" for x in xticks], fontsize=7)
    for idx in range(len(variants), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    axes[0][0].set_ylabel("Vegetarian Fraction")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"../visualisations_output/{filename}", dpi=200, bbox_inches="tight")
    print(f"\nSaved {filename}", flush=True)

if __name__ == "__main__":
    p = copy.deepcopy(params)
    p["steps"] = 400000
    N_TRAJ = 3

    print("=" * 60, flush=True)
    print("Cumulative Exposure Round 2: Bug-fix + Decay Sweep", flush=True)
    print("=" * 60, flush=True)

    configs = [
        ("Baseline\n(memory, no cum)", 0.995, None),
        ("Buggy (round 1)\ndecay=0.99", 0.99,  _make_cum_replaces_memory_buggy()),
        ("Fixed\ndecay=0.99", 0.99,  _make_cum_replaces_memory_fixed()),
        ("Fixed\ndecay=0.995", 0.995, _make_cum_replaces_memory_fixed()),
        ("Fixed\ndecay=0.998", 0.998, _make_cum_replaces_memory_fixed()),
        ("Fixed\ndecay=0.999", 0.999, _make_cum_replaces_memory_fixed()),
    ]

    results = []
    for i, (label, decay, util_fn) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {label.replace(chr(10), ' ')}", flush=True)
        if util_fn is None:
            Agent.step = _original_step
            Agent.calc_utility = _original_calc_utility
            Agent.prob_calc = _original_prob_calc
            trajs = []
            for t in range(N_TRAJ):
                model = Model(p)
                model.run()
                trajs.append(model.fraction_veg)
                print(f"  {label} traj {t+1}/{N_TRAJ}: final={model.fraction_veg[-1]:.3f}", flush=True)
        else:
            trajs = run_variant(label, p, decay=decay, calc_util_fn=util_fn, n_traj=N_TRAJ)
        results.append((label, trajs))

    plot_variants(results, "scurve_cum_fixed.png",
                  "Cumulative Exposure: Bug-fix + Decay Sweep (400k steps)", p["steps"])

    Agent.step = _original_step
    Agent.calc_utility = _original_calc_utility
    Agent.prob_calc = _original_prob_calc
    print("\nDone.", flush=True)
