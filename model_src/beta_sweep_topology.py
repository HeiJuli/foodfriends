#!/usr/bin/env python3
"""
Beta sweep x topology: how does making the Boltzmann/Hamiltonian interaction more
stochastic (lower beta) change the adoption trajectory, and does network topology
start to matter once interactions are noisier?

Arms (identical except topology):
  - ER               : Erdos-Renyi null (matched mean degree)
  - homophilic_emp    : in-review production (theta-assort ~0, mean-field)
  - homophilic_theta  : NEW theta-homophily (theta-assort ~0.27, PA kept on)

For each beta in BETAS, runs SEEDS replicates per arm and plots the mean +/- band
F_veg trajectory. Trajectories only (no stat suite), per user request.

Run from model_src/:
    python beta_sweep_topology.py [--betas 5 10 20 35 50] [--seeds 4] [--steps 30000] [--quick]
"""
import sys, os, random, pickle, argparse
from multiprocessing import Pool
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.'); sys.path.append('..')
import model_main

TOPOS  = ["ER", "homophilic_emp", "homophilic_theta"]
LABELS = {"ER": "Erdos-Renyi", "homophilic_emp": "Production (theta-assort ~0)",
          "homophilic_theta": "Theta-homophily (assort ~0.27)"}
COLORS = {"ER": "#e29578", "homophilic_emp": "#006d77", "homophilic_theta": "#9b2226"}

BASE_PARAMS = dict(model_main.params)
BASE_PARAMS.update({
    "agent_ini": "sample-max",
    "survey_file": "../data/hierarchical_agents.csv",
    "snapshot_dense_start": 12000,
    # theta-homophily frozen calibration (used only by homophilic_theta arm):
    "theta_w": 1.0, "sim_power": 4.0, "pa_power": 1.0, "tc_sim": True,
})


def _run_one(args):
    topo, beta, seed, steps = args
    np.random.seed(seed); random.seed(seed)
    p = BASE_PARAMS.copy()
    p.update({"topology": topo, "beta": beta, "seed": seed, "steps": steps})
    m = model_main.Model(p, pmf_tables=None)
    m.run()
    return {"topology": topo, "beta": beta, "seed": seed,
            "fraction_veg_trajectory": m.fraction_veg,
            "final_veg_f": m.fraction_veg[-1]}


def fig_sweep(results, betas, out):
    """One panel per beta; 3 topology mean+/-band curves each."""
    n = len(betas)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, beta in zip(axes, betas):
        for topo in TOPOS:
            trajs = [r["fraction_veg_trajectory"] for r in results
                     if r["topology"] == topo and r["beta"] == beta]
            if not trajs:
                continue
            L = min(len(x) for x in trajs)
            mat = np.array([x[:L] for x in trajs])
            t = np.arange(L)
            mu = mat.mean(0)
            lo, hi = np.percentile(mat, 5, 0), np.percentile(mat, 95, 0)
            ax.plot(t, mu, color=COLORS[topo], lw=1.8, label=LABELS[topo])
            ax.fill_between(t, lo, hi, color=COLORS[topo], alpha=0.16)
        ax.set_title(f"beta = {beta}", fontsize=10)
        ax.set_xlabel("Timestep"); ax.set_ylim(0, 1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, ls="--", alpha=0.3, zorder=-10)
    axes[0].set_ylabel("Vegetarian fraction")
    axes[-1].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle("Adoption trajectory vs Boltzmann stochasticity (lower beta = noisier)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"INFO: Saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--betas', type=float, nargs='+', default=[5, 10, 20, 35, 50])
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--cores', type=int, default=max(1, int(0.6 * os.cpu_count())))
    ap.add_argument('--quick', action='store_true', help='2 seeds x 8000 steps smoke test')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 2, 8000

    os.makedirs("../visualisations_output", exist_ok=True)
    os.makedirs("../model_output", exist_ok=True)
    today = date.today().strftime('%Y%m%d')

    jobs = [(t, b, 42 + i, args.steps)
            for t in TOPOS for b in args.betas for i in range(args.seeds)]
    print(f"INFO: {len(jobs)} runs ({len(TOPOS)} topos x {len(args.betas)} betas "
          f"x {args.seeds} seeds, steps={args.steps}) on {args.cores} cores")
    with Pool(args.cores) as pool:
        results = pool.map(_run_one, jobs)

    pkl = f"../model_output/beta_sweep_topology_{today}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"results": results, "betas": args.betas, "seeds": args.seeds,
                     "steps": args.steps}, f)
    print(f"INFO: Saved raw -> {pkl}")

    # final F_veg table
    print(f"\n{'='*64}\n FINAL F_veg (mean +/- std over seeds)\n{'='*64}")
    print(f"{'beta':>6}" + "".join(f"{LABELS[t][:18]:>20}" for t in TOPOS))
    for b in args.betas:
        cells = []
        for t in TOPOS:
            fv = [r["final_veg_f"] for r in results
                  if r["topology"] == t and r["beta"] == b]
            cells.append(f"{np.mean(fv):.3f}+/-{np.std(fv):.3f}")
        print(f"{b:>6}" + "".join(f"{c:>20}" for c in cells))

    fig_sweep(results, args.betas,
              f"../visualisations_output/beta_sweep_topology_{today}.pdf")
    print("\nINFO: done.")


if __name__ == "__main__":
    main()
