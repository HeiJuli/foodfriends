"""Parameter sweep: W_t x M x tc x w_d for model_main_single.py"""

import sys, os, itertools, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_main_single import Model, params as base_params

# Sweep values
W_T_VALUES = [15, 18, 21, 24, 27, 30]
M_VALUES = [7, 8, 9]
TC_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
W_D_VALUES = [5, 10, 15, 20]


def run_single(args):
    """Run one parameter combination"""
    w_t, M, tc, w_d, run_id = args
    p = deepcopy(base_params)
    p["w_t"] = w_t
    p["M"] = M
    p["tc"] = tc
    p["w_d"] = w_d
    p["steps"] = 80000

    print(f"[RUN] w_t={w_t}, M={M}, tc={tc:.1f}, w_d={w_d} (run {run_id})")
    model = Model(p)
    model.run()

    frac = model.fraction_veg[::200]
    return (w_t, M, tc, w_d, frac)


def main():
    combos = [(w, m, tc, wd, i) for i, (w, m, tc, wd) in
              enumerate(itertools.product(W_T_VALUES, M_VALUES, TC_VALUES, W_D_VALUES))]
    n_total = len(combos)
    n_workers = min(cpu_count() - 1, 8)
    print(f"Sweep: {n_total} combos, {n_workers} workers")

    with Pool(n_workers) as pool:
        results = pool.map(run_single, combos)

    data = {(r[0], r[1], r[2], r[3]): r[4] for r in results}

    os.makedirs("../model_output", exist_ok=True)
    with open("../model_output/sweep_wt_m_tc_wd.pkl", "wb") as f:
        pickle.dump({"data": data, "W_T": W_T_VALUES, "M": M_VALUES,
                     "TC": TC_VALUES, "W_D": W_D_VALUES}, f)
    print("Results saved to model_output/sweep_wt_m_tc_wd.pkl")

    plot_results(data)


def plot_results(data):
    os.makedirs("../visualisations_output", exist_ok=True)

    # --- Plot 1: Trajectory grids per w_d (rows=M, cols=tc, lines=W_t) ---
    cmap = plt.cm.viridis(np.linspace(0, 1, len(W_T_VALUES)))
    for wd in W_D_VALUES:
        fig, axes = plt.subplots(len(M_VALUES), len(TC_VALUES),
                                 figsize=(4*len(TC_VALUES), 3.5*len(M_VALUES)),
                                 sharex=True, sharey=True)
        for ri, M in enumerate(M_VALUES):
            for ci, tc in enumerate(TC_VALUES):
                ax = axes[ri, ci]
                for wi, w_t in enumerate(W_T_VALUES):
                    traj = data.get((w_t, M, tc, wd))
                    if traj is not None:
                        ax.plot(np.arange(len(traj)) * 200, traj,
                                color=cmap[wi], label=f"$w_t$={w_t}", lw=1.2)
                if ri == 0:
                    ax.set_title(f"tc={tc:.1f}", fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f"M={M}\nVeg fraction", fontsize=9)
                if ri == len(M_VALUES)-1:
                    ax.set_xlabel("Steps", fontsize=9)
                ax.set_ylim(0, 0.6)
                ax.tick_params(labelsize=8)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(W_T_VALUES),
                   fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"Trajectories: $w_t$ x M x tc  (w_d={wd})", fontsize=12, y=1.05)
        plt.tight_layout()
        fname = f"../visualisations_output/sweep_trajectories_wd{wd}.png"
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fname}")

    # --- Plot 2: Heatmaps of final veg fraction per w_d (rows=W_t, cols=tc, panels=M) ---
    for wd in W_D_VALUES:
        fig2, axes2 = plt.subplots(1, len(M_VALUES),
                                   figsize=(5*len(M_VALUES), 4.5), sharey=True)
        for mi, M in enumerate(M_VALUES):
            mat = np.array([[data.get((w_t, M, tc, wd), [np.nan])[-1]
                             for tc in TC_VALUES] for w_t in W_T_VALUES])
            ax = axes2[mi]
            im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5, origin='lower')
            ax.set_xticks(range(len(TC_VALUES)))
            ax.set_xticklabels([f"{v:.1f}" for v in TC_VALUES], fontsize=9)
            ax.set_xlabel("tc", fontsize=10)
            ax.set_title(f"M={M}", fontsize=11)
            if mi == 0:
                ax.set_yticks(range(len(W_T_VALUES)))
                ax.set_yticklabels(W_T_VALUES, fontsize=9)
                ax.set_ylabel("$w_t$", fontsize=10)
            for wi in range(len(W_T_VALUES)):
                for ci in range(len(TC_VALUES)):
                    ax.text(ci, wi, f"{mat[wi,ci]:.2f}", ha='center', va='center', fontsize=7)

        fig2.colorbar(im, ax=axes2, label="Final veg fraction", shrink=0.8)
        fig2.suptitle(f"Final veg fraction: $w_t$ x tc x M  (w_d={wd})", fontsize=12)
        plt.tight_layout()
        fname = f"../visualisations_output/sweep_heatmaps_wd{wd}.png"
        fig2.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {fname}")

    # --- Plot 3: Final veg fraction vs w_d (rows=M, cols=W_t, lines=tc) ---
    tc_colors = plt.cm.cool(np.linspace(0, 1, len(TC_VALUES)))
    fig3, axes3 = plt.subplots(len(M_VALUES), len(W_T_VALUES),
                                figsize=(4*len(W_T_VALUES), 3.5*len(M_VALUES)),
                                sharex=True, sharey=True)
    for ri, M in enumerate(M_VALUES):
        for wi, w_t in enumerate(W_T_VALUES):
            ax = axes3[ri, wi]
            for ci, tc in enumerate(TC_VALUES):
                finals = [data.get((w_t, M, tc, wd), [np.nan])[-1] for wd in W_D_VALUES]
                ax.plot(W_D_VALUES, finals, 'o-', color=tc_colors[ci],
                        label=f"tc={tc:.1f}", lw=1.5)
            if ri == 0:
                ax.set_title(f"$w_t$={w_t}", fontsize=10)
            if wi == 0:
                ax.set_ylabel(f"M={M}\nFinal veg frac", fontsize=9)
            if ri == len(M_VALUES)-1:
                ax.set_xlabel("$w_d$", fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)

    handles, labels = axes3[0, 0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc='upper center', ncol=len(TC_VALUES),
                fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig3.suptitle("Final veg fraction vs $w_d$", fontsize=12, y=1.05)
    plt.tight_layout()
    fig3.savefig("../visualisations_output/sweep_final_vs_wd.png", dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: visualisations_output/sweep_final_vs_wd.png")

    # --- Plot 4: Final veg fraction vs W_t per (M, w_d), lines=tc ---
    fig4, axes4 = plt.subplots(len(M_VALUES), len(W_D_VALUES),
                                figsize=(4*len(W_D_VALUES), 3.5*len(M_VALUES)),
                                sharex=True, sharey=True)
    for ri, M in enumerate(M_VALUES):
        for di, wd in enumerate(W_D_VALUES):
            ax = axes4[ri, di]
            for ci, tc in enumerate(TC_VALUES):
                finals = [data.get((w_t, M, tc, wd), [np.nan])[-1] for w_t in W_T_VALUES]
                ax.plot(W_T_VALUES, finals, 'o-', color=tc_colors[ci],
                        label=f"tc={tc:.1f}", lw=1.5)
            if ri == 0:
                ax.set_title(f"$w_d$={wd}", fontsize=10)
            if di == 0:
                ax.set_ylabel(f"M={M}\nFinal veg frac", fontsize=9)
            if ri == len(M_VALUES)-1:
                ax.set_xlabel("$w_t$", fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)

    handles, labels = axes4[0, 0].get_legend_handles_labels()
    fig4.legend(handles, labels, loc='upper center', ncol=len(TC_VALUES),
                fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig4.suptitle("Final veg fraction vs $w_t$ by $w_d$", fontsize=12, y=1.05)
    plt.tight_layout()
    fig4.savefig("../visualisations_output/sweep_final_vs_wt_by_wd.png", dpi=200, bbox_inches='tight')
    plt.close(fig4)
    print("Saved: visualisations_output/sweep_final_vs_wt_by_wd.png")


if __name__ == "__main__":
    main()
