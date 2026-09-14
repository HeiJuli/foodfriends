# Source data for the main figures

Numerical values behind every main-figure panel, written by
`analysis/export_source_data.py` from the artefacts the plotting scripts read
(kappa = 0.55, N = 2000 twin ensemble of 50 runs, 2026-09-03; see `README.md`).
No per-agent survey parameter (theta, rho, alpha) is included; Fig. 5 is binned.

| Figure / panel | File | Producing script |
|---|---|---|
| Fig. 1A, F_veg trajectories, 50 runs, every 100 steps to 350,000 | `fig1a_fveg_trajectories.csv` | `plotting/publication_plots_main.py` (`network_agency_evolution_ensemble`) |
| Fig. 1B CCDF and 1C Lorenz: per-agent veg-time credit, own veg-time and A at t_end = 310,000 | `fig1bc_fig3_vegtime_credit_per_agent.csv` | same; ledger from `analysis/vegtime_accounting.py` |
| Fig. 1 network row: the small-N median run (run 14 of `trajectory_sample-max_20260903_kappa0p55_100k.pkl`), snapshots t = 0, 30,000, 56,000 | `fig1_network_{initial,mid,final}_t*_{nodes,edges}.csv`, `fig1_network_run_events.csv` | same |
| Fig. 2 cascade band: full conversion event log, initial diets and parameters of the seed-42 run | `fig2_cascade_run_events.csv`, `fig2_cascade_run_initial_diets.csv`, `fig2_cascade_run_params.json` | `plotting/cascade_overview.py` |
| Fig. 3 amplification: per-agent A (file above) and per-run ledger statistics | `fig3_vegtime_stats_per_run.csv` | `plotting/publication_plots_main.py` (`amplification_ensemble`) |
| Fig. 4 two DVs: per-run pseudo-R2, R2 and coefficients at t = 310,000 | `fig4_two_dv_vegtime.csv`, `fig4_two_dv_vegtime_network.csv` | `plotting/agency_predictor_analysis.py --from-csv` |
| Fig. 5 survey parameter distributions: theta 40-bin histogram with the skew-normal fit, rho and alpha value counts | `fig5a_theta_histogram.csv`, `fig5b_rho_bar.csv`, `fig5c_alpha_bar.csv`, `fig5_summary.json` | `data/data_analysis/parameter_distributions_paper.py` |

Event rows: `kind` is `conv` (meat to veg) or `rev` (veg to meat); `t` the step;
for conversions `partner`/`partner_diet` is the sampled contact and
`memory_buffer_diet_source_t` the agent's memory buffer at that step as a JSON
list of `[diet, source agent, step sampled]`. Credit conventions:
`analysis/attribution_ledger.py`.
