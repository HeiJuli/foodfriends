"""Self-credit share on the reported (veg-time) ledger, notation audit row 13."""
import sys, glob, numpy as np, pandas as pd
sys.path[:0] = ['analysis', 'model_src', '.']
from attribution_ledger import replay
T_END = 310000
rows = []
for f in sorted(glob.glob('model_output/trajectory_analysis_twin_20260903_kappa0p55_N2000_reduced/run_*.pkl'))[:10]:
    d = pd.read_pickle(f)
    for unit in ('time', 'event'):
        so = {}
        c = replay(d['events'], d['initial_diets'], d['params'],
                   parent='exposure', weight='none', unit=unit,
                   decay=0.7, t_end=T_END, self_out=so)
        rows.append(dict(run=f[-10:-4], unit=unit, total=c.sum(),
                         self_credit=sum(so.values()),
                         share=sum(so.values()) / c.sum(),
                         max_depth=max(so) if so else 0))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
for u, g in df.groupby('unit'):
    print(f"{u}: mean share {g['share'].mean()*100:.2f}%  median {g['share'].median()*100:.2f}%  "
          f"range [{g['share'].min()*100:.2f}, {g['share'].max()*100:.2f}]%")
