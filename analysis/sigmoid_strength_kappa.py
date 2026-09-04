"""Run from analysis/: python sigmoid_strength_kappa.py
Sigmoid strength of the kappa=0.55 N=2000 ensemble, per run, on [0, T].
Metrics: logistic vs linear R^2 on the raw trajectory; t10-t90 width / T;
max slope / mean slope (savgol on 100x decimated); F at max velocity."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
T = 310000
z = np.load('../model_output/fc_viability_cache_kappa0p55_N2000.npz')
X = z['trajectories'][:, :T+1]
def logi(t, L, k, t0, b): return L/(1+np.exp(-k*(t-t0)))+b
rows = []
for x in X:
    t = np.arange(len(x), dtype=float)
    lin = np.polyfit(t, x, 1); r2_lin = 1-np.sum((x-np.polyval(lin,t))**2)/np.sum((x-x.mean())**2)
    p, _ = curve_fit(logi, t, x, p0=[x[-1]-x[0], 1e-4, T*0.5, x[0]],
                     bounds=([0,0,0,0],[1,1e-2,2*T,0.5]), maxfev=50000)
    r2_log = 1-np.sum((x-logi(t,*p))**2)/np.sum((x-x.mean())**2)
    xd = x[::100]; td = t[::100]
    s = savgol_filter(xd, 201, 3, deriv=1, delta=100.)   # per step
    lo, hi = x[0], x[-1]; F10 = lo+0.1*(hi-lo); F90 = lo+0.9*(hi-lo)
    t10 = t[np.argmax(x>=F10)]; t90 = t[np.argmax(x>=F90)]
    rows.append([r2_lin, r2_log, (t90-t10)/T, s.max()/((hi-lo)/T), xd[np.argmax(s)]])
R = np.array(rows)
names = ['R2 linear','R2 logistic','(t90-t10)/T','max/mean slope','F at max slope']
for n, c in zip(names, R.T):
    q = np.percentile(c, [25,50,75]); print(f"{n:16s} median {q[1]:.3f}  IQR [{q[0]:.3f}, {q[2]:.3f}]  min {c.min():.3f} max {c.max():.3f}")
print("logistic > linear in", int((R[:,1]>R[:,0]).sum()), "/ 50 runs")
