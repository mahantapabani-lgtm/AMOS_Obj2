import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.7, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "legend.fontsize": 6.5, "legend.frameon": False,
    "lines.linewidth": 1.3, "lines.markersize": 4.0,
})

R = np.load("R_pareto.npy", allow_pickle=True)[0]
LAMS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
LOADS = [1.0, 1.5]

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax2 = ax.twinx()
c_g = {1.0: "#1e88e5", 1.5: "#1565c0"}
c_f = {1.0: "#ef6c00", 1.5: "#c62828"}
for L in LOADS:
    g = np.array([R[L][lam]["goodput"] for lam in LAMS])
    f = np.array([R[L][lam]["mean_fidelity"] for lam in LAMS])
    ax.errorbar(LAMS, g.mean(1), yerr=g.std(1), color=c_g[L], marker="o",
                ls="-", capsize=1.5, elinewidth=0.6,
                label=f"goodput ($\\rho$={L})")
    ax2.errorbar(LAMS, f.mean(1), yerr=f.std(1), color=c_f[L], marker="s",
                 ls="--", capsize=1.5, elinewidth=0.6,
                 label=f"fidelity ($\\rho$={L})")
ax.set_xlabel("Fidelity weighting $\\lambda$  (satisfice $\\to$ maximize)")
ax.set_ylabel("Goodput", color="#1565c0")
ax2.set_ylabel("Mean delivered fidelity", color="#c62828")
ax.set_ylim(0.44, 0.55)
ax2.set_ylim(0.44, 0.52)
ax.tick_params(axis="y", labelcolor="#1565c0")
ax2.tick_params(axis="y", labelcolor="#c62828")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=5.6, ncol=1)
fig.tight_layout(pad=0.3)
fig.savefig("fig_sensitivity.pdf")
print("saved fig_sensitivity.pdf")
