"""
Multi-objective orchestration experiment.

Sweeps the AMOS fidelity-floor knob f_target (the delivered-fidelity target the
satisficing term aims for). Raising it above the hard success threshold F_th
biases placement toward higher-fidelity hosts, trading goodput for mean delivered
fidelity. This traces a real Pareto front the orchestrator can select an
operating point on. Everything is measured from the same discrete-event
simulator; nothing is hand-set.
"""
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dqsim import build_fleet, SimConfig, gen_tasks, Simulator
from schedulers import AMOS

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.7, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "legend.fontsize": 6.5, "legend.frameon": False,
    "lines.linewidth": 1.3, "lines.markersize": 4.5,
})

N_TASKS = 1500
SEEDS = 15
F_TARGETS = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]   # lam blend values
LOADS = [1.0, 1.5]


def run_point(lam, load, seed):
    fleet = build_fleet()
    wl = np.random.default_rng(1000 + seed)
    cfg = SimConfig(seed=seed)
    tasks = gen_tasks(N_TASKS, load, cfg, wl)
    rr = np.random.default_rng(7000 + seed)
    sched = AMOS(learn=True, lam=lam)
    sim = Simulator(fleet, cfg, sched, rr)
    return sim.run(copy.deepcopy(tasks))


def main():
    R = {L: {ft: {"goodput": [], "mean_fidelity": [], "deadline_hit": []}
             for ft in F_TARGETS} for L in LOADS}
    for L in LOADS:
        for ft in F_TARGETS:
            for seed in range(SEEDS):
                m = run_point(ft, L, seed)
                R[L][ft]["goodput"].append(m["goodput"])
                R[L][ft]["mean_fidelity"].append(m["mean_fidelity"])
                R[L][ft]["deadline_hit"].append(m["deadline_hit"])
            g = np.mean(R[L][ft]["goodput"])
            f = np.mean(R[L][ft]["mean_fidelity"])
            print(f"[pareto] load={L} f_target={ft:.2f} "
                  f"goodput={g:.3f} mean_fid={f:.3f}", flush=True)
    np.save("R_pareto.npy", np.array([R], dtype=object), allow_pickle=True)

    # ---- figure: Pareto front (goodput vs delivered fidelity) ----
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    colors = {1.0: "#1e88e5", 1.5: "#c62828"}
    marks = {1.0: "o", 1.5: "s"}
    for L in LOADS:
        gs = [np.mean(R[L][ft]["goodput"]) for ft in F_TARGETS]
        fs = [np.mean(R[L][ft]["mean_fidelity"]) for ft in F_TARGETS]
        ax.plot(fs, gs, color=colors[L], marker=marks[L], ls="-",
                label=f"$\\rho={L}$")
        # annotate the extreme operating points
        ax.annotate(f"$\\lambda$={F_TARGETS[0]:.2f}", (fs[0], gs[0]),
                    textcoords="offset points", xytext=(4, -8), fontsize=5.5,
                    color=colors[L])
        ax.annotate(f"{F_TARGETS[-1]:.2f}", (fs[-1], gs[-1]),
                    textcoords="offset points", xytext=(2, 4), fontsize=5.5,
                    color=colors[L])
    ax.set_xlabel("Mean delivered fidelity")
    ax.set_ylabel("Goodput (success fraction)")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_pareto.pdf")
    plt.close(fig)
    print("saved fig_pareto.pdf")

    # ---- table dump ----
    with open("pareto_table.txt", "w") as fh:
        for L in LOADS:
            fh.write(f"\n== load {L} ==\n")
            fh.write(f"{'f_target':>9} {'goodput':>9} {'mean_fid':>9} "
                     f"{'dl_hit':>8}\n")
            for ft in F_TARGETS:
                g = np.mean(R[L][ft]["goodput"])
                f = np.mean(R[L][ft]["mean_fidelity"])
                d = np.mean(R[L][ft]["deadline_hit"])
                fh.write(f"{ft:9.2f} {g:9.3f} {f:9.3f} {d:8.3f}\n")
    print(open("pareto_table.txt").read())


if __name__ == "__main__":
    main()
