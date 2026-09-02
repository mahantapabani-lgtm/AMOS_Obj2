import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dqsim import build_fleet, SimConfig, gen_tasks, Simulator
from run_experiments import make_scheduler, SCHEDULERS

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
})

STYLE = {
    "Random":        ("#9e9e9e", "o", "-"),
    "Round-Robin":   ("#7e57c2", "s", "-"),
    "Least-Latency": ("#26a69a", "^", "-"),
    "Max-Fidelity":  ("#ef6c00", "D", "-"),
    "AMOS-S":   ("#1e88e5", "v", "--"),
    "AMOS":          ("#c62828", "*", "-"),
}
N_TASKS = 1500
SEEDS = 15
LOADS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


# --------------------------------------------------------------------------- #
def experiment_load_sweep():
    fleet = build_fleet()
    keys = ["goodput", "deadline_hit", "mean_fidelity", "utilization",
            "mean_wait"]
    R = {s: {L: {k: [] for k in keys} for L in LOADS} for s in SCHEDULERS}
    for L in LOADS:
        for seed in range(SEEDS):
            wl = np.random.default_rng(1000 + seed)
            cfg = SimConfig(seed=seed)
            master = gen_tasks(N_TASKS, L, cfg, wl)
            for s in SCHEDULERS:
                tasks = copy.deepcopy(master)
                rr = np.random.default_rng(7000 + seed)
                sim = Simulator(fleet, cfg,
                                make_scheduler(s, np.random.default_rng(50 + seed)),
                                rr)
                m = sim.run(tasks)
                for k in keys:
                    R[s][L][k].append(m[k])
        print(f"[sweep] load {L} done", flush=True)
    np.save("R_sweep.npy", np.array([R], dtype=object), allow_pickle=True)
    return R, keys


def experiment_nonstationary():
    fleet = build_fleet()
    changes = [(0.40, {1: 10.0, 5: 10.0})]      # cleanest nodes degrade 10x
    subs = ["Max-Fidelity", "AMOS-S", "AMOS"]
    W = 120
    curves = {s: [] for s in subs}
    prepost = {s: {"pre": [], "post": []} for s in subs}
    for s in subs:
        for seed in range(SEEDS):
            wl = np.random.default_rng(1000 + seed)
            cfg = SimConfig(seed=seed)
            master = gen_tasks(N_TASKS, 1.0, cfg, wl)
            tasks = copy.deepcopy(master)
            rr = np.random.default_rng(7000 + seed)
            sim = Simulator(fleet, cfg,
                            make_scheduler(s, np.random.default_rng(50 + seed)),
                            rr, changes=changes)
            sim.run(tasks)
            outc = [x[1] for x in sim._trace]
            n = len(outc); cp = int(0.40 * n)
            prepost[s]["pre"].append(np.mean(outc[:cp]))
            prepost[s]["post"].append(np.mean(outc[cp:]))
            wc = [np.mean(outc[max(0, i - W):i]) for i in range(W, n + 1)]
            curves[s].append(wc)
    # align curve lengths
    for s in subs:
        m = min(len(c) for c in curves[s])
        curves[s] = np.array([c[:m] for c in curves[s]])
    np.save("R_nonstat.npy",
            np.array([{"curves": curves, "prepost": prepost, "W": W,
                       "changes": changes}], dtype=object), allow_pickle=True)
    return curves, prepost, W


# --------------------------------------------------------------------------- #
def fig_goodput(R):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for s in SCHEDULERS:
        c, mk, ls = STYLE[s]
        mu = [np.mean(R[s][L]["goodput"]) for L in LOADS]
        sd = [np.std(R[s][L]["goodput"]) for L in LOADS]
        ax.errorbar(LOADS, mu, yerr=sd, color=c, marker=mk, ls=ls,
                    capsize=1.5, elinewidth=0.6, label=s)
    ax.set_xlabel("Offered load  $\\rho$")
    ax.set_ylabel("Goodput (success fraction)")
    ax.set_ylim(0.15, 0.68)
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_goodput.pdf")
    plt.close(fig)


def fig_metrics_bar(R):
    Lrep = 1.5
    metrics = [("goodput", "Goodput"), ("deadline_hit", "Deadline hit"),
               ("mean_fidelity", "Mean fidelity"), ("utilization", "Utilization")]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(metrics))
    w = 0.13
    for i, s in enumerate(SCHEDULERS):
        c, _, _ = STYLE[s]
        vals = [np.mean(R[s][Lrep][k]) for k, _ in metrics]
        errs = [np.std(R[s][Lrep][k]) for k, _ in metrics]
        ax.bar(x + (i - 2.5) * w, vals, w, color=c, yerr=errs,
               error_kw=dict(elinewidth=0.5, capsize=1), label=s)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], rotation=12)
    ax.set_ylabel(f"Value at $\\rho={Lrep}$")
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28),
              fontsize=6, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_metrics_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_nonstat(curves, W):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for s in ["Max-Fidelity", "AMOS-S", "AMOS"]:
        c, mk, ls = STYLE[s]
        arr = curves[s]
        mu = arr.mean(0)
        sd = arr.std(0)
        xs = np.arange(len(mu)) + W
        ax.plot(xs, mu, color=c, ls=ls, label=s)
        ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
    # change point marker: 40% of the trace
    total = len(curves["AMOS"][0]) + W
    cp = 0.40 * (total)
    ax.axvline(cp, color="k", ls=":", lw=0.8)
    ax.text(cp + 15, 0.60, "device\ndegradation", fontsize=6)
    ax.set_xlabel("Resolved task index")
    ax.set_ylabel(f"Windowed goodput (w={W})")
    ax.legend(loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_nonstat.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def tables(R, prepost):
    with open("tables.txt", "w") as f:
        f.write("=== TABLE I (paper): objective metrics at rho=1.0 and rho=1.5 (mean+/-std) ===\n")
        for L in [1.0, 1.5]:
            f.write(f"\n-- load {L} --\n")
            f.write(f"{'sched':14s} {'goodput':>16s} {'dl_hit':>14s} "
                    f"{'fidelity':>14s} {'util':>12s}\n")
            for s in SCHEDULERS:
                g = R[s][L]["goodput"]; d = R[s][L]["deadline_hit"]
                fi = R[s][L]["mean_fidelity"]; u = R[s][L]["utilization"]
                f.write(f"{s:14s} {np.mean(g):.3f}+/-{np.std(g):.3f}   "
                        f"{np.mean(d):.3f}+/-{np.std(d):.3f}  "
                        f"{np.mean(fi):.3f}+/-{np.std(fi):.3f}  "
                        f"{np.mean(u):.3f}+/-{np.std(u):.3f}\n")
        f.write("\n=== goodput vs load (mean) ===\n")
        f.write("load  " + " ".join(f"{s:>13}" for s in SCHEDULERS) + "\n")
        for L in LOADS:
            f.write(f"{L:4.2f} " +
                    " ".join(f"{np.mean(R[s][L]['goodput']):13.3f}"
                             for s in SCHEDULERS) + "\n")
        f.write("\n=== TABLE III (paper): non-stationary pre/post degradation ===\n(Table II in the paper = weight-sensitivity, produced by pareto_run.py)\n")
        for s in ["Max-Fidelity", "AMOS-S", "AMOS"]:
            pre = np.mean(prepost[s]["pre"]); post = np.mean(prepost[s]["post"])
            f.write(f"{s:14s} pre={pre:.3f}  post={post:.3f}  "
                    f"drop={pre-post:+.3f}\n")
    print(open("tables.txt").read())


if __name__ == "__main__":
    R, _ = experiment_load_sweep()
    curves, prepost, W = experiment_nonstationary()
    fig_goodput(R)
    fig_metrics_bar(R)
    fig_nonstat(curves, W)
    tables(R, prepost)
    print("\nAll figures and tables generated.")
