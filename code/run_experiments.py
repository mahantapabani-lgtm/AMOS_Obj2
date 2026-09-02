import sys
import numpy as np
from dqsim import build_fleet, SimConfig, gen_tasks, Simulator
from schedulers import (RandomScheduler, RoundRobin, LeastLatency,
                        MaxFidelityGreedy, AMOS)


def make_scheduler(name, rng):
    if name == "Random":       return RandomScheduler(rng)
    if name == "Round-Robin":  return RoundRobin()
    if name == "Least-Latency":return LeastLatency()
    if name == "Max-Fidelity": return MaxFidelityGreedy()
    if name == "AMOS-S":  return AMOS(learn=False)
    if name == "AMOS":         return AMOS(learn=True)
    raise ValueError(name)


SCHEDULERS = ["Random", "Round-Robin", "Least-Latency",
              "Max-Fidelity", "AMOS-S", "AMOS"]


def run_grid(n_tasks, loads, seeds, out_prefix):
    fleet = build_fleet()
    metrics_keys = ["goodput", "deadline_hit", "mean_fidelity",
                    "mean_wait", "utilization", "dispatched"]
    # results[sched][load][metric] = list over seeds
    results = {s: {L: {m: [] for m in metrics_keys} for L in loads}
               for s in SCHEDULERS}

    for L in loads:
        for seed in range(seeds):
            # same workload across schedulers for paired comparison
            wl_rng = np.random.default_rng(1000 + seed)
            cfg = SimConfig(seed=seed)
            tasks_master = gen_tasks(n_tasks, L, cfg, wl_rng)
            for s in SCHEDULERS:
                # fresh copies of tasks (mutable runtime fields)
                import copy
                tasks = copy.deepcopy(tasks_master)
                run_rng = np.random.default_rng(7000 + seed)  # same drift stream
                sched = make_scheduler(s, np.random.default_rng(50 + seed))
                sim = Simulator(fleet, cfg, sched, run_rng)
                m = sim.run(tasks)
                for k in metrics_keys:
                    results[s][L][k].append(m[k])
        print(f"  load={L:.2f} done", flush=True)

    np.save(out_prefix + ".npy",
            np.array([results], dtype=object), allow_pickle=True)
    return results, loads, metrics_keys


def summarize(results, loads):
    print("\n=== Goodput (mean +/- std) ===")
    header = "load  " + "  ".join(f"{s:>13}" for s in SCHEDULERS)
    print(header)
    for L in loads:
        row = f"{L:4.2f}  "
        for s in SCHEDULERS:
            g = results[s][L]["goodput"]
            row += f"{np.mean(g):6.3f}±{np.std(g):5.3f}  "
        print(row)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        res, loads, _ = run_grid(200, [0.6, 1.2], 2, "results_smoke")
        summarize(res, loads)
    else:
        loads = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
        res, loads, _ = run_grid(1500, loads, 15, "results_full")
        summarize(res, loads)
