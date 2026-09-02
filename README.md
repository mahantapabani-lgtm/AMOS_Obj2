# Reproducing the results in the AMOS paper

This directory contains **all** the code that produces every quantitative result,
figure, and table in the paper *"AI-Driven Multi-Objective Orchestration and
Adaptive Scheduling for Noise-Aware Distributed Quantum Computing."*

Everything is a self-contained discrete-event simulation in Python (NumPy +
Matplotlib). No quantum hardware, no external datasets, no network access. All
runs are seeded, so the numbers below reproduce **exactly**.

---

## 1. Requirements

```
python >= 3.9
numpy
matplotlib
```

Install:

```bash
pip install -r requirements.txt
```

---

## 2. One-command reproduction

```bash
python final_run.py      # Figs 1, 2, 4 and Tables I, III  (~1-3 min)
python pareto_run.py     # Table II (weight-sensitivity sweep)
python make_sens_fig.py  # Fig 3 (reads the file pareto_run.py just wrote)
```

After these three commands the four figure PDFs used by the paper
(`fig_goodput.pdf`, `fig_metrics_bar.pdf`, `fig_sensitivity.pdf`,
`fig_nonstat.pdf`) and the two text tables (`tables.txt`, `pareto_table.txt`)
are regenerated in this folder.

---

## 3. File-by-file description

| File | Role in the paper |
|------|-------------------|
| `dqsim.py` | The simulator. Implements the **System Model (Sec. III)**: the 8-QPU heterogeneous fleet (`build_fleet`), Poisson task stream (`gen_tasks`), the monolithic and distributed fidelity/timing models of Eqs. (1)-(3), calibration drift, the change-point (degradation) mechanism, and the success / goodput accounting. |
| `schedulers.py` | All scheduling policies. Baselines `RandomScheduler`, `RoundRobin`, `LeastLatency`, `MaxFidelityGreedy`, and the proposed **`AMOS`** class (**Sec. IV**): least-laxity ordering, the satisficing multi-objective host score of Eqs. (5)-(6), the `lam` satisfice-to-maximize knob, and the online reliability derate. `AMOS(learn=False)` is the **AMOS-S** ablation. |
| `run_experiments.py` | Helper module: `make_scheduler()` and the canonical `SCHEDULERS` list (import target for the runners). Can also be run standalone for a quick grid. |
| `final_run.py` | **Main experiment driver.** Runs the load sweep and the non-stationary experiment and writes the figures/tables below. |
| `pareto_run.py` | **Weight-sensitivity experiment.** Sweeps the AMOS fidelity knob `lam` from 0 (satisfice) to 1 (maximize); writes `R_pareto.npy` and `pareto_table.txt`. |
| `make_sens_fig.py` | Renders **Fig. 3** from `R_pareto.npy`. |
| `requirements.txt` | Python dependencies. |

Intermediate data files written during a run (safe to delete/regenerate):
`R_sweep.npy`, `R_nonstat.npy`, `R_pareto.npy`.

---

## 4. Exact mapping: paper artifact -> code

| Paper artifact | Produced by | Function |
|----------------|-------------|----------|
| **Fig. 1** — Goodput vs. offered load | `final_run.py` | `experiment_load_sweep()` -> `fig_goodput()` |
| **Fig. 2** — Cross-objective bars at rho=1.5 | `final_run.py` | `experiment_load_sweep()` -> `fig_metrics_bar()` |
| **Fig. 3** — Weight-sensitivity (lambda sweep) | `pareto_run.py` + `make_sens_fig.py` | `main()` -> `make_sens_fig.py` |
| **Fig. 4** — Windowed goodput under degradation | `final_run.py` | `experiment_nonstationary()` -> `fig_nonstat()` |
| **Table I** — Objective metrics at rho=1.0, 1.5 | `final_run.py` | `tables()` -> `tables.txt` (block "TABLE I") |
| **Table II** — Weight-sensitivity of AMOS | `pareto_run.py` | -> `pareto_table.txt` |
| **Table III** — Non-stationary pre/post goodput | `final_run.py` | `tables()` -> `tables.txt` (block "TABLE III") |
| Goodput-vs-load values quoted in Sec. VI-A | `final_run.py` | `tables()` -> `tables.txt` ("goodput vs load") |

> Naming note: the paper calls the proposed orchestrator **AMOS** and its
> no-learning ablation **AMOS-S**. The code uses those exact names. (Earlier
> internal drafts called them "IDQS"/"IDQS-Static"; all code here has been
> renamed to match the paper.)

---

## 5. Key parameters (match paper Sec. V)

Set in `dqsim.py::SimConfig` and `schedulers.py::AMOS.__init__`:

```
Fleet:        M = 8 heterogeneous QPUs        (build_fleet)
Physics:      F_th = 0.50   F_link = 0.94   t_epr = 2 us
              beta = 0.12   kappa = 0.5     drift_sigma = 0.35
Workload:     N = 1500 tasks, 15 seeds
Load sweep:   rho in {0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0}
AMOS weights: wF = 1.6, wD = 1.3, wT = 0.7, wL = 0.25, alpha = 0.2
              lam = 0.0 (satisficing default; swept in pareto_run.py)
Degradation:  at 40% of the stream, nodes {1,5} error x10  (final_run.py)
```

### Determinism
Every configuration uses paired seeds: the workload (`np.random.default_rng(1000+seed)`)
and the drift stream (`np.random.default_rng(7000+seed)`) are shared across all
schedulers for a given seed, so comparisons are paired and the reported means are
reproducible bit-for-bit on the same NumPy version.

---

## 6. Scope / honesty note
These are simulation results under an abstracted, first-order noise model
(see the paper's *Limitations* section). They are meant as controlled, relative
comparisons between scheduling policies within one common model, not as
predictions of any specific hardware.
