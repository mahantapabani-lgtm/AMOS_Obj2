"""
dqsim.py — Discrete-event simulator for Intelligent Distributed Quantum Task
Scheduling (AMOS study).

Model summary (see paper Sec. III for the formal statement):

QPUs  : heterogeneous NISQ nodes. QPU j has capacity n_j (qubits), nominal
        per-gate error eps_j, layer time tau_j (s), and coherence time T_j (s).
Tasks : quantum circuits. Task i has width w_i (qubits), depth L_i (layers),
        gate count g_i, arrival a_i, and absolute deadline d_i. Success requires
        completion before d_i AND estimated process fidelity >= F_th.
Exec  : monolithic (w_i <= n_j) or distributed across a set S of QPUs using
        entanglement (gate teleportation). Distribution injects |S|-1 scaling of
        non-local gates, each consuming an EPR pair (link fidelity F_link,
        latency t_epr).
Drift : realized eps_j and T_j fluctuate around nominal at each execution
        (NISQ calibration drift), so the deterministic estimate is imperfect and
        an adaptive (learning) scheduler can gain.

The simulator is scheduler-agnostic: a Scheduler object only decides, at each
event, how to map currently-pending tasks onto currently-free QPUs.
"""

import heapq
import math
import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
#  Static configuration
# --------------------------------------------------------------------------- #
@dataclass
class QPU:
    idx: int
    n: int          # qubit capacity
    eps: float      # nominal per-gate error rate
    tau: float      # layer execution time (s)
    T: float        # coherence time T2 (s)
    drift: float = 1.0   # relative calibration instability (>1 == less stable)


def build_fleet():
    """8 heterogeneous QPUs spanning the NISQ design space.

    Deliberately diverse: small-but-clean, large-but-noisy, fast, slow, etc.
    Numbers are representative of superconducting/ion-trap NISQ regimes
    (order-of-magnitude), not a specific device.
    """
    #        idx  n    eps      tau(s)   T2(s)    drift
    # Two nominally-attractive nodes (1, 4) are calibration-UNSTABLE: their
    # datasheet specs look good but realized fidelity fluctuates heavily. A
    # static policy trusts the datasheet; only an adaptive policy can learn to
    # derate them. Node 3 is both clean AND stable (the hidden "safe" choice).
    specs = [
        (0,  27, 3.0e-3, 4.0e-7, 1.2e-4, 1.0),   # mid, balanced, stable
        (1,  16, 8.0e-4, 6.0e-7, 2.5e-4, 2.6),   # clean on paper, UNSTABLE
        (2,  65, 9.0e-3, 3.0e-7, 7.0e-5, 1.0),   # large, noisy, stable
        (3,  20, 1.5e-3, 5.0e-7, 1.8e-4, 0.6),   # clean AND very stable
        (4,  40, 5.0e-3, 3.5e-7, 1.0e-4, 2.4),   # moderate on paper, UNSTABLE
        (5,  12, 6.0e-4, 8.0e-7, 3.0e-4, 0.8),   # cleanest, slow, stable
        (6,  32, 4.0e-3, 4.5e-7, 1.1e-4, 1.0),   # mid, stable
        (7,  50, 7.0e-3, 3.2e-7, 8.5e-5, 1.1),   # large, noisy, stable
    ]
    return [QPU(*s) for s in specs]


@dataclass
class SimConfig:
    F_th: float = 0.50          # success fidelity threshold
    F_link: float = 0.94        # EPR link fidelity per non-local gate
    t_epr: float = 2.0e-6       # EPR generation latency per non-local gate (s)
    beta: float = 0.12          # non-local gate density factor
    kappa: float = 0.5          # two-qubit gate density -> gate count factor
    drift_sigma: float = 0.35   # lognormal drift std on eps and 1/T at run time
    F_link_drift: float = 0.02  # additive noise on link fidelity
    seed: int = 0


@dataclass
class Task:
    idx: int
    arrival: float
    w: int              # width (qubits)
    L: int              # depth (layers)
    g: int              # gate count
    deadline: float     # absolute
    # runtime bookkeeping
    started: float = -1.0
    finished: float = -1.0
    fidelity: float = -1.0
    success: bool = False
    resolved: bool = False
    qpus: tuple = ()


# --------------------------------------------------------------------------- #
#  Workload generation
# --------------------------------------------------------------------------- #
def gen_tasks(n_tasks, load, cfg, rng):
    """Poisson arrivals; heavy-tailed-ish width/depth. `load` sets arrival rate
    relative to aggregate service capability so higher load => more contention.
    """
    # nominal mean service time reference (used only to set arrival rate scale)
    ref_service = 40 * 5.0e-7          # ~depth 40 * mid layer time
    fleet_parallelism = 8
    lam = load * fleet_parallelism / ref_service   # arrivals per second

    tasks = []
    t = 0.0
    for i in range(n_tasks):
        t += rng.exponential(1.0 / lam)
        # width: mostly small, some medium, few large (need distribution)
        u = rng.random()
        if u < 0.60:
            w = rng.integers(2, 13)      # fits smallest QPUs
        elif u < 0.88:
            w = rng.integers(13, 41)     # mid
        else:
            w = rng.integers(41, 96)     # large -> may need multiple QPUs
        L = int(rng.integers(6, 56))     # depth 8..80 layers
        g = max(1, int(cfg.kappa * w * L))
        # deadline: arrival + slack proportional to a nominal exec estimate
        nominal_exec = L * 4.5e-7
        slack = rng.uniform(2.0, 6.0)    # 2x..6x nominal exec as laxity
        deadline = t + slack * nominal_exec + 5.0e-6
        tasks.append(Task(idx=i, arrival=t, w=w, L=L, g=g, deadline=deadline))
    return tasks


# --------------------------------------------------------------------------- #
#  Physics / execution model
# --------------------------------------------------------------------------- #
def choose_partition(task, free_qpus):
    """Return the smallest set (list) of currently-free QPUs whose combined
    capacity covers task.w, preferring higher-capacity + cleaner nodes.
    Returns None if the free set cannot host the task right now.
    """
    # single-QPU fit first (prefer cleanest that fits)
    fit = [q for q in free_qpus if q.n >= task.w]
    if fit:
        return None  # signal: monolithic handled by scheduler choice
    # need distribution: greedily take largest-capacity free QPUs
    ordered = sorted(free_qpus, key=lambda q: -q.n)
    chosen, cap = [], 0
    for q in ordered:
        chosen.append(q)
        cap += q.n
        if cap >= task.w:
            return chosen
    return None  # not enough aggregate free capacity


def exec_time_and_fidelity(task, qpus, cfg, rng, use_nominal=False,
                           degrade=None):
    """Compute (exec_time, fidelity). If use_nominal, use datasheet specs
    (what the scheduler sees for prediction). Otherwise inject calibration drift
    (the realized outcome). `degrade` maps qpu_idx -> multiplicative error factor
    applied to realized eps (and inverse to T) to model post-deployment device
    degradation not reflected in the datasheet.
    """
    degrade = degrade or {}
    k = len(qpus)
    # per-QPU realized parameters
    def realized(q):
        if use_nominal:
            return q.eps, q.T
        sig = cfg.drift_sigma * q.drift
        d = degrade.get(q.idx, 1.0)
        eps = q.eps * d * rng.lognormal(0.0, sig)
        T = (q.T / d) / rng.lognormal(0.0, sig)
        return eps, T

    if k == 1:
        q = qpus[0]
        eps, T = realized(q)
        t_exec = task.L * q.tau
        F_gate = (1.0 - eps) ** task.g
        F_dec = math.exp(-t_exec / T)
        F = F_gate * F_dec
        return t_exec, F

    # distributed across k QPUs
    c = max(1, int(cfg.beta * task.L * (k - 1)))     # non-local gates
    # local layer time = slowest node; EPR ops serialized on top
    t_local = task.L * max(q.tau for q in qpus)
    t_exec = t_local + c * cfg.t_epr
    F = 1.0
    share = task.g / k
    for q in qpus:
        eps, T = realized(q)
        F *= (1.0 - eps) ** share
    Tmin = min((realized(q)[1]) for q in qpus)
    F *= math.exp(-t_exec / Tmin)
    F_link = cfg.F_link if use_nominal else max(
        0.0, cfg.F_link + rng.normal(0.0, cfg.F_link_drift))
    F *= F_link ** c
    return t_exec, F


def predict(task, qpus, cfg):
    """Deterministic nominal prediction the scheduler uses."""
    # reuse exec_time_and_fidelity with a dummy rng and use_nominal=True
    return exec_time_and_fidelity(task, qpus, cfg, np.random.default_rng(0),
                                  use_nominal=True)


# --------------------------------------------------------------------------- #
#  Simulator core
# --------------------------------------------------------------------------- #
class Simulator:
    def __init__(self, fleet, cfg, scheduler, rng, changes=None):
        self.fleet = fleet
        self.cfg = cfg
        self.sched = scheduler
        self.rng = rng
        self.free_at = {q.idx: 0.0 for q in fleet}
        # changes: list of (trigger_fraction, {qpu_idx: factor}); applied to the
        # realized-outcome degradation map when that fraction of tasks resolves.
        self.changes = sorted(changes or [], key=lambda c: c[0])
        self.degrade = {}
        self._trace = []            # (task_index_resolved, success) for curves

    def run(self, tasks):
        cfg, rng = self.cfg, self.rng
        by_idx = {t.idx: t for t in tasks}
        # event queue: (time, type, payload)
        ev = []
        for t in tasks:
            heapq.heappush(ev, (t.arrival, 0, t.idx))   # arrival
        pending = []                                    # arrived, unassigned
        n_done = 0
        n_total = len(tasks)

        while ev:
            now, etype, pid = heapq.heappop(ev)
            if etype == 0:                              # arrival
                pending.append(by_idx[pid])
            else:                                       # completion
                self.free_at[pid_qpu := pid] = now      # (qpu freed already set)

            # process all simultaneous events at this timestamp
            while ev and ev[0][0] <= now:
                nt, et, p = heapq.heappop(ev)
                if et == 0:
                    pending.append(by_idx[p])
                # completion frees handled via free_at set at scheduling time

            # expire tasks whose deadline already passed (uniform, fair)
            still = []
            for tk in pending:
                if tk.deadline <= now:
                    tk.resolved = True
                    tk.success = False
                    tk.finished = now
                    n_done += 1
                else:
                    still.append(tk)
            pending = still

            # free QPUs = those whose free_at <= now
            free_ids = [j for j, fa in self.free_at.items() if fa <= now + 1e-15]
            free_qpus = [self.fleet[j] for j in free_ids]

            if pending and free_qpus:
                assignments = self.sched.assign(now, pending, free_qpus,
                                                self.fleet, cfg)
                assigned_ids = set()
                for tk, qpus in assignments:
                    # commit: compute realized outcome, mark QPUs busy
                    t_exec, F = exec_time_and_fidelity(tk, qpus, cfg, rng,
                                                       use_nominal=False,
                                                       degrade=self.degrade)
                    start = now
                    finish = start + t_exec
                    tk.started = start
                    tk.finished = finish
                    tk.fidelity = F
                    tk.qpus = tuple(q.idx for q in qpus)
                    tk.success = (finish <= tk.deadline) and (F >= cfg.F_th)
                    tk.resolved = True
                    n_done += 1
                    self._trace.append((n_done, tk.success))
                    self._apply_changes(n_done / n_total)
                    assigned_ids.add(tk.idx)
                    for q in qpus:
                        self.free_at[q.idx] = finish
                        heapq.heappush(ev, (finish, 1, q.idx))
                    # let the scheduler learn from the (predicted) outcome signal
                    self.sched.feedback(tk, qpus, cfg)
                pending = [t for t in pending if t.idx not in assigned_ids]

            # if event queue emptied but tasks still pending (all QPUs busy),
            # the next completion event will re-trigger scheduling.
            if not ev and pending:
                # advance to earliest QPU free time
                nxt = min(self.free_at.values())
                if nxt > now:
                    heapq.heappush(ev, (nxt, 1, min(self.free_at,
                                                    key=self.free_at.get)))
                else:
                    break  # deadlock guard (should not happen)

        # resolve any leftover pending as failures
        for tk in pending:
            if not tk.resolved:
                tk.resolved = True
                tk.success = False

        return self._metrics(tasks)

    def _apply_changes(self, frac):
        while self.changes and frac >= self.changes[0][0]:
            _, factors = self.changes.pop(0)
            self.degrade.update(factors)

    def _metrics(self, tasks):
        n = len(tasks)
        succ = sum(1 for t in tasks if t.success)
        started = [t for t in tasks if t.started >= 0]
        met_dl = sum(1 for t in started if t.finished <= t.deadline)
        fids = [t.fidelity for t in started if t.fidelity >= 0]
        waits = [(t.started - t.arrival) for t in started]
        busy_time = sum((t.finished - t.started) * len(t.qpus)
                        for t in started)
        span = max((t.finished for t in started), default=1.0)
        cap = len(self.fleet)
        return dict(
            goodput=succ / n,
            deadline_hit=(met_dl / len(started)) if started else 0.0,
            mean_fidelity=float(np.mean(fids)) if fids else 0.0,
            mean_wait=float(np.mean(waits)) if waits else 0.0,
            utilization=busy_time / (span * cap) if span > 0 else 0.0,
            dispatched=len(started) / n,
        )
