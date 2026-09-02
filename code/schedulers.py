"""
schedulers.py — Baseline and proposed scheduling policies.

Every scheduler exposes:
    assign(now, pending, free_qpus, fleet, cfg) -> list[(task, [qpu,...])]
    feedback(task, qpus, cfg)                   -> None   (online learning hook)

`assign` maps a subset of currently-pending tasks onto currently-free QPUs.
A QPU may be used at most once per call. Distributed tasks (width exceeds any
single free QPU) claim a set of free QPUs.
"""

import numpy as np
from dqsim import predict, choose_partition


# --------------------------------------------------------------------------- #
def _hosts_for(task, free_pool, prefer):
    """Return a list of QPUs from free_pool that can host `task`, or None.
    `prefer` is a key(qpu)->score; monolithic picks best single fit, else the
    smallest distributed set covering the width.
    """
    fit = [q for q in free_pool if q.n >= task.w]
    if fit:
        return [max(fit, key=prefer)]
    # distributed: take largest-capacity free QPUs until covered
    ordered = sorted(free_pool, key=lambda q: -q.n)
    chosen, cap = [], 0
    for q in ordered:
        chosen.append(q); cap += q.n
        if cap >= task.w:
            return chosen
    return None


class BaseScheduler:
    name = "base"
    def feedback(self, task, qpus, cfg):
        pass


class RandomScheduler(BaseScheduler):
    name = "Random"
    def __init__(self, rng):
        self.rng = rng
    def assign(self, now, pending, free_qpus, fleet, cfg):
        out, pool = [], list(free_qpus)
        order = list(pending); self.rng.shuffle(order)
        for tk in order:
            hosts = _hosts_for(tk, pool, prefer=lambda q: self.rng.random())
            if hosts is None:
                continue
            out.append((tk, hosts))
            for q in hosts:
                pool.remove(q)
        return out


class RoundRobin(BaseScheduler):
    name = "Round-Robin"
    def __init__(self):
        self._k = 0
    def assign(self, now, pending, free_qpus, fleet, cfg):
        out, pool = [], list(free_qpus)
        for tk in sorted(pending, key=lambda t: t.arrival):
            if not pool:
                break
            # rotate preference over free pool
            key = lambda q: (q.idx - self._k) % len(fleet)
            hosts = _hosts_for(tk, pool, prefer=lambda q: -key(q))
            if hosts is None:
                continue
            out.append((tk, hosts))
            self._k = (hosts[0].idx + 1) % len(fleet)
            for q in hosts:
                pool.remove(q)
        return out


class LeastLatency(BaseScheduler):
    """FCFS ordering, assign to the free QPU giving the shortest execution
    time (throughput/latency-greedy). Ignores fidelity."""
    name = "Least-Latency"
    def assign(self, now, pending, free_qpus, fleet, cfg):
        out, pool = [], list(free_qpus)
        for tk in sorted(pending, key=lambda t: t.arrival):
            hosts = _hosts_for(tk, pool,
                               prefer=lambda q: -(tk.L * q.tau))
            if hosts is None:
                continue
            out.append((tk, hosts))
            for q in hosts:
                pool.remove(q)
        return out


class MaxFidelityGreedy(BaseScheduler):
    """FCFS ordering, assign to the free host set maximizing nominal predicted
    fidelity. Uses datasheet specs only (no adaptation)."""
    name = "Max-Fidelity"
    def assign(self, now, pending, free_qpus, fleet, cfg):
        out, pool = [], list(free_qpus)
        for tk in sorted(pending, key=lambda t: t.arrival):
            def score(q):
                _, F = predict(tk, [q], cfg)
                return F
            hosts = _hosts_for(tk, pool, prefer=score)
            if hosts is None:
                continue
            out.append((tk, hosts))
            for q in hosts:
                pool.remove(q)
        return out


# --------------------------------------------------------------------------- #
#  Proposed: AMOS
# --------------------------------------------------------------------------- #
class AMOS(BaseScheduler):
    """Intelligent Distributed Quantum Scheduler.

    (1) Urgency ordering: least-laxity-first (EDF-style) over pending tasks,
        laxity = deadline - now - min-feasible-exec-time. Hopeless tasks
        (negative laxity) are deprioritized.
    (2) Multi-objective host scoring per (task, host-set):
            s = wF*Fhat_norm + wD*deadline_feasible - wL*load - wR? + wR*Qval
        combining predicted fidelity, deadline feasibility, fleet load
        balancing, and a learned reliability value Q.
    (3) Online reliability learning: a tabular Q(state, qpu) estimates realized
        success probability for a discretized task-state, updated from feedback.
        This adapts to calibration drift the datasheet fidelity cannot capture.

    Setting `learn=False` yields the static ablation (AMOS-S).
    """
    def __init__(self, learn=True, alpha=0.2,
                 wF=1.6, wD=1.3, wL=0.25, wT=0.7, f_target=None, lam=0.0):
        self.learn = learn
        self.alpha = alpha
        self.wF, self.wD, self.wL, self.wT = wF, wD, wL, wT
        self.f_target = f_target
        # Orchestration knob lam in [0,1]: blends the fidelity term between
        # satisficing (lam=0, reward crossing F_th and saturate) and maximizing
        # (lam=1, reward raw delivered fidelity). Higher lam biases routing onto
        # the cleanest devices, trading throughput/goodput for delivered
        # fidelity -- this traces a Pareto operating curve.
        self.lam = lam
        # reliability estimate r(state,qpu) = EMA of realized success, and a
        # nominal expectation e(state,qpu) = EMA of predicted success, so the
        # ratio r/e derates devices that under-perform their datasheet.
        self.rel = {}               # (state, qpu_idx) -> realized success EMA
        self.exp = {}               # (state, qpu_idx) -> predicted success EMA
        self.load = {}              # qpu_idx -> exp-moving busy estimate
        self.name = "AMOS" if learn else "AMOS-S"

    # ---- state discretization for the learner -------------------------------
    @staticmethod
    def _state(task):
        wb = 0 if task.w <= 12 else (1 if task.w <= 40 else 2)
        db = 0 if task.L <= 25 else (1 if task.L <= 55 else 2)
        return (wb, db)

    def _derate(self, state, qidx):
        """Learned multiplicative reliability correction in (0, ~1.1].
        Returns realized/expected success ratio; 1.0 if unseen (trust datasheet).
        """
        if not self.learn:
            return 1.0
        r = self.rel.get((state, qidx))
        e = self.exp.get((state, qidx))
        if r is None or e is None or e < 1e-3:
            return 1.0
        return min(1.1, (r + 0.02) / (e + 0.02))

    # ---- main assignment ----------------------------------------------------
    def assign(self, now, pending, free_qpus, fleet, cfg):
        out, pool = [], list(free_qpus)

        def min_exec(tk):
            fit = [q for q in fleet if q.n >= tk.w]
            if fit:
                return tk.L * min(q.tau for q in fit)
            return tk.L * min(q.tau for q in fleet)

        # (1) least-laxity-first
        order = sorted(pending,
                       key=lambda t: (t.deadline - now - min_exec(t)))

        maxload = max(self.load.values()) if self.load else 1.0
        for tk in order:
            state = self._state(tk)

            fastest = min(tk.L * q.tau for q in fleet)

            def hostscore(hosts):
                _, F = predict(tk, hosts, cfg)
                # apply learned reliability derate to the predicted fidelity
                derate = np.mean([self._derate(state, q.idx) for q in hosts])
                Fc = F * derate
                # satisficing: reward crossing threshold, saturate above it, so
                # we do NOT overspend clean-but-slow QPUs on already-safe tasks.
                center = cfg.F_th if self.f_target is None else self.f_target
                Fn_sat = 1.0 / (1.0 + np.exp(-14.0 * (Fc - center)))
                # blend toward raw-fidelity maximization as lam -> 1
                Fn = (1.0 - self.lam) * Fn_sat + self.lam * Fc
                t_exec = tk.L * max(q.tau for q in hosts) + \
                    (0 if len(hosts) == 1 else
                     int(cfg.beta * tk.L * (len(hosts) - 1)) * cfg.t_epr)
                feasible = 1.0 if (now + t_exec) <= tk.deadline else -1.0
                # throughput: prefer faster hosts (free the resource sooner)
                speed = fastest / (t_exec + 1e-12)
                ld = np.mean([self.load.get(q.idx, 0.0) for q in hosts]) / \
                    (maxload + 1e-9)
                return (self.wF * Fn + self.wD * feasible + self.wT * speed
                        - self.wL * ld)

            # candidate host-sets: each single fitting QPU, or one distributed set
            fit = [q for q in pool if q.n >= tk.w]
            candidates = [[q] for q in fit]
            if not fit:
                dist = _dist_set(tk, pool)
                if dist:
                    candidates = [dist]
            if not candidates:
                continue
            best = max(candidates, key=hostscore)
            # skip clearly hopeless assignment if a better use of QPU exists
            _, Fb = predict(tk, best, cfg)
            t_execb = tk.L * max(q.tau for q in best)
            if (now + t_execb) > tk.deadline and (tk.deadline - now) > 0:
                # deadline unreachable -> don't waste the QPU on it now
                continue
            out.append((tk, best))
            for q in best:
                pool.remove(q)
                self.load[q.idx] = 0.9 * self.load.get(q.idx, 0.0) + 0.1
        # decay load for idle QPUs
        for q in fleet:
            if q.idx not in [qq.idx for _, hs in out for qq in hs]:
                self.load[q.idx] = 0.9 * self.load.get(q.idx, 0.0)
        return out

    # ---- learning from realized outcome ------------------------------------
    def feedback(self, task, qpus, cfg):
        if not self.learn:
            return
        state = self._state(task)
        realized = 1.0 if task.success else 0.0
        # datasheet-predicted success for this placement (deadline + fidelity)
        _, Fp = predict(task, list(qpus), cfg)
        t_exec = task.L * max(q.tau for q in qpus)
        pred = 1.0 if (Fp >= cfg.F_th and task.started + t_exec
                       <= task.deadline) else 0.0
        a = self.alpha
        for q in qpus:
            key = (state, q.idx)
            self.rel[key] = (1 - a) * self.rel.get(key, realized) + a * realized
            self.exp[key] = (1 - a) * self.exp.get(key, pred) + a * pred


def _dist_set(task, pool):
    ordered = sorted(pool, key=lambda q: -q.n)
    chosen, cap = [], 0
    for q in ordered:
        chosen.append(q); cap += q.n
        if cap >= task.w:
            return chosen
    return None
