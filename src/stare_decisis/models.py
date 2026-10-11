"""Belief models held in segment memory.

A model declares what it can do with its sufficient statistics:

- incremental: fold a new observation into the state (inserts are cheap)
- subtractable: remove an observation exactly (retraction without replay)

Anything that is neither is refit from the surviving evidence (replay).
Conjugate exponential-family models are both; EM mixtures are neither.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Iterable

from .hashing import content_hash


class SegmentModel(ABC):
    name: str = "abstract"
    incremental: bool = False
    subtractable: bool = False

    @abstractmethod
    def params(self) -> dict[str, Any]: ...

    @abstractmethod
    def empty_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def summary(self, state: dict[str, Any]) -> dict[str, Any]: ...

    def add(self, state: dict[str, Any], x: Any) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} is not incremental")

    def subtract(self, state: dict[str, Any], x: Any) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} is not subtractable")

    def fit(self, values: Iterable[Any]) -> dict[str, Any]:
        state = self.empty_state()
        for x in values:
            state = self.add(state, x)
        return state

    def spec(self) -> dict[str, Any]:
        return {"model": self.name, "params": self.params()}

    def spec_hash(self) -> str:
        return content_hash(self.spec())


class BetaBernoulli(SegmentModel):
    """Posterior over a success rate.  Observations are 0/1 or bool."""

    name = "beta_bernoulli"
    incremental = True
    subtractable = True

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def params(self) -> dict[str, Any]:
        return {"prior_alpha": self.prior_alpha, "prior_beta": self.prior_beta}

    def empty_state(self) -> dict[str, Any]:
        return {"successes": 0, "failures": 0}

    @staticmethod
    def _bit(x: Any) -> int:
        if x in (0, 1) or isinstance(x, bool):
            return int(x)
        raise ValueError(f"BetaBernoulli expects 0/1, got {x!r}")

    def add(self, state, x):
        s = dict(state)
        if self._bit(x):
            s["successes"] += 1
        else:
            s["failures"] += 1
        return s

    def subtract(self, state, x):
        s = dict(state)
        key = "successes" if self._bit(x) else "failures"
        if s[key] <= 0:
            raise ValueError("cannot subtract an observation that was never added")
        s[key] -= 1
        return s

    def summary(self, state):
        a = self.prior_alpha + state["successes"]
        b = self.prior_beta + state["failures"]
        n = state["successes"] + state["failures"]
        return {
            "alpha": a,
            "beta": b,
            "mean": a / (a + b),
            "var": (a * b) / ((a + b) ** 2 * (a + b + 1)),
            "n": n,
        }


class NormalKnownVariance(SegmentModel):
    """Posterior over a mean with known observation noise."""

    name = "normal_known_variance"
    incremental = True
    subtractable = True

    def __init__(self, prior_mean: float, prior_var: float, noise_var: float):
        if prior_var <= 0 or noise_var <= 0:
            raise ValueError("variances must be positive")
        self.prior_mean = float(prior_mean)
        self.prior_var = float(prior_var)
        self.noise_var = float(noise_var)

    def params(self) -> dict[str, Any]:
        return {
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "noise_var": self.noise_var,
        }

    def empty_state(self) -> dict[str, Any]:
        return {"n": 0, "sum": 0.0}

    def add(self, state, x):
        return {"n": state["n"] + 1, "sum": state["sum"] + float(x)}

    def subtract(self, state, x):
        if state["n"] <= 0:
            raise ValueError("cannot subtract from an empty state")
        return {"n": state["n"] - 1, "sum": state["sum"] - float(x)}

    def summary(self, state):
        precision = 1.0 / self.prior_var + state["n"] / self.noise_var
        var = 1.0 / precision
        mean = var * (self.prior_mean / self.prior_var + state["sum"] / self.noise_var)
        return {"mean": mean, "var": var, "n": state["n"]}


def _log_normal_pdf(x: float, mean: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + (x - mean) ** 2 / var)


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


class GaussianMixtureEM(SegmentModel):
    """One-dimensional Gaussian mixture fitted by EM.

    Neither incremental nor subtractable: every change is a replay.  The
    summary reports the component with the highest mean as the "high"
    regime, which is the usual way a latent-state estimate feeds a decision.
    """

    name = "gaussian_mixture_em"
    incremental = False
    subtractable = False

    def __init__(self, k: int = 2, max_iter: int = 200, tol: float = 1e-8, var_floor: float = 1e-6):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = int(k)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.var_floor = float(var_floor)

    def params(self) -> dict[str, Any]:
        return {"k": self.k, "max_iter": self.max_iter, "tol": self.tol, "var_floor": self.var_floor}

    def empty_state(self) -> dict[str, Any]:
        return {
            "n": 0,
            "weights": [],
            "means": [],
            "variances": [],
            "loglik": None,
            "iterations": 0,
            "converged": True,
        }

    def fit(self, values: Iterable[Any]) -> dict[str, Any]:
        xs = [float(v) for v in values]
        n = len(xs)
        if n == 0:
            return self.empty_state()
        mean_all = sum(xs) / n
        var_all = max(sum((x - mean_all) ** 2 for x in xs) / n, self.var_floor)
        k = min(self.k, len(set(xs)))
        if k <= 1:
            ll = sum(_log_normal_pdf(x, mean_all, var_all) for x in xs)
            return {
                "n": n,
                "weights": [1.0],
                "means": [mean_all],
                "variances": [var_all],
                "loglik": ll,
                "iterations": 0,
                "converged": True,
            }

        s = sorted(xs)
        means = [s[min(n - 1, int((j + 0.5) * n / k))] for j in range(k)]
        variances = [var_all] * k
        weights = [1.0 / k] * k
        prev_ll = -math.inf
        converged = False
        it = 0
        ll = -math.inf
        for it in range(1, self.max_iter + 1):
            # E-step: responsibilities in log space
            resp: list[list[float]] = []
            ll = 0.0
            for x in xs:
                logs = [
                    math.log(weights[j]) + _log_normal_pdf(x, means[j], variances[j])
                    if weights[j] > 0
                    else -math.inf
                    for j in range(k)
                ]
                lse = _logsumexp(logs)
                ll += lse
                resp.append([math.exp(lj - lse) for lj in logs])
            # M-step
            for j in range(k):
                nj = sum(r[j] for r in resp)
                if nj < 1e-12:
                    weights[j] = 0.0
                    continue
                weights[j] = nj / n
                means[j] = sum(r[j] * x for r, x in zip(resp, xs)) / nj
                variances[j] = max(
                    sum(r[j] * (x - means[j]) ** 2 for r, x in zip(resp, xs)) / nj,
                    self.var_floor,
                )
            if ll - prev_ll < self.tol:
                converged = True
                break
            prev_ll = ll

        order = sorted(range(k), key=lambda j: means[j])
        return {
            "n": n,
            "weights": [weights[j] for j in order],
            "means": [means[j] for j in order],
            "variances": [variances[j] for j in order],
            "loglik": ll,
            "iterations": it,
            "converged": converged,
        }

    def summary(self, state):
        if state["n"] == 0:
            return {"n": 0, "mean": None, "p_high": None, "components": []}
        mixture_mean = sum(w * m for w, m in zip(state["weights"], state["means"]))
        return {
            "n": state["n"],
            "mean": mixture_mean,
            "p_high": state["weights"][-1],
            "high_mean": state["means"][-1],
            "components": [
                {"weight": w, "mean": m, "var": v}
                for w, m, v in zip(state["weights"], state["means"], state["variances"])
            ],
            "loglik": state["loglik"],
            "iterations": state["iterations"],
            "converged": state["converged"],
        }
