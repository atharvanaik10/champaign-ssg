from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)


def build_payoffs_from_risk(risk: np.ndarray, alpha: float, beta: float, gamma: float, delta: float):
    risk = np.array(risk, dtype=float)
    defender_reward = alpha * risk
    defender_loss = -beta * risk
    attacker_reward = gamma * risk
    attacker_loss = -delta * risk
    return defender_reward, defender_loss, attacker_reward, attacker_loss


def solve_ssg(R_d, P_d, R_a, P_a, w, K):
    n = len(R_d)
    R_d = np.array(R_d, float)
    P_d = np.array(P_d, float)
    R_a = np.array(R_a, float)
    P_a = np.array(P_a, float)
    w = np.array(w, float)

    best_utility = -np.inf
    best_x = None

    for attacked in range(n):
        x = cp.Variable(n)

        constraints = [
            x >= 0,
            x <= 1,
            w @ x <= K,
        ]

        Ua_attacked = x[attacked] * P_a[attacked] + (1 - x[attacked]) * R_a[attacked]

        for j in range(n):
            if j == attacked:
                continue
            Ua_j = x[j] * P_a[j] + (1 - x[j]) * R_a[j]
            constraints.append(Ua_attacked >= Ua_j)

        Ud_attacked = x[attacked] * R_d[attacked] + (1 - x[attacked]) * P_d[attacked]
        prob = cp.Problem(cp.Maximize(Ud_attacked), constraints)
        prob.solve(solver=cp.ECOS)

        if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            continue

        if prob.value > best_utility + 1e-6:
            best_utility = prob.value
            best_x = np.clip(x.value, 0, 1)

    if best_x is None:
        raise RuntimeError("All SSG LPs infeasible. Check payoffs/resources.")

    logger.info("Best defender utility: %.3f", best_utility)
    return best_x, float(best_utility)
