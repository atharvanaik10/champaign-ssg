import json
import random

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ---------- 1. Load graph & extract risk ----------


def load_graph(path) -> nx.Graph:
    """Load uiuc_graph.json into a NetworkX Graph."""
    with open(path, "r") as f:
        data = json.load(f)

    G = nx.Graph()

    # First add all nodes with attributes
    for nid, info in data.items():
        G.add_node(
            nid,
            lat=info.get("lat"),
            lon=info.get("lon"),
            risk_factor=info.get("risk_factor", 0.0),
            crimes=info.get("crimes", []),
        )

    # Then add edges from neighbors lists
    for nid, info in data.items():
        for nbr in info.get("neighbors", []):
            if nbr in data:
                G.add_edge(nid, nbr)

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_node_list_and_risk(G):
    """Return list of node IDs and corresponding risk_factor array."""
    node_list = list(G.nodes())
    risk = np.array(
        [G.nodes[nid].get("risk_factor", 0.0) for nid in node_list],
        dtype=float,
    )
    print(f"Total risk over nodes: {risk.sum():.2f}")
    return node_list, risk


# ---------- 2. Build payoffs & solve SSG ----------


def build_payoffs_from_risk(risk, alpha, beta, gamma, delta):
    """
    Map risk -> defender & attacker payoffs.

    - Defender(police) loses alpha * risk if node is attacked & *uncovered*, 0 if covered.
    - Attacker(criminal) gains risk if uncovered, 0 if covered.
    """
    risk = np.array(risk, dtype=float)
    defender_reward = (
        alpha * risk
    )  # defender(police) reward if covered higher reward for hgiher risk.
    defender_loss = (
        -beta * risk
    )  # defender(police) loss if uncovered more loss for higher risk
    attacker_reward = (
        gamma * risk
    )  # attacker(criminal) reward if uncovered more payoff for higher risk.
    attacker_loss = (
        -delta * risk
    )  # attacker(criminal) penalty if covered they do not want to be caught in high risk places
    return defender_reward, defender_loss, attacker_reward, attacker_loss


def solve_ssg(R_d, P_d, R_a, P_a, w, K):
    """
    Single-defender Stackelberg Security Game via
    'enumerate attacked target + LP'.

    Returns:
        best_x: np.array of coverage probabilities for each node
        best_U: defender utility
    """
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
            w @ x <= K,  # resource budget
        ]

        Ua_attacked = x[attacked] * P_a[attacked] + (1 - x[attacked]) * R_a[attacked]

        for j in range(n):
            if j == attacked:
                continue
            Ua_j = x[j] * P_a[j] + (1 - x[j]) * R_a[j]
            constraints.append(Ua_attacked >= Ua_j)

        Ud_attacked = x[attacked] * R_d[attacked] + (1 - x[attacked]) * P_d[attacked]
        prob = cp.Problem(cp.Maximize(Ud_attacked), constraints)
        prob.solve(solver=cp.ECOS)  # let cvxpy pick a solver

        if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            continue

        if prob.value > best_utility + 1e-6:
            best_utility = prob.value
            best_x = np.clip(x.value, 0, 1)

    if best_x is None:
        raise RuntimeError("All SSG LPs infeasible. Check payoffs/resources.")
    return best_x, best_utility


# ---------- 3. Coverage -> patrol schedule on your custom Graph ----------


def build_transition_matrix(G: nx.Graph, node_list, coverage: np.ndarray):
    """
    Build a Markov transition matrix P over nodes, biased toward high-coverage nodes.
    Uses G.neighbors(u) to respect your graph structure.
    """
    idx = {nid: i for i, nid in enumerate(node_list)}
    n = len(node_list)
    P = np.zeros((n, n))

    for u_i, u in enumerate(node_list):
        nbrs = list(G.neighbors(u))

        if not nbrs:
            P[u_i, u_i] = 1.0
            continue

        weights = np.array([coverage[idx[v]] for v in nbrs], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        for w_i, v in enumerate(nbrs):
            P[u_i, idx[v]] = weights[w_i]

    assert np.allclose(P.sum(axis=1), 1.0)
    return P


def build_uniform_transition_matrix(G: nx.Graph, node_list):
    """Baseline patrol policy: random walk ignoring risk."""
    idx = {nid: i for i, nid in enumerate(node_list)}
    n = len(node_list)
    P = np.zeros((n, n))

    for u_i, u in enumerate(node_list):
        nbrs = list(G.neighbors(u))
        if not nbrs:
            P[u_i, u_i] = 1.0
            continue
        probs = np.ones(len(nbrs)) / len(nbrs)
        for w_i, v in enumerate(nbrs):
            P[u_i, idx[v]] = probs[w_i]

    assert np.allclose(P.sum(axis=1), 1.0)
    return P


def simulate_patrol(P, node_list, start_idx, T, num_units=1):
    """
    Random-walk patrol simulation.

    Returns a list of (time_step, unit_id, node_id).
    """
    n = len(node_list)
    current = [start_idx] * num_units
    records = []

    for t in range(T + 1):
        for u in range(num_units):
            records.append((t, u, node_list[current[u]]))
        if t < T:
            for u in range(num_units):
                current[u] = np.random.choice(n, p=P[current[u]])
    return records


# ---------------------------------------------------------------------
# 4. Crime simulation and policy evaluation
# ---------------------------------------------------------------------


def simulate_policy_once(P, node_list, risk, T, num_units=3, p_event=0.3, start_idx=0):
    """
    Run one simulation of length T for a given patrol policy P.

    Crime model:
        - At each time step, with prob p_event, a crime occurs.
        - Crime location is drawn from nodes with probability proportional to risk[i].

    Efficiency metric:
        efficiency = caught_risk / total_risk
    """
    n = len(node_list)
    risk = np.array(risk, float)
    crime_probs = risk / risk.sum()

    path = simulate_patrol(P, node_list, start_idx, T, num_units=num_units)

    patrol_by_time = {t: set() for t in range(T + 1)}
    for t, u, nid in path:
        patrol_by_time[t].add(nid)

    caught_risk = 0.0
    total_risk = 0.0

    for t in range(T + 1):
        if random.random() < p_event:
            j = np.random.choice(n, p=crime_probs)
            nid = node_list[j]
            rj = risk[j]
            total_risk += rj
            if nid in patrol_by_time[t]:
                caught_risk += rj

    return caught_risk, total_risk


def evaluate_policy(
    P, node_list, risk, T, num_units=3, p_event=0.3, start_idx=0, num_runs=200
):
    """Run many simulations and return array of efficiencies."""
    efficiencies = []
    for _ in range(num_runs):
        caught, total = simulate_policy_once(
            P,
            node_list,
            risk,
            T,
            num_units=num_units,
            p_event=p_event,
            start_idx=start_idx,
        )
        if total > 0:
            efficiencies.append(caught / total)

    efficiencies = np.array(efficiencies)
    print(
        f"Runs with ≥1 crime: {len(efficiencies)}/{num_runs}, "
        f"mean={efficiencies.mean():.3f}, std={efficiencies.std():.3f}"
    )
    return efficiencies


# ---------------------------------------------------------------------
# 5. Main: solve SSG, build policies, run sims, make plots
# ---------------------------------------------------------------------


def main():
    np.random.seed(0)
    random.seed(0)
    path = "data/uiuc_graph.json"
    G = load_graph(path)
    node_list, risk = get_node_list_and_risk(G)

    # --- Stackelberg game ---
    R_d, P_d, R_a, P_a = build_payoffs_from_risk(risk, 1.0, 1.0, 1.0, 1.0)
    K = 10.0
    w = np.ones_like(risk)

    print("\nSolving Stackelberg security game...")
    coverage, best_U = solve_ssg(R_d, P_d, R_a, P_a, w, K)
    print(f"Best defender utility: {best_U:.3f}")

    P_ssg = build_transition_matrix(G, node_list, coverage)
    P_uniform = build_uniform_transition_matrix(G, node_list)

    start_idx = 0
    T = 48
    base_num_units = 5
    p_event = 0.3
    num_runs = 300

    # --- NEW: generate one concrete SSG patrol schedule and save to CSV ---
    print("\nSimulating SSG patrol to generate patrol_schedule.csv ...")
    schedule = simulate_patrol(
        P_ssg,
        node_list,
        start_idx=start_idx,
        T=T,
        num_units=base_num_units,
    )
    df_schedule = pd.DataFrame(schedule, columns=["time_step", "unit_id", "node_id"])
    df_schedule.to_csv("patrol_schedule.csv", index=False)
    print("Saved patrol schedule to patrol_schedule.csv")

    # --- fixed units comparison (simple model) ---
    print(f"\nEvaluating SSG policy (units={base_num_units})...")
    effs_ssg = evaluate_policy(
        P_ssg,
        node_list,
        risk,
        T=T,
        num_units=base_num_units,
        p_event=p_event,
        start_idx=start_idx,
        num_runs=num_runs,
    )

    print(f"\nEvaluating Uniform policy (units={base_num_units})...")
    effs_uniform = evaluate_policy(
        P_uniform,
        node_list,
        risk,
        T=T,
        num_units=base_num_units,
        p_event=p_event,
        start_idx=start_idx,
        num_runs=num_runs,
    )

    print(
        f"\nSimple model, units={base_num_units}: "
        f"mean efficiency SSG = {effs_ssg.mean():.4f}, "
        f"Uniform = {effs_uniform.mean():.4f}"
    )

    # --- Figure 2: efficiency vs units ---
    units_list = [i for i in range(1, 10)]
    means_ssg = []
    means_uniform = []

    for u in units_list:
        print(f"\nEvaluating vs num_units={u} ...")
        effs_ssg_u = evaluate_policy(
            P_ssg,
            node_list,
            risk,
            T=T,
            num_units=u,
            p_event=p_event,
            start_idx=start_idx,
            num_runs=num_runs,
        )
        effs_uniform_u = evaluate_policy(
            P_uniform,
            node_list,
            risk,
            T=T,
            num_units=u,
            p_event=p_event,
            start_idx=start_idx,
            num_runs=num_runs,
        )
        means_ssg.append(effs_ssg_u.mean())
        means_uniform.append(effs_uniform_u.mean())

    plt.figure()
    plt.plot(units_list, means_ssg, marker="o", label="SSG patrol")
    plt.plot(units_list, means_uniform, marker="s", label="Uniform patrol")
    plt.xlabel("Number of patrol units")
    plt.ylabel("Mean efficiency")
    plt.xticks(units_list)
    plt.title(f"Efficiency vs patrol units (T={T}, p_event={p_event})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_efficiency_vs_units_nx.png", dpi=300)
    print("Saved line plot to fig_efficiency_vs_units_nx.png")


if __name__ == "__main__":
    main()
