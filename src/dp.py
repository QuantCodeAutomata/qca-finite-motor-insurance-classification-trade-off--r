"""Dynamic-programming optimal contiguous classification.

Implements the recursion for the weighted objective L(C) + λ·Γ(C) over
contiguous partitions of the ordered price levels {r_1,...,r_K}.

Block cost w_λ(i,j) = Σ_{h=i}^{j} q_h (r_h - r̄_B)² + 2λ s_B ‖η_B - η̄‖²
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data import R_LEVELS, RHO


def precompute_block_costs(
    marginal_R: np.ndarray,
    cond_prob: np.ndarray,
    rho_reg: np.ndarray,
    lambda_val: float,
    eta_bar: Optional[np.ndarray] = None,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Dict]]:
    """Pre-compute block statistics and costs for all contiguous blocks.

    Parameters
    ----------
    marginal_R : (K,) unconditional masses q_h = P(R = r_h)
    cond_prob : (D, K) conditional P(R = r_h | D = d)
    rho_reg : (D,) regulatory group weights (= rho in main case)
    lambda_val : trade-off weight λ ≥ 0
    eta_bar : (D,) target group profile, defaults to rho_reg

    Returns
    -------
    block_costs : dict mapping (i,j) -> scalar cost w_λ(i,j), 0-indexed inclusive
    block_info  : dict mapping (i,j) -> {'q_B','s_B','rbar_B','eta_B','cost_L','cost_Gamma'}
    """
    K = len(marginal_R)
    if eta_bar is None:
        eta_bar = rho_reg.copy()

    block_costs: Dict[Tuple[int, int], float] = {}
    block_info: Dict[Tuple[int, int], Dict] = {}

    for i in range(K):
        q_acc = 0.0
        s_acc = 0.0
        joint_acc = np.zeros(len(rho_reg), dtype=np.float64)
        weighted_R = 0.0

        for j in range(i, K):
            q_h = marginal_R[j]
            q_acc += q_h
            # joint mass for block (i,j) by group: rho_reg[d] * cond_prob[d, j]
            joint_acc += rho_reg * cond_prob[:, j]
            s_acc = joint_acc.sum()
            weighted_R += q_h * R_LEVELS[j]

            rbar_B = weighted_R / q_acc if q_acc > 0 else R_LEVELS[i]

            # Within-block variance: Σ q_h (r_h - r̄_B)²
            cost_L = sum(
                marginal_R[h] * (R_LEVELS[h] - rbar_B) ** 2 for h in range(i, j + 1)
            )

            # Group profile: η_B = joint_acc_by_group / s_acc
            if s_acc > 1e-15:
                eta_B = joint_acc / s_acc
                cost_Gamma = 2.0 * lambda_val * s_acc * np.sum((eta_B - eta_bar) ** 2)
            else:
                eta_B = eta_bar.copy()
                cost_Gamma = 0.0

            w = cost_L + cost_Gamma
            block_costs[(i, j)] = w
            block_info[(i, j)] = {
                "q_B": q_acc,
                "s_B": s_acc,
                "rbar_B": rbar_B,
                "eta_B": eta_B,
                "cost_L": cost_L,
                "cost_Gamma": cost_Gamma,
                "cost_total": w,
            }

    return block_costs, block_info


def dp_optimal_partition(
    block_costs: Dict[Tuple[int, int], float],
    K: int = 8,
    max_classes: int = 8,
) -> Dict:
    """Run DP recursion for optimal contiguous partition.

    V(j, l) = min_{l-1 ≤ i < j} { V(i, l-1) + w(i+1, j) }   for l ≥ 2
    V(j, 1) = w(0, j)

    All indices 0-based; j ∈ {0,...,K-1}, l ∈ {1,...,max_classes}.

    Parameters
    ----------
    block_costs : dict mapping (i,j) -> cost (from precompute_block_costs)
    K : number of price levels
    max_classes : maximum number of classes to consider

    Returns
    -------
    dict with:
        'V'          : (K, max_classes+1) objective array; V[j, l] = optimal cost for
                       first j+1 levels in l classes (inf if infeasible)
        'partitions' : dict mapping l -> list of (i, j) blocks for optimal partition
                       of all K levels into l classes
    """
    INF = np.inf
    max_classes = min(max_classes, K)

    # V[j, l]: optimal cost using price levels 0..j in exactly l contiguous classes
    V = np.full((K, max_classes + 1), INF, dtype=np.float64)
    # Back-pointers: bp[j, l] = i* where block (i*+1, j) is the last block
    bp = np.full((K, max_classes + 1), -1, dtype=np.int64)

    # Base case: l=1, a single block from 0 to j
    for j in range(K):
        V[j, 1] = block_costs[(0, j)]
        bp[j, 1] = -1  # the block starts at 0

    # Recursion
    for l_cls in range(2, max_classes + 1):
        for j in range(l_cls - 1, K):
            best = INF
            best_i = -1
            for i in range(l_cls - 1, j + 1):
                # block (i, j) is the last class; first i levels in l_cls-1 classes
                prev = V[i - 1, l_cls - 1] if i > 0 else INF
                candidate = prev + block_costs[(i, j)]
                if candidate < best:
                    best = candidate
                    best_i = i - 1  # back-pointer: last index before this block

            V[j, l_cls] = best
            bp[j, l_cls] = best_i

    # Trace back optimal partitions for each l
    partitions: Dict[int, List[Tuple[int, int]]] = {}
    for l_cls in range(1, max_classes + 1):
        if V[K - 1, l_cls] == INF:
            continue
        blocks = []
        j = K - 1
        lc = l_cls
        while lc > 0:
            if lc == 1:
                blocks.append((0, j))
                break
            i_end = bp[j, lc]   # last index of the second-to-last class
            blocks.append((i_end + 1, j))
            j = i_end
            lc -= 1
        partitions[l_cls] = list(reversed(blocks))

    return {"V": V, "partitions": partitions, "bp": bp}


def brute_force_partitions(K: int = 8, n_classes: int = 4) -> List[List[Tuple[int, int]]]:
    """Enumerate all contiguous partitions of K levels into n_classes blocks.

    Parameters
    ----------
    K : total number of ordered levels
    n_classes : desired number of contiguous blocks

    Returns
    -------
    List of partitions; each partition is a list of (i, j) inclusive blocks.
    """
    if n_classes > K or n_classes < 1:
        return []
    # Choose n_classes-1 cut points from {1,...,K-1}
    partitions = []
    for cuts in itertools.combinations(range(1, K), n_classes - 1):
        boundaries = [0] + list(cuts) + [K]
        blocks = [(boundaries[k], boundaries[k + 1] - 1) for k in range(n_classes)]
        partitions.append(blocks)
    return partitions


def evaluate_partition(
    partition: List[Tuple[int, int]], block_costs: Dict[Tuple[int, int], float]
) -> float:
    """Sum block costs for a given partition."""
    return sum(block_costs[blk] for blk in partition)


def verify_dp_vs_brute_force(
    block_costs: Dict[Tuple[int, int], float],
    K: int = 8,
    tol: float = 1e-12,
) -> Dict:
    """Compare DP optimum against brute-force for each class count.

    Parameters
    ----------
    block_costs : precomputed block costs
    K : number of levels
    tol : objective tolerance for match check

    Returns
    -------
    dict with per-n_classes results and 'all_match' summary flag
    """
    dp_result = dp_optimal_partition(block_costs, K=K, max_classes=K)
    V = dp_result["V"]
    dp_partitions = dp_result["partitions"]

    results = {}
    all_match = True

    for n in range(1, K + 1):
        bf_parts = brute_force_partitions(K, n)
        if not bf_parts:
            results[n] = {"feasible": False}
            continue

        bf_costs = [evaluate_partition(p, block_costs) for p in bf_parts]
        bf_best = min(bf_costs)
        dp_best = float(V[K - 1, n])

        match = abs(bf_best - dp_best) < tol
        if not match:
            all_match = False

        results[n] = {
            "feasible": True,
            "dp_objective": dp_best,
            "bf_objective": bf_best,
            "residual": abs(bf_best - dp_best),
            "match": match,
            "dp_partition": dp_partitions.get(n),
        }

    results["all_match"] = all_match
    return results
