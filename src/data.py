"""Motor-insurance classification experiment: probability distributions and classification maps."""

import numpy as np
from typing import Dict

# Eight repair-cost price levels (in hundreds of currency units)
R_LEVELS: np.ndarray = np.array([88, 92, 96, 100, 104, 108, 112, 116], dtype=np.float64)

# Three driver-risk groups
D_LEVELS: np.ndarray = np.array([1, 2, 3])

# P(R = r_j | D = d): row d-1 is the conditional distribution for group D = d
COND_PROB: np.ndarray = np.array(
    [
        [0.20, 0.20, 0.16, 0.13, 0.11, 0.09, 0.07, 0.04],  # D = 1
        [0.08, 0.12, 0.17, 0.20, 0.18, 0.13, 0.08, 0.04],  # D = 2
        [0.04, 0.06, 0.09, 0.12, 0.15, 0.18, 0.19, 0.17],  # D = 3
    ],
    dtype=np.float64,
)

# Equal group prior probabilities
RHO: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)


def build_joint_distribution(rho: np.ndarray = RHO) -> Dict[str, np.ndarray]:
    """Build the joint distribution of (D, R) and derived quantities.

    Parameters
    ----------
    rho:
        (3,) array of group probabilities P(D = d) for d = 1, 2, 3.
        Defaults to the uniform prior RHO = (1/3, 1/3, 1/3).

    Returns
    -------
    dict containing:
        'joint'               : (3, 8) array; p_joint[d, j] = rho[d] * COND_PROB[d, j]
        'marginal_R'          : (8,) marginal distribution of R
        'marginal_D'          : (3,) marginal distribution of D (equals rho)
        'cond_prob_R_given_D' : (3, 8) COND_PROB (unchanged)
        'group_means'         : (3,) E[R | D = d] for d = 1, 2, 3
        'overall_mean'        : scalar E[R]
    """
    rho = np.asarray(rho, dtype=np.float64)

    joint = rho[:, np.newaxis] * COND_PROB          # (3, 8)
    marginal_R = joint.sum(axis=0)                   # (8,)
    marginal_D = joint.sum(axis=1)                   # (3,)
    group_means = (COND_PROB * R_LEVELS).sum(axis=1) # (3,)
    overall_mean = float((marginal_R * R_LEVELS).sum())

    _validate_joint(joint, marginal_R, group_means, overall_mean)

    return {
        "joint": joint,
        "marginal_R": marginal_R,
        "marginal_D": marginal_D,
        "cond_prob_R_given_D": COND_PROB,
        "group_means": group_means,
        "overall_mean": overall_mean,
    }


def _validate_joint(
    joint: np.ndarray,
    marginal_R: np.ndarray,
    group_means: np.ndarray,
    overall_mean: float,
) -> None:
    """Run internal consistency checks on the joint distribution."""
    if not np.allclose(COND_PROB.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Each row of COND_PROB must sum to 1.")
    if not np.allclose(joint.sum(), 1.0, atol=1e-12):
        raise ValueError("Joint distribution must sum to 1.")

    expected_marg = np.array([32, 38, 42, 45, 44, 40, 34, 25], dtype=np.float64) / 300
    if not np.allclose(marginal_R, expected_marg, atol=1e-12):
        raise ValueError(
            f"marginal_R mismatch.\n  computed: {marginal_R}\n  expected: {expected_marg}"
        )

    expected_means = np.array([98.0, 100.76, 105.72])
    if not np.allclose(group_means, expected_means, atol=1e-9):
        raise ValueError(
            f"group_means mismatch.\n  computed: {group_means}\n  expected: {expected_means}"
        )

    expected_overall = 7612 / 75
    if not np.isclose(overall_mean, expected_overall, atol=1e-9):
        raise ValueError(
            f"overall_mean mismatch: computed {overall_mean}, expected {expected_overall}"
        )


def get_classification_map(name: str) -> np.ndarray:
    """Return the (8,) integer array mapping each R price level to a class index (0-based).

    Parameters
    ----------
    name:
        One of 'C8', 'C4', 'C3', 'C2'.

    Returns
    -------
    np.ndarray of shape (8,) with dtype int64.
        class_map[j] is the class assigned to price level R_LEVELS[j].

    Class definitions
    -----------------
    C8: {88}->0, {92}->1, ..., {116}->7  (identity; eight singleton classes)
    C4: {88,92}->0, {96,100}->1, {104,108}->2, {112,116}->3
    C3: {88,92,96}->0, {100,104}->1, {108,112,116}->2
    C2: {88,92,96,100}->0, {104,108,112,116}->1
    """
    maps: Dict[str, np.ndarray] = {
        "C8": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        "C4": np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
        "C3": np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64),
        "C2": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
    }
    if name not in maps:
        raise ValueError(f"Unknown classification name '{name}'. Choose from {list(maps)}.")
    return maps[name]
