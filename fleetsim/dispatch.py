"""Matching policies: which idle car serves which waiting rider.

Both take a cost matrix C of shape (riders, cars) holding pickup minutes,
with np.inf where a pairing is not allowed (too far, not enough battery).
They return a list of (rider_index, car_index) pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

INFEASIBLE = 1e9


def greedy(cost: np.ndarray) -> list[tuple[int, int]]:
    """First come, first served: oldest rider takes the nearest free car.

    Fast and simple, but an early rider can 'steal' a car that was the only
    nearby option for someone else.
    """
    pairs = []
    taken = np.zeros(cost.shape[1], dtype=bool)
    for i in range(cost.shape[0]):                 # riders are in request order
        row = np.where(taken, np.inf, cost[i])
        j = int(np.argmin(row))
        if np.isfinite(row[j]):
            pairs.append((i, j))
            taken[j] = True
    return pairs


def batched(cost: np.ndarray) -> list[tuple[int, int]]:
    """Batch matching: minimise TOTAL pickup time across everyone waiting.

    Solved exactly with the Hungarian algorithm (scipy linear_sum_assignment).
    Fewer empty kilometres usually means more trips per car-hour.
    """
    finite = np.where(np.isfinite(cost), cost, INFEASIBLE)
    rows, cols = linear_sum_assignment(finite)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if finite[i, j] < INFEASIBLE]


POLICIES = {"greedy": greedy, "batched": batched}
