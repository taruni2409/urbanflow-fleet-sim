"""Scenario experiments: many configurations x many random seeds, run in parallel.

One simulated day is one random sample. Decisions need the *distribution*, so
every scenario runs with several seeds and we report the mean with a 95%
confidence interval (Student t, because the number of seeds is small).
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config
from .simulation import simulate


def _run_one(args) -> dict:
    config_path, n_vehicles, policy, seed = args
    cfg = load_config(config_path).with_overrides(n_vehicles=n_vehicles, policy=policy)
    return {"city": cfg.city.name, "n_vehicles": n_vehicles, "policy": policy,
            "seed": seed, **simulate(cfg, seed=seed)}


def run_grid(config_path: str, fleet_sizes, policies, seeds, workers: int | None = None) -> pd.DataFrame:
    jobs = list(itertools.product([config_path], fleet_sizes, policies, seeds))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_run_one, jobs))
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Mean and 95% CI half-width for each metric, per (n_vehicles, policy)."""
    keys = ["city", "n_vehicles", "policy"]
    metrics = metrics or [c for c in runs.columns if c not in keys + ["seed"]]
    out = []
    for key, g in runs.groupby(keys):
        row = dict(zip(keys, key))
        n = len(g)
        t = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        for m in metrics:
            row[m] = g[m].mean()
            row[f"{m}_ci95"] = t * g[m].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        row["n_seeds"] = n
        out.append(row)
    return pd.DataFrame(out)
