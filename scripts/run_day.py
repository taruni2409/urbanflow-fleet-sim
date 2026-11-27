"""Simulate one day and print its KPIs.

    python scripts/run_day.py --config configs/dense_metro.yaml --vehicles 100 --policy batched
"""
import argparse
from pathlib import Path

import pandas as pd

from fleetsim import FleetSimulation, load_config
from fleetsim.plots import day_timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense_metro.yaml")
    ap.add_argument("--vehicles", type=int)
    ap.add_argument("--policy", choices=["greedy", "batched"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", help="save a timeline chart to this path")
    args = ap.parse_args()

    cfg = load_config(args.config).with_overrides(n_vehicles=args.vehicles, policy=args.policy)
    sim = FleetSimulation(cfg, seed=args.seed)
    kpis = sim.run()

    print(f"{cfg.city.name} | {cfg.fleet.n_vehicles} vehicles | {cfg.ops.policy} | seed {args.seed}")
    for k, v in kpis.items():
        print(f"  {k:<26} {v:,.3f}" if isinstance(v, float) else f"  {k:<26} {v:,}")

    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        day_timeline(pd.DataFrame(sim.timeline), args.plot,
                     f"{cfg.city.name}: one simulated day, {cfg.fleet.n_vehicles} vehicles, {cfg.ops.policy}")
        print(f"chart -> {args.plot}")


if __name__ == "__main__":
    main()
