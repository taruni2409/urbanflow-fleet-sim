"""Sweep fleet size x dispatch policy x seeds, in parallel; save CSVs and a chart.

    python scripts/run_scenarios.py --config configs/dense_metro.yaml \
        --fleet-sizes 60 80 100 120 150 --seeds 10
"""
import argparse
import time
from pathlib import Path

from fleetsim.experiments import run_grid, summarise
from fleetsim.plots import fleet_size_panels

SHOW = ["revenue_per_vehicle_hour", "net_per_vehicle_hour", "fulfilment",
        "utilisation", "empty_km_share", "p90_wait_min", "mean_charger_queue_min"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense_metro.yaml")
    ap.add_argument("--fleet-sizes", type=int, nargs="+", default=[60, 80, 100, 120, 150])
    ap.add_argument("--policies", nargs="+", default=["greedy", "batched"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    runs = run_grid(args.config, args.fleet_sizes, args.policies, range(args.seeds), args.workers)
    summary = summarise(runs)
    city = runs["city"].iloc[0]

    out = Path(args.out) / city
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    fleet_size_panels(summary, out / "fleet_size.png", city)

    print(f"{len(runs)} simulated days in {time.time() - t0:.0f}s -> {out}/")
    table = summary[["n_vehicles", "policy"] + SHOW].round(3)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
