"""Verification tests: does the simulator do what the model says it does?

These are the 'is the code right?' checks (verification). Whether the model
matches a real city is a separate question (validation) - see the README roadmap.
"""
from pathlib import Path

import numpy as np
import pytest

from fleetsim import FleetSimulation, load_config
from fleetsim.dispatch import batched, greedy

CFG = load_config(Path(__file__).parents[1] / "configs" / "dense_metro.yaml")


@pytest.fixture(scope="module")
def sim():
    s = FleetSimulation(CFG.with_overrides(n_vehicles=80), seed=7)
    s.run()
    return s


def test_same_seed_same_result():
    a = FleetSimulation(CFG, seed=3).run()
    b = FleetSimulation(CFG, seed=3).run()
    assert a == b


def test_every_request_is_accounted_for(sim):
    statuses = [r.status for r in sim.requests]
    assert set(statuses) <= {"waiting", "assigned", "completed", "lost"}
    # Only requests near the end of the day may still be open.
    end = CFG.ops.duration_hours * 60
    for r in sim.requests:
        if r.status in ("waiting", "assigned"):
            assert r.t_request > end - CFG.ops.max_wait_min - 120


def test_event_times_are_ordered(sim):
    for r in sim.requests:
        if r.status == "completed":
            assert r.t_request <= r.t_assigned <= r.t_pickup <= r.t_dropoff


def test_riders_matched_within_patience(sim):
    for r in sim.requests:
        if r.t_assigned is not None:
            assert r.t_assigned - r.t_request <= CFG.ops.max_wait_min + 1e-9


def test_batteries_never_run_flat(sim):
    for v in sim.vehicles:
        assert v.soc_kwh >= 0
        assert v.soc_kwh <= CFG.fleet.battery_kwh + 1e-9


def test_charger_capacity_is_respected(sim):
    for hub, spec in zip(sim.hubs, CFG.charging_hubs):
        assert hub.resource.capacity == spec["points"]
        assert hub.resource.count <= spec["points"]


def test_demand_volume_matches_input():
    """With no cars, arrivals are pure Poisson: total ~ daily_requests +/- 4 sigma."""
    lam = CFG.city.daily_requests * CFG.ops.duration_hours / 24
    for seed in range(5):
        s = FleetSimulation(CFG.with_overrides(n_vehicles=0), seed=seed)
        s.run()
        assert abs(len(s.requests) - lam) < 4 * np.sqrt(lam)


def test_demand_follows_hourly_profile():
    """Busiest simulated hour should be one of the configured peak hours."""
    s = FleetSimulation(CFG.with_overrides(n_vehicles=0, daily_requests=30000), seed=5)
    s.run()
    hours = np.bincount([int(r.t_request // 60) for r in s.requests], minlength=24)
    top_config = set(np.argsort(CFG.city.hourly_share)[-3:])
    assert int(np.argmax(hours)) in top_config


def test_hungarian_never_worse_than_greedy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        r, v = rng.integers(1, 8), rng.integers(1, 8)
        cost = rng.uniform(1, 20, size=(r, v))
        g, b = greedy(cost), batched(cost)
        assert len(b) == len(g) == min(r, v)
        assert sum(cost[i, j] for i, j in b) <= sum(cost[i, j] for i, j in g) + 1e-9


def test_infeasible_pairs_never_matched():
    cost = np.array([[np.inf, 5.0], [np.inf, np.inf]])
    assert greedy(cost) == [(0, 1)]
    assert batched(cost) == [(0, 1)]
