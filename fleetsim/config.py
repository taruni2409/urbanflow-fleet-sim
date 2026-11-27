"""Scenario configuration.

Everything city-specific lives in a YAML file, so the same simulator can be
pointed at a different city by swapping the config - no code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CityConfig:
    name: str
    grid_size: int                  # city is grid_size x grid_size square zones
    zone_km: float                  # side length of one zone
    detour_factor: float            # road distance / straight-line grid distance
    daily_requests: float           # expected ride requests per day
    hourly_share: tuple             # 24 relative weights (normalised on load)
    speed_kmh_by_hour: tuple        # 24 average traffic speeds
    base_weight: float              # background demand everywhere
    hotspots: tuple                 # ({x, y, weight, spread}, ...)


@dataclass(frozen=True)
class FleetConfig:
    n_vehicles: int
    battery_kwh: float
    kwh_per_km: float
    charge_kw: float
    low_soc: float                  # go and charge below this state of charge
    target_soc: float               # charge up to this level
    reserve_soc: float              # never accept a trip that would end below this
    start_soc_min: float
    start_soc_max: float


@dataclass(frozen=True)
class PricingConfig:
    base_fare: float
    per_km: float
    per_min: float
    energy_cost_per_kwh: float


@dataclass(frozen=True)
class OpsConfig:
    duration_hours: float
    dispatch_interval_min: float    # how often the dispatcher matches cars to riders
    max_wait_min: float             # rider cancels if not matched within this
    max_pickup_min: float           # never send a car further away than this
    policy: str                     # "greedy" or "batched"


@dataclass(frozen=True)
class SimConfig:
    city: CityConfig
    fleet: FleetConfig
    pricing: PricingConfig
    ops: OpsConfig
    charging_hubs: tuple            # ({x, y, points}, ...)

    def with_overrides(self, n_vehicles: int | None = None, policy: str | None = None,
                       daily_requests: float | None = None) -> "SimConfig":
        cfg = self
        if n_vehicles is not None:
            cfg = replace(cfg, fleet=replace(cfg.fleet, n_vehicles=n_vehicles))
        if policy is not None:
            cfg = replace(cfg, ops=replace(cfg.ops, policy=policy))
        if daily_requests is not None:
            cfg = replace(cfg, city=replace(cfg.city, daily_requests=daily_requests))
        return cfg


def _tuplify(d: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def load_config(path: str | Path) -> SimConfig:
    raw = yaml.safe_load(Path(path).read_text())
    city = CityConfig(**_tuplify(raw["city"]))
    if len(city.hourly_share) != 24 or len(city.speed_kmh_by_hour) != 24:
        raise ValueError("hourly_share and speed_kmh_by_hour need 24 values each")
    return SimConfig(
        city=city,
        fleet=FleetConfig(**raw["fleet"]),
        pricing=PricingConfig(**raw["pricing"]),
        ops=OpsConfig(**raw["ops"]),
        charging_hubs=tuple(raw["charging_hubs"]),
    )
