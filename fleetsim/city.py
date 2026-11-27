"""The city: zones, where demand comes from, and how long it takes to drive.

Zones are a square grid (a stand-in for H3 hexagons or postcode areas).
Demand intensity is lambda(zone, hour) = daily_requests * hourly_share[hour] * zone_weight[zone].
"""
from __future__ import annotations

import numpy as np

from .config import CityConfig


class City:
    def __init__(self, cfg: CityConfig):
        self.cfg = cfg
        n = cfg.grid_size
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        w = np.full((n, n), cfg.base_weight, dtype=float)
        for h in cfg.hotspots:
            d2 = (xs - h["x"]) ** 2 + (ys - h["y"]) ** 2
            w += h["weight"] * np.exp(-d2 / (2 * h["spread"] ** 2))
        self.zone_xy = np.stack([xs.ravel(), ys.ravel()], axis=1)    # (Z, 2)
        self.zone_weight = (w / w.sum()).ravel()                      # sums to 1
        share = np.asarray(cfg.hourly_share, dtype=float)
        self.hourly_share = share / share.sum()
        self.speeds = np.asarray(cfg.speed_kmh_by_hour, dtype=float)

    # --- demand -------------------------------------------------------------
    def rate_per_min(self, t_min: float) -> float:
        """Total request rate (all zones) at simulated minute t."""
        return self.cfg.daily_requests * self.hourly_share[int(t_min // 60) % 24] / 60.0

    @property
    def peak_rate_per_min(self) -> float:
        return self.cfg.daily_requests * self.hourly_share.max() / 60.0

    def sample_trip(self, rng: np.random.Generator) -> tuple[tuple, tuple]:
        """Origin and destination drawn independently from zone popularity."""
        o, d = rng.choice(len(self.zone_weight), size=2, p=self.zone_weight)
        return tuple(self.zone_xy[o]), tuple(self.zone_xy[d])

    # --- movement -----------------------------------------------------------
    def distance_km(self, a, b) -> np.ndarray | float:
        """Road distance between zone centres; works on scalars or numpy arrays.

        a, b are (..., 2) arrays or (x, y) tuples. Same-zone trips get half a zone.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        grid = np.abs(a[..., 0] - b[..., 0]) + np.abs(a[..., 1] - b[..., 1])
        grid = np.maximum(grid, 0.5)
        return grid * self.cfg.zone_km * self.cfg.detour_factor

    def speed_kmh(self, t_min: float) -> float:
        return float(self.speeds[int(t_min // 60) % 24])

    def travel_min(self, km, t_min: float):
        """Driving time at the traffic speed of the hour the drive starts in."""
        return km / self.speed_kmh(t_min) * 60.0
