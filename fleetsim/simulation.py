"""Discrete-event simulation of an electric ride-hailing fleet (SimPy).

Processes running concurrently on one simulated clock:
  demand      - riders appear as a non-homogeneous Poisson process (thinning)
  patience    - one per rider: cancels the request after max_wait_min
  dispatcher  - every dispatch_interval_min, matches waiting riders to idle cars
  serve       - one per assigned trip: drive to pickup, drive to drop-off, maybe charge
  monitor     - snapshots fleet state every 15 minutes

Vehicle states:  idle -> to_pickup -> occupied -> idle
                                         \\-> to_charger -> charging -> idle
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy

from .city import City
from .config import SimConfig
from .dispatch import POLICIES

SNAPSHOT_EVERY_MIN = 15


@dataclass
class Request:
    id: int
    t_request: float
    origin: tuple
    dest: tuple
    trip_km: float
    km_dest_to_hub: float           # so we never strand a car after drop-off
    status: str = "waiting"         # waiting -> assigned -> completed | lost
    t_assigned: float | None = None
    t_pickup: float | None = None
    t_dropoff: float | None = None
    fare: float = 0.0


@dataclass
class Vehicle:
    id: int
    pos: tuple
    soc_kwh: float
    state: str = "idle"
    revenue: float = 0.0
    energy_cost: float = 0.0
    trips: int = 0
    occupied_min: float = 0.0
    occupied_km: float = 0.0
    empty_min: float = 0.0          # driving to pickups and to chargers
    empty_km: float = 0.0
    charging_min: float = 0.0       # queueing + plugged in
    charger_queue_min: float = 0.0
    charge_sessions: int = 0


@dataclass
class Hub:
    pos: tuple
    resource: simpy.Resource


class FleetSimulation:
    def __init__(self, cfg: SimConfig, seed: int = 0):
        if cfg.ops.policy not in POLICIES:
            raise ValueError(f"unknown policy {cfg.ops.policy!r}; choose from {list(POLICIES)}")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.city = City(cfg.city)
        self.match = POLICIES[cfg.ops.policy]

        self.hubs = [Hub((h["x"], h["y"]), simpy.Resource(self.env, capacity=h["points"]))
                     for h in cfg.charging_hubs]
        self.hub_xy = np.array([h.pos for h in self.hubs], dtype=float)

        f = cfg.fleet
        self.vehicles = []
        for i in range(f.n_vehicles):
            zone = self.rng.choice(len(self.city.zone_weight), p=self.city.zone_weight)
            soc = self.rng.uniform(f.start_soc_min, f.start_soc_max) * f.battery_kwh
            self.vehicles.append(Vehicle(i, tuple(self.city.zone_xy[zone]), soc))

        self.requests: list[Request] = []
        self.waiting: list[Request] = []
        self.timeline: list[dict] = []

        self.env.process(self._demand())
        self.env.process(self._dispatcher())
        self.env.process(self._monitor())

    # ------------------------------------------------------------------ demand
    def _demand(self):
        """Non-homogeneous Poisson arrivals via thinning (Lewis & Shedler).

        Propose arrivals at the PEAK rate, then keep each one with probability
        rate(t) / peak. The kept arrivals follow the time-varying rate exactly.
        """
        lam_max = self.city.peak_rate_per_min
        if lam_max <= 0:
            return
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / lam_max))
            if self.rng.random() < self.city.rate_per_min(self.env.now) / lam_max:
                origin, dest = self.city.sample_trip(self.rng)
                req = Request(
                    id=len(self.requests),
                    t_request=self.env.now,
                    origin=origin,
                    dest=dest,
                    trip_km=float(self.city.distance_km(origin, dest)),
                    km_dest_to_hub=float(self.city.distance_km(self.hub_xy, dest).min()),
                )
                self.requests.append(req)
                self.waiting.append(req)
                self.env.process(self._patience(req))

    def _patience(self, req: Request):
        yield self.env.timeout(self.cfg.ops.max_wait_min)
        if req.status == "waiting":
            req.status = "lost"
            self.waiting.remove(req)

    # -------------------------------------------------------------- dispatcher
    def _dispatcher(self):
        f, ops = self.cfg.fleet, self.cfg.ops
        while True:
            yield self.env.timeout(ops.dispatch_interval_min)
            idle = [v for v in self.vehicles if v.state == "idle"]
            if not self.waiting or not idle:
                continue

            riders = self.waiting
            r_xy = np.array([r.origin for r in riders], dtype=float)       # (R, 2)
            v_xy = np.array([v.pos for v in idle], dtype=float)            # (V, 2)
            pickup_km = self.city.distance_km(r_xy[:, None, :], v_xy[None, :, :])
            pickup_min = self.city.travel_min(pickup_km, self.env.now)

            # Energy check: pickup + trip + reach a charger, keeping a reserve.
            need_km = pickup_km + np.array([r.trip_km + r.km_dest_to_hub for r in riders])[:, None]
            soc = np.array([v.soc_kwh for v in idle])[None, :]
            enough = soc - need_km * f.kwh_per_km >= f.reserve_soc * f.battery_kwh

            cost = np.where(enough & (pickup_min <= ops.max_pickup_min), pickup_min, np.inf)
            pairs = self.match(cost)

            matched = set()
            for i, j in pairs:
                req, veh = riders[i], idle[j]
                req.status, req.t_assigned = "assigned", self.env.now
                veh.state = "to_pickup"
                matched.add(req.id)
                self.env.process(self._serve(veh, req))
            self.waiting = [r for r in riders if r.id not in matched]

    # ------------------------------------------------------------------- trips
    def _drive(self, veh: Vehicle, dest: tuple, occupied: bool):
        km = float(self.city.distance_km(veh.pos, dest))
        minutes = float(self.city.travel_min(km, self.env.now))
        yield self.env.timeout(minutes)
        veh.soc_kwh -= km * self.cfg.fleet.kwh_per_km
        veh.pos = dest
        if occupied:
            veh.occupied_km += km
            veh.occupied_min += minutes
        else:
            veh.empty_km += km
            veh.empty_min += minutes
        return minutes

    def _serve(self, veh: Vehicle, req: Request):
        p = self.cfg.pricing
        yield from self._drive(veh, req.origin, occupied=False)
        req.t_pickup = self.env.now
        veh.state = "occupied"

        ride_min = yield from self._drive(veh, req.dest, occupied=True)
        req.t_dropoff = self.env.now
        req.fare = p.base_fare + p.per_km * req.trip_km + p.per_min * ride_min
        req.status = "completed"
        veh.revenue += req.fare
        veh.trips += 1

        if veh.soc_kwh < self.cfg.fleet.low_soc * self.cfg.fleet.battery_kwh:
            yield from self._charge(veh)
        veh.state = "idle"

    def _charge(self, veh: Vehicle):
        f = self.cfg.fleet
        dists = self.city.distance_km(self.hub_xy, np.asarray(veh.pos, dtype=float))
        hub = self.hubs[int(np.argmin(dists))]

        veh.state = "to_charger"
        yield from self._drive(veh, hub.pos, occupied=False)

        veh.state = "charging"
        t0 = self.env.now
        with hub.resource.request() as plug:
            yield plug                                        # queue for a free point
            veh.charger_queue_min += self.env.now - t0
            kwh = max(0.0, f.target_soc * f.battery_kwh - veh.soc_kwh)
            yield self.env.timeout(kwh / f.charge_kw * 60.0)
        veh.soc_kwh += kwh
        veh.energy_cost += kwh * self.cfg.pricing.energy_cost_per_kwh
        veh.charging_min += self.env.now - t0
        veh.charge_sessions += 1

    # ----------------------------------------------------------------- monitor
    def _monitor(self):
        while True:
            states = [v.state for v in self.vehicles]
            self.timeline.append({
                "t_min": self.env.now,
                "requests_so_far": len(self.requests),
                "waiting": len(self.waiting),
                "idle": states.count("idle"),
                "serving": states.count("to_pickup") + states.count("occupied"),
                "charging": states.count("to_charger") + states.count("charging"),
            })
            yield self.env.timeout(SNAPSHOT_EVERY_MIN)

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        self.env.run(until=self.cfg.ops.duration_hours * 60)
        return self.kpis()

    def kpis(self) -> dict:
        vs = self.vehicles
        n = max(1, len(vs))
        hours = self.cfg.ops.duration_hours
        fleet_hours = n * hours
        done = [r for r in self.requests if r.status == "completed"]
        lost = [r for r in self.requests if r.status == "lost"]
        waits = np.array([r.t_pickup - r.t_request for r in self.requests if r.t_pickup is not None])
        revenue = sum(v.revenue for v in vs)
        energy = sum(v.energy_cost for v in vs)
        occ_km = sum(v.occupied_km for v in vs)
        empty_km = sum(v.empty_km for v in vs)
        sessions = sum(v.charge_sessions for v in vs)
        resolved = len(done) + len(lost)
        return {
            "requests": len(self.requests),
            "completed": len(done),
            "lost": len(lost),
            "fulfilment": len(done) / resolved if resolved else float("nan"),
            "revenue_per_vehicle_hour": revenue / fleet_hours,
            "net_per_vehicle_hour": (revenue - energy) / fleet_hours,
            "trips_per_vehicle_hour": len(done) / fleet_hours,
            "utilisation": sum(v.occupied_min for v in vs) / (fleet_hours * 60),
            "empty_km_share": empty_km / (empty_km + occ_km) if (empty_km + occ_km) else float("nan"),
            "charging_time_share": sum(v.charging_min for v in vs) / (fleet_hours * 60),
            "mean_charger_queue_min": sum(v.charger_queue_min for v in vs) / sessions if sessions else 0.0,
            "mean_wait_min": float(waits.mean()) if waits.size else float("nan"),
            "p90_wait_min": float(np.percentile(waits, 90)) if waits.size else float("nan"),
            "min_soc_fraction": min((v.soc_kwh for v in vs), default=0) / self.cfg.fleet.battery_kwh,
        }


def simulate(cfg: SimConfig, seed: int = 0) -> dict:
    return FleetSimulation(cfg, seed).run()
