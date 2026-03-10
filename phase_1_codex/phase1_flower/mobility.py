"""Open-road mobility model for MAVFL's one-way road-segment setting.

Design
------
The paper (arXiv:2410.10451) describes a one-way 1000 m road segment where
vehicles enter at one end, traverse the coverage zone, and permanently exit
at the other end.  New vehicles continuously arrive to replace departures.

This module implements that model with three disjoint node-lifecycle sets:

    _pending   -- vehicles that have not yet entered the road
    _active    -- vehicles currently on the road (0 <= position <= road_length_m)
    _exited    -- vehicles that have passed road_length_m; permanently done

Arrival model (Poisson)
-----------------------
Transit time at constant speed v across a road of length L is T = L / v.
To maintain N_target vehicles on the road at steady state, the required
arrival rate is:

    lambda = N_target / T = N_target * v / L

For the paper's defaults (v = 60 km/h = 16.67 m/s, L = 1000 m, N = 10):

    lambda = 10 * 16.67 / 1000 = 0.167 vehicles/second

Each call to ``advance(duration_s)`` draws arrivals from
Poisson(lambda * duration_s) and activates that many pending vehicles.

IDM car-following is applied only to _active vehicles.  Once a vehicle's
position exceeds road_length_m it is moved to _exited and never returns.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Vehicle kinematics for the road segment."""

    position_m: float
    speed_mps: float


class IDMRoadMobility:
    """IDM-style 1D mobility over an open one-way road segment.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    road_length_m : float
        Length of the coverage zone in metres.
    num_zones : int
        Number of equal-width zones for BS distance calculation.
    bs_height_m : float
        Base-station antenna height in metres.
    desired_speed_kmh : float
        Free-flow desired speed in km/h.
    max_accel_mps2, comfort_decel_mps2, min_gap_m, time_headway_s,
    accel_exponent : float
        Standard IDM parameters.
    time_step_s : float
        Internal integration step for the IDM simulation.
    arrival_rate_hz : float
        Mean Poisson arrival rate (vehicles/second).  Derived from
        steady-state target: lambda = N_target * v / L.
    initial_active : int
        Number of vehicles placed on the road at simulation start.
        These are drawn from the pending pool and spread uniformly
        across [0, road_length_m).
    """

    def __init__(
        self,
        *,
        seed: int,
        road_length_m: float,
        num_zones: int,
        bs_height_m: float,
        desired_speed_kmh: float,
        max_accel_mps2: float,
        comfort_decel_mps2: float,
        min_gap_m: float,
        time_headway_s: float,
        accel_exponent: float,
        time_step_s: float,
        arrival_rate_hz: float,
        initial_active: int,
    ) -> None:
        self.rng = random.Random(seed)
        self.road_length_m = road_length_m
        self.num_zones = num_zones
        self.bs_height_m = bs_height_m
        self.desired_speed_mps = desired_speed_kmh / 3.6
        self.max_accel_mps2 = max_accel_mps2
        self.comfort_decel_mps2 = comfort_decel_mps2
        self.min_gap_m = min_gap_m
        self.time_headway_s = time_headway_s
        self.accel_exponent = accel_exponent
        self.time_step_s = time_step_s
        self.arrival_rate_hz = arrival_rate_hz
        self.initial_active = initial_active

        # Vehicle kinematics (only for active vehicles).
        self._states: dict[int, VehicleState] = {}

        # Lifecycle sets -- populated lazily on first call to _ensure_pool().
        self._pending: list[int] = []  # ordered queue; front = next to arrive
        self._active: set[int] = set()
        self._exited: set[int] = set()
        self._pool_initialised: bool = False

    # ------------------------------------------------------------------
    # Pool initialisation
    # ------------------------------------------------------------------

    def _ensure_pool(self, node_ids: list[int]) -> None:
        """Partition all known node_ids into lifecycle sets on first call.

        The first ``initial_active`` nodes (after shuffling) are placed on the
        road with uniform random positions.  The rest go into the pending queue.
        """
        if self._pool_initialised:
            return
        self._pool_initialised = True

        # Shuffle so the initial placement is not biased by node-id ordering.
        shuffled = list(node_ids)
        self.rng.shuffle(shuffled)

        n_place = min(self.initial_active, len(shuffled))
        initial_ids = shuffled[:n_place]
        rest_ids = shuffled[n_place:]

        # Place initial vehicles uniformly across [0, road_length_m).
        for node_id in initial_ids:
            pos = self.rng.uniform(0.0, self.road_length_m)
            speed = self.rng.uniform(0.7, 1.1) * self.desired_speed_mps
            self._states[node_id] = VehicleState(position_m=pos, speed_mps=speed)
            self._active.add(node_id)

        # Remaining nodes are pending (not yet on the road).
        self._pending = rest_ids

    # ------------------------------------------------------------------
    # Arrivals
    # ------------------------------------------------------------------

    def _activate_arrivals(self, duration_s: float) -> list[int]:
        """Draw Poisson arrivals and move pending -> active.

        Returns the list of newly activated node_ids (may be empty).
        """
        if duration_s <= 0.0 or not self._pending:
            return []

        # Poisson draw: number of arrivals in this time window.
        expected = self.arrival_rate_hz * duration_s
        n_arrivals = min(self._poisson_draw(expected), len(self._pending))

        newly_active: list[int] = []
        for _ in range(n_arrivals):
            node_id = self._pending.pop(0)
            # New arrivals enter at position 0 with random speed around v_desired.
            speed = self.rng.uniform(0.7, 1.1) * self.desired_speed_mps
            self._states[node_id] = VehicleState(position_m=0.0, speed_mps=speed)
            self._active.add(node_id)
            newly_active.append(node_id)

        return newly_active

    def _poisson_draw(self, lam: float) -> int:
        """Draw from Poisson(lam) using inverse-CDF (Knuth algorithm).

        For the expected range of lam (<50) this is efficient and avoids
        importing numpy just for one RNG call.
        """
        if lam <= 0.0:
            return 0
        exp_neg_lam = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= self.rng.random()
            if p <= exp_neg_lam:
                return k
            k += 1

    # ------------------------------------------------------------------
    # IDM car-following
    # ------------------------------------------------------------------

    def _idm_accel(self, speed: float, gap: float, delta_v: float) -> float:
        """Compute IDM acceleration for one vehicle."""
        desired_gap = self.min_gap_m + max(
            0.0,
            speed * self.time_headway_s
            + (speed * delta_v)
            / (2.0 * math.sqrt(self.max_accel_mps2 * self.comfort_decel_mps2)),
        )
        free_flow = (speed / max(self.desired_speed_mps, 1e-6)) ** self.accel_exponent
        interaction = (desired_gap / max(gap, 1e-3)) ** 2
        return self.max_accel_mps2 * (1.0 - free_flow - interaction)

    def _step_once(self, dt: float) -> None:
        """Advance all active vehicles by one IDM time step.

        Vehicles whose position exceeds road_length_m are moved to _exited.
        """
        active_ids = list(self._active)
        if not active_ids:
            return

        ordered = sorted(active_ids, key=lambda nid: self._states[nid].position_m)

        # Compute gaps and delta-v for each vehicle.
        gaps: dict[int, float] = {}
        delta_v: dict[int, float] = {}
        for idx, node_id in enumerate(ordered):
            state = self._states[node_id]
            if idx == len(ordered) - 1:
                # Leader has no vehicle ahead -- free road.
                gaps[node_id] = self.road_length_m
                delta_v[node_id] = 0.0
            else:
                leader = self._states[ordered[idx + 1]]
                gaps[node_id] = max(leader.position_m - state.position_m, 1e-3)
                delta_v[node_id] = state.speed_mps - leader.speed_mps

        # Update kinematics.
        newly_exited: list[int] = []
        for node_id in ordered:
            state = self._states[node_id]
            accel = self._idm_accel(state.speed_mps, gaps[node_id], delta_v[node_id])
            state.speed_mps = max(0.0, state.speed_mps + accel * dt)
            state.position_m += state.speed_mps * dt

            if state.position_m >= self.road_length_m:
                newly_exited.append(node_id)

        # Move exited vehicles out of the active set.
        for node_id in newly_exited:
            self._active.discard(node_id)
            self._exited.add(node_id)
            del self._states[node_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advance(self, duration_s: float, node_ids: list[int]) -> None:
        """Advance mobility for all vehicles by duration_s.

        This is the main tick method called once per FL round from the
        strategy's ``aggregate_train``.  It:

        1. Ensures the vehicle pool is initialised (first call only).
        2. Runs IDM integration for ``duration_s``, exiting vehicles that
           pass the end of the road.
        3. Draws Poisson arrivals and activates pending vehicles.

        Parameters
        ----------
        duration_s : float
            Simulated time to advance (typically the round delay).
        node_ids : list[int]
            All node IDs known to the simulation (from ``grid.get_node_ids()``).
            Used only on the first call to initialise the pool.
        """
        self._ensure_pool(node_ids)
        if duration_s <= 0.0:
            return

        # IDM integration in small steps.
        steps = max(1, int(math.ceil(duration_s / self.time_step_s)))
        dt = duration_s / steps
        for _ in range(steps):
            self._step_once(dt)

        # Activate new arrivals after the time window.
        self._activate_arrivals(duration_s)

    def predict_position(self, node_id: int, duration_s: float) -> float:
        """Predict position after duration_s using constant-speed extrapolation.

        Returns road_length_m + 1 for exited or unknown vehicles so that
        callers treating position > road_length_m as "exited" work correctly.
        """
        if node_id not in self._states:
            return self.road_length_m + 1.0
        state = self._states[node_id]
        if duration_s <= 0.0:
            return state.position_m
        return state.position_m + state.speed_mps * duration_s

    def positions(self, node_ids: list[int]) -> dict[int, float]:
        """Return current positions for the given node_ids.

        Non-active nodes are silently omitted (the caller should only pass
        eligible node_ids obtained from ``eligible_nodes``).
        """
        return {
            nid: self._states[nid].position_m
            for nid in node_ids
            if nid in self._states
        }

    def speeds(self, node_ids: list[int]) -> dict[int, float]:
        """Return current speeds for the given node_ids."""
        return {
            nid: self._states[nid].speed_mps
            for nid in node_ids
            if nid in self._states
        }

    def eligible_nodes(self, node_ids: list[int]) -> list[int]:
        """Return node_ids that are currently active on the road.

        A node is eligible iff it is in the _active set and its position is
        within [0, road_length_m].  The ``node_ids`` argument is used only on
        the first call to initialise the pool; after that, eligibility is
        determined purely by the lifecycle sets.
        """
        self._ensure_pool(node_ids)
        return [nid for nid in self._active if nid in self._states]

    def zone_index(self, position_m: float) -> int:
        """Return the zone index (0-based) for a given position."""
        zone_len = self.road_length_m / self.num_zones
        return min(self.num_zones - 1, max(0, int(position_m / max(zone_len, 1e-6))))

    def distance_to_bs(self, position_m: float) -> float:
        """Euclidean distance from position to the base station.

        The BS is at the centre of the road at height ``bs_height_m``.
        """
        zone_len = self.road_length_m / self.num_zones
        zone_center = (self.zone_index(position_m) + 0.5) * zone_len
        horizontal = abs(zone_center - self.road_length_m / 2.0)
        return math.sqrt(horizontal**2 + self.bs_height_m**2)

    def time_to_exit(self, position_m: float, speed_mps: float) -> float:
        """Estimated time until the vehicle exits the road segment."""
        if position_m >= self.road_length_m:
            return 0.0
        if speed_mps <= 0.0:
            return float("inf")
        return max((self.road_length_m - position_m) / speed_mps, 0.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pool_summary(self) -> dict[str, int]:
        """Return counts for each lifecycle set (useful for logging)."""
        return {
            "pending": len(self._pending),
            "active": len(self._active),
            "exited": len(self._exited),
        }
