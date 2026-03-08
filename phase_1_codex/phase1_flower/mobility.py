"""Mobility and geometry utilities aligned with MAVFL's road-segment setting."""

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
    """IDM-style 1D mobility over a single road segment."""

    def __init__(
        self,
        *,
        seed: int,
        road_length_m: float,
        road_loop_m: float,
        num_zones: int,
        bs_height_m: float,
        desired_speed_kmh: float,
        max_accel_mps2: float,
        comfort_decel_mps2: float,
        min_gap_m: float,
        time_headway_s: float,
        accel_exponent: float,
        time_step_s: float,
    ) -> None:
        self.rng = random.Random(seed)
        self.road_length_m = road_length_m
        self.road_loop_m = road_loop_m
        self.num_zones = num_zones
        self.bs_height_m = bs_height_m
        self.desired_speed_mps = desired_speed_kmh / 3.6
        self.max_accel_mps2 = max_accel_mps2
        self.comfort_decel_mps2 = comfort_decel_mps2
        self.min_gap_m = min_gap_m
        self.time_headway_s = time_headway_s
        self.accel_exponent = accel_exponent
        self.time_step_s = time_step_s
        self._states: dict[int, VehicleState] = {}

    def _ensure_states(self, node_ids: list[int]) -> None:
        for node_id in node_ids:
            if node_id in self._states:
                continue
            position = self.rng.uniform(0.0, self.road_length_m)
            speed = self.rng.uniform(0.7, 1.1) * self.desired_speed_mps
            self._states[node_id] = VehicleState(position_m=position, speed_mps=speed)

    def _idm_accel(self, speed: float, gap: float, delta_v: float) -> float:
        desired_gap = self.min_gap_m + max(
            0.0,
            speed * self.time_headway_s
            + (speed * delta_v) / (2.0 * math.sqrt(self.max_accel_mps2 * self.comfort_decel_mps2)),
        )
        free_flow = (speed / max(self.desired_speed_mps, 1e-6)) ** self.accel_exponent
        interaction = (desired_gap / max(gap, 1e-3)) ** 2
        return self.max_accel_mps2 * (1.0 - free_flow - interaction)

    def _step_once(self, node_ids: list[int], dt: float) -> None:
        if not node_ids:
            return
        ordered = sorted(node_ids, key=lambda nid: self._states[nid].position_m)
        gaps: dict[int, float] = {}
        delta_v: dict[int, float] = {}
        for idx, node_id in enumerate(ordered):
            state = self._states[node_id]
            if idx == len(ordered) - 1:
                gaps[node_id] = self.road_length_m
                delta_v[node_id] = 0.0
            else:
                leader = self._states[ordered[idx + 1]]
                gaps[node_id] = max(leader.position_m - state.position_m, 1e-3)
                delta_v[node_id] = state.speed_mps - leader.speed_mps

        for node_id in ordered:
            state = self._states[node_id]
            accel = self._idm_accel(state.speed_mps, gaps[node_id], delta_v[node_id])
            state.speed_mps = max(0.0, state.speed_mps + accel * dt)
            state.position_m += state.speed_mps * dt
            if state.position_m >= self.road_loop_m:
                state.position_m -= self.road_loop_m

    def advance(self, duration_s: float, node_ids: list[int]) -> None:
        """Advance mobility for all vehicles by duration_s."""
        self._ensure_states(node_ids)
        if duration_s <= 0.0:
            return
        steps = max(1, int(math.ceil(duration_s / self.time_step_s)))
        dt = duration_s / steps
        for _ in range(steps):
            self._step_once(node_ids, dt)

    def predict_position(self, node_id: int, duration_s: float) -> float:
        """Predict position after duration_s using local copy."""
        self._ensure_states([node_id])
        state = self._states[node_id]
        pos = state.position_m
        speed = state.speed_mps
        if duration_s <= 0.0:
            return pos
        pos = (pos + speed * duration_s) % self.road_loop_m
        return pos

    def positions(self, node_ids: list[int]) -> dict[int, float]:
        self._ensure_states(node_ids)
        return {node_id: self._states[node_id].position_m for node_id in node_ids}

    def speeds(self, node_ids: list[int]) -> dict[int, float]:
        self._ensure_states(node_ids)
        return {node_id: self._states[node_id].speed_mps for node_id in node_ids}

    def eligible_nodes(self, node_ids: list[int]) -> list[int]:
        self._ensure_states(node_ids)
        return [
            node_id
            for node_id in node_ids
            if 0.0 <= self._states[node_id].position_m <= self.road_length_m
        ]

    def zone_index(self, position_m: float) -> int:
        zone_len = self.road_length_m / self.num_zones
        return min(self.num_zones - 1, max(0, int(position_m / max(zone_len, 1e-6))))

    def distance_to_bs(self, position_m: float) -> float:
        zone_len = self.road_length_m / self.num_zones
        zone_center = (self.zone_index(position_m) + 0.5) * zone_len
        horizontal = abs(zone_center - self.road_length_m / 2.0)
        return math.sqrt(horizontal**2 + self.bs_height_m**2)

    def time_to_exit(self, position_m: float, speed_mps: float) -> float:
        if position_m > self.road_length_m:
            return 0.0
        if speed_mps <= 0.0:
            return float("inf")
        return max((self.road_length_m - position_m) / speed_mps, 0.0)
