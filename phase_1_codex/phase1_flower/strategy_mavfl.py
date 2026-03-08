"""Mobility-aware strategies aligned with the MAVFL paper."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from logging import INFO

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common import ArrayRecord, MessageType, MetricRecord, log
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
)

from .mobility import IDMRoadMobility


@dataclass
class DelayParams:
    bandwidth_hz: float
    tx_power_dbm: float
    noise_power_dbm: float
    bs_antenna_gain_db: float
    path_loss_a: float
    path_loss_b: float
    gpu_frequency_ghz: float
    gpu_cycles_per_bit: float
    compute_normalization: float


@dataclass
class RoundState:
    all_nodes: list[int]
    selected_nodes: list[int]
    eligible_nodes: list[int]
    per_node_time_s: dict[int, float]
    round_delay_s: float
    success_nodes: set[int]
    success_ratio: float


def _db_to_linear(db_val: float) -> float:
    return 10 ** (db_val / 10.0)


def _uplink_rate(
    distance_m: float,
    bandwidth_hz: float,
    tx_power_dbm: float,
    noise_power_dbm: float,
    bs_antenna_gain_db: float,
    path_loss_a: float,
    path_loss_b: float,
) -> float:
    distance_km = max(distance_m / 1000.0, 1e-6)
    path_loss_db = path_loss_a + path_loss_b * math.log10(distance_km)
    channel_gain_linear = _db_to_linear(bs_antenna_gain_db - path_loss_db)
    tx_power_mw = _db_to_linear(tx_power_dbm)
    noise_mw = _db_to_linear(noise_power_dbm)
    snr = max(tx_power_mw * channel_gain_linear / max(noise_mw, 1e-12), 1e-12)
    return bandwidth_hz * math.log2(1.0 + snr)


class MobilitySelectionStrategy(FedAvg):
    """Base strategy that plugs a mobility-aware selection policy into FedAvg."""

    def __init__(
        self,
        *args,
        mobility: IDMRoadMobility,
        delay_params: DelayParams,
        selection_size: int,
        alpha: float,
        model_size_bits: float,
        data_bits_per_client: float,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mobility = mobility
        self.delay_params = delay_params
        self.selection_size = selection_size
        self.alpha = alpha
        self.model_size_bits = model_size_bits
        self.data_bits_per_client = data_bits_per_client
        self._rng = random.Random(seed)
        self._round_state: RoundState | None = None
        self._min_delay: float | None = None
        self._max_delay: float | None = None

    def _select_nodes(self, eligible: list[int], server_round: int) -> list[int]:
        raise NotImplementedError

    def _compute_round_state(
        self, server_round: int, all_nodes: list[int], eligible: list[int], selected: list[int]
    ) -> RoundState:
        if not selected:
            return RoundState(all_nodes, selected, eligible, {}, 0.0, set(), 0.0)

        bandwidth_per_node = self.delay_params.bandwidth_hz / max(len(selected), 1)
        positions = self.mobility.positions(selected)
        speeds = self.mobility.speeds(selected)
        per_node_time: dict[int, float] = {}
        success_nodes: set[int] = set()

        for node_id in selected:
            distance_m = self.mobility.distance_to_bs(positions[node_id])
            uplink_rate = _uplink_rate(
                distance_m=distance_m,
                bandwidth_hz=bandwidth_per_node,
                tx_power_dbm=self.delay_params.tx_power_dbm,
                noise_power_dbm=self.delay_params.noise_power_dbm,
                bs_antenna_gain_db=self.delay_params.bs_antenna_gain_db,
                path_loss_a=self.delay_params.path_loss_a,
                path_loss_b=self.delay_params.path_loss_b,
            )
            comm_time_s = self.model_size_bits / max(uplink_rate, 1e-9)
            gpu_freq_hz = self.delay_params.gpu_frequency_ghz * 1e9
            compute_time_s = (
                self.data_bits_per_client
                * self.delay_params.gpu_cycles_per_bit
                / max(self.delay_params.compute_normalization * gpu_freq_hz, 1e-9)
            )
            total_time = comm_time_s + compute_time_s
            per_node_time[node_id] = total_time

            time_to_exit = self.mobility.time_to_exit(positions[node_id], speeds[node_id])
            if time_to_exit >= total_time:
                success_nodes.add(node_id)

        round_delay = max(per_node_time.values()) if per_node_time else 0.0
        success_ratio = len(success_nodes) / max(len(selected), 1)
        return RoundState(all_nodes, selected, eligible, per_node_time, round_delay, success_nodes, success_ratio)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> list[Message]:
        all_nodes = list(grid.get_node_ids())
        if len(all_nodes) < self.min_available_nodes:
            log(INFO, "configure_train: waiting for min available nodes")
            return []

        eligible_nodes = self.mobility.eligible_nodes(all_nodes)
        if not eligible_nodes:
            eligible_nodes = all_nodes

        sample_size = min(self.selection_size, len(eligible_nodes))
        selected_nodes = self._select_nodes(eligible_nodes, server_round)[:sample_size]

        config["server-round"] = server_round
        config["selected-count"] = len(selected_nodes)
        config["eligible-count"] = len(eligible_nodes)

        self._round_state = self._compute_round_state(
            server_round, all_nodes, eligible_nodes, selected_nodes
        )
        if self._round_state.round_delay_s > 0.0:
            self._min_delay = (
                self._round_state.round_delay_s
                if self._min_delay is None
                else min(self._min_delay, self._round_state.round_delay_s)
            )
            self._max_delay = (
                self._round_state.round_delay_s
                if self._max_delay is None
                else max(self._max_delay, self._round_state.round_delay_s)
            )

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        log(
            INFO,
            "configure_train: selected %d/%d eligible nodes (%d total)",
            len(selected_nodes),
            len(eligible_nodes),
            len(all_nodes),
        )
        return list(self._construct_messages(record, selected_nodes, MessageType.TRAIN))

    def _utility(self, success_ratio: float, round_delay_s: float) -> float:
        if self._min_delay is None or self._max_delay is None or self._max_delay <= self._min_delay:
            normalized_delay = 0.0
        else:
            normalized_delay = (round_delay_s - self._min_delay) / (self._max_delay - self._min_delay)
        return self.alpha * success_ratio - (1.0 - self.alpha) * normalized_delay

    def _post_round_update(self, server_round: int, utility: float) -> None:
        del server_round
        del utility

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        valid_replies, _ = self._check_and_log_replies(replies_list, is_train=True)

        state = self._round_state
        if state is None:
            return None, None

        filtered_replies = [
            msg for msg in valid_replies if msg.metadata.src_node_id in state.success_nodes
        ]

        arrays: ArrayRecord | None = None
        metrics: MetricRecord | None = None
        if filtered_replies:
            reply_contents = [msg.content for msg in filtered_replies]
            # Paper Eq. 3: equal weighting among successful vehicles
            for rc in reply_contents:
                for mr in rc.metric_records.values():
                    mr[self.weighted_by_key] = 1.0
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)

            utility = self._utility(state.success_ratio, state.round_delay_s)
            metrics["mavfl-round-delay-s"] = state.round_delay_s
            metrics["mavfl-success-ratio"] = state.success_ratio
            metrics["mavfl-utility"] = utility
            metrics["mavfl-round"] = float(server_round)

            self._post_round_update(server_round, utility)

        if state.round_delay_s > 0.0:
            self.mobility.advance(state.round_delay_s, state.all_nodes)

        return arrays, metrics


class MAVFLStrategy(MobilitySelectionStrategy):
    """MAVFL UCB-based vehicle selection."""

    def __init__(
        self,
        *args,
        ucb_discount: float,
        ucb_exploration: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ucb_discount = ucb_discount
        self.ucb_exploration = ucb_exploration
        self.node_stats: dict[int, dict[str, float]] = {}

    def _ucb_score(self, node_id: int, total_pulls: float) -> float:
        stats = self.node_stats.get(node_id)
        if stats is None or stats["count"] <= 0:
            return float("inf")
        mean_reward = stats["utility_sum"] / stats["count"]
        bonus = self.ucb_exploration * math.sqrt(
            max(2.0 * math.log(max(total_pulls, 1.0)) / stats["count"], 0.0)
        )
        return mean_reward + bonus

    def _select_nodes(self, eligible: list[int], server_round: int) -> list[int]:
        if server_round == 1:
            return self._rng.sample(eligible, k=min(self.selection_size, len(eligible)))

        total_pulls = sum(stats["count"] for stats in self.node_stats.values()) + 1.0
        scored = [(node_id, self._ucb_score(node_id, total_pulls)) for node_id in eligible]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [node_id for node_id, _ in scored[: self.selection_size]]

    def _post_round_update(self, server_round: int, utility: float) -> None:
        for stats in self.node_stats.values():
            stats["count"] *= self.ucb_discount
            stats["utility_sum"] *= self.ucb_discount

        state = self._round_state
        if state is None:
            return
        for node_id in state.selected_nodes:
            stats = self.node_stats.setdefault(node_id, {"count": 0.0, "utility_sum": 0.0})
            stats["count"] += 1.0
            node_utility = utility if node_id in state.success_nodes else 0.0
            stats["utility_sum"] += node_utility


class CBSStrategy(MobilitySelectionStrategy):
    """Closest-to-BS selection (baseline)."""

    def _select_nodes(self, eligible: list[int], server_round: int) -> list[int]:
        del server_round
        positions = self.mobility.positions(eligible)
        scored = [
            (node_id, self.mobility.distance_to_bs(positions[node_id])) for node_id in eligible
        ]
        scored.sort(key=lambda item: item[1])
        return [node_id for node_id, _ in scored[: self.selection_size]]


class RBSStrategy(MobilitySelectionStrategy):
    """Longest-remaining-time selection (baseline)."""

    def _select_nodes(self, eligible: list[int], server_round: int) -> list[int]:
        del server_round
        positions = self.mobility.positions(eligible)
        speeds = self.mobility.speeds(eligible)
        scored = [
            (node_id, self.mobility.time_to_exit(positions[node_id], speeds[node_id]))
            for node_id in eligible
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [node_id for node_id, _ in scored[: self.selection_size]]


class RandomStrategy(MobilitySelectionStrategy):
    """Uniform random selection (baseline)."""

    def _select_nodes(self, eligible: list[int], server_round: int) -> list[int]:
        del server_round
        return self._rng.sample(eligible, k=min(self.selection_size, len(eligible)))
