"""In-process smoke checks for strategy.start with mock Grid."""

from __future__ import annotations

import copy

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from phase1_flower.mobility import IDMRoadMobility
from phase1_flower.models import build_model
from phase1_flower.strategy_mavfl import DelayParams, MAVFLStrategy


class MockGrid:
    """Minimal Grid implementation for strategy.start smoke checks."""

    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = node_ids

    def get_node_ids(self):
        return self._node_ids

    def send_and_receive(self, messages, timeout: float):
        replies = []
        for msg in list(messages):
            if msg.metadata.message_type.startswith("train"):
                replies.append(self._train_reply(msg))
            elif msg.metadata.message_type.startswith("evaluate"):
                replies.append(self._evaluate_reply(msg))
        return replies

    def _train_reply(self, msg: Message) -> Message:
        model_state = msg.content["arrays"].to_torch_state_dict()
        client_state = copy.deepcopy(model_state)
        scale = 1.0 + 0.001 * (msg.metadata.dst_node_id % 7)
        for key, value in client_state.items():
            client_state[key] = value * scale

        content = RecordDict(
            {
                "arrays": ArrayRecord(client_state),
                "metrics": MetricRecord(
                    {
                        "num-examples": 64,
                        "train-loss": 1.1,
                        "train-accuracy": 0.55,
                    }
                ),
            }
        )
        return Message(content=content, reply_to=msg)

    def _evaluate_reply(self, msg: Message) -> Message:
        content = RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "num-examples": 64,
                        "eval-loss": 1.0,
                        "eval-accuracy": 0.6,
                    }
                )
            }
        )
        return Message(content=content, reply_to=msg)


def run_fedavg_smoke() -> None:
    model = build_model("cifar10_resnet18")
    strategy = FedAvg(
        fraction_train=0.5,
        fraction_evaluate=0.5,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
    )
    result = strategy.start(
        grid=MockGrid(node_ids=list(range(10))),
        initial_arrays=ArrayRecord(model.state_dict()),
        num_rounds=2,
        train_config=ConfigRecord({"lr": 0.01, "local-epochs": 1, "task": "cifar10_resnet18"}),
        evaluate_config=ConfigRecord({"task": "cifar10_resnet18"}),
    )
    assert result.arrays is not None
    assert len(result.train_metrics_clientapp) == 2


def run_mavfl_smoke() -> None:
    model = build_model("gtsrb_lenet")
    mobility = IDMRoadMobility(
        seed=42,
        road_length_m=1000.0,
        num_zones=20,
        bs_height_m=25.0,
        desired_speed_kmh=60.0,
        max_accel_mps2=1.5,
        comfort_decel_mps2=2.0,
        min_gap_m=2.0,
        time_headway_s=1.5,
        accel_exponent=4.0,
        time_step_s=1.0,
        arrival_rate_hz=0.167,
        initial_active=5,
    )
    delay_params = DelayParams(
        bandwidth_hz=3_000_000.0,
        tx_power_dbm=23.0,
        noise_power_dbm=-114.0,
        bs_antenna_gain_db=6.0,
        path_loss_a=128.1,
        path_loss_b=37.6,
        gpu_frequency_ghz=1.3,
        gpu_cycles_per_bit=1.0,
        compute_normalization=1.0,
    )
    strategy = MAVFLStrategy(
        fraction_train=0.5,
        fraction_evaluate=0.5,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        ucb_exploration=1.0,
        ucb_discount=0.9,
        mobility=mobility,
        delay_params=delay_params,
        selection_size=5,
        alpha=0.6,
        model_size_bits=1_000_000.0,
        data_bits_per_client=64 * 32 * 32 * 3 * 8,
    )
    result = strategy.start(
        grid=MockGrid(node_ids=list(range(10))),
        initial_arrays=ArrayRecord(model.state_dict()),
        num_rounds=3,
        train_config=ConfigRecord({"lr": 0.01, "local-epochs": 1, "task": "gtsrb_lenet"}),
        evaluate_config=ConfigRecord({"task": "gtsrb_lenet"}),
    )
    assert result.arrays is not None
    assert len(result.train_metrics_clientapp) == 3


def main() -> None:
    run_fedavg_smoke()
    run_mavfl_smoke()
    print("smoke_strategy_start: PASS")


if __name__ == "__main__":
    main()
