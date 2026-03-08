# Phase 1 Codex (Fresh Implementation)

Fresh Flower Phase 1 implementation built from scratch for:
- Week 1: reproducible baseline (Flower + PyTorch, strategy.start smoke run)
- Week 2: MAVFL-style round-wise subset selection with UCB updates + baselines (CBS/RBS/Random)
- Week 3: benchmark scaffolding for CIFAR-10/ResNet-18, GTSRB/LeNet, and simplified mobility

## Environment

This project is pinned in `pyproject.toml`:
- `flwr[simulation]>=1.26.1`
- `torch==2.8.0`
- `torchvision==0.23.0`

## Structure

- `phase1_flower/server_app.py`: Server orchestration, strategy selection, centralized eval
- `phase1_flower/client_app.py`: Client train/evaluate message handlers
- `phase1_flower/strategy_mavfl.py`: Mobility-aware strategies (MAVFL + CBS/RBS/Random baselines)
- `phase1_flower/mobility.py`: Simplified road-segment mobility/coverage model
- `phase1_flower/models.py`: ResNet-18 and LeNet model builders
- `phase1_flower/data.py`: Dataset partition loading with fake-data fallback
- `phase1_flower/task.py`: Training/evaluation routines
- `configs/*.toml`: run-config variants for baseline and MAVFL scenarios

## Quickstart

From `phase_1_codex`:

```bash
./scripts/smoke_week1.sh
```

MAVFL smoke run:

```bash
./scripts/smoke_week2_mavfl.sh
```

To run with the Flower CLI simulation launcher:

```bash
../venv/bin/flwr run . --run-config "num-server-rounds=2 strategy='mavfl' use-fake-data=false"
```

## Week 3 Scenario Configs

- `configs/cifar10_mavfl.toml`
- `configs/gtsrb_mavfl.toml`

Run with:

```bash
../venv/bin/flwr run . --run-config configs/cifar10_mavfl.toml
../venv/bin/flwr run . --run-config configs/gtsrb_mavfl.toml
```

Baselines (match the paper):

```bash
../venv/bin/flwr run . --run-config "num-server-rounds=5 strategy='cbs' use-fake-data=false"
../venv/bin/flwr run . --run-config "num-server-rounds=5 strategy='rbs' use-fake-data=false"
../venv/bin/flwr run . --run-config "num-server-rounds=5 strategy='random' use-fake-data=false"
```
