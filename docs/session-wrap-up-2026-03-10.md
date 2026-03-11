# Session Wrap-Up (MAVFL Fidelity Pass)

## Objective

Make the Phase 1 implementation align with the MAVFL paper (arXiv:2410.10451), use Flower end-to-end simulation, and add baselines (CBS/RBS/Random).

## Key decisions

- Moved reward/utility computation to the **server** to match the paper’s formulation.
- Replaced synthetic delay + sinusoidal coverage with **IDM-style mobility** and **OFDMA delay model**.
- Implemented baselines **CBS / RBS / Random** as in the paper.
- Exposed paper-relevant parameters in configs (selection size K0, alpha, delay, IDM, channel).

## What changed (files)

- `experiments/phase1/phase1_flower/mobility.py`
  - New `IDMRoadMobility` with road segment length 1000m and loop length 2000m.
  - Zone-based distance to BS (20 zones, BS height 25m).
  - Time-to-exit and mobility advance per round.
- `experiments/phase1/phase1_flower/strategy_mavfl.py`
  - New mobility-aware strategy base.
  - MAVFL UCB selection with discount + utility (`alpha * p^r − (1−alpha) * normalized_delay`).
  - Baseline strategies: `CBSStrategy`, `RBSStrategy`, `RandomStrategy`.
  - Success ratio computed server-side; updates only from “successful” vehicles.
- `experiments/phase1/phase1_flower/client_app.py`
  - Removed synthetic delay/reward from clients.
  - Clients now return only train/eval metrics.
- `experiments/phase1/phase1_flower/server_app.py`
  - Strategy factory supports `mavfl`, `cbs`, `rbs`, `random`.
  - Computes model size bits and per-client data bits.
  - Injects IDM + channel parameters via run config.
- `experiments/phase1/pyproject.toml`
  - Added MAVFL parameters and mobility/channel defaults:
    - `num-selected`, `mavfl-alpha`, `ucb-discount`, IDM params, road lengths, OFDMA params, compute params.
- `experiments/phase1/configs/cifar10_mavfl.toml`
  - `use-fake-data = false`, added `ucb-discount`, `mavfl-alpha`, `num-selected`, `vehicle-speed-kmh`.
- `experiments/phase1/configs/gtsrb_mavfl.toml`
  - Same as CIFAR config.
- `experiments/phase1/README.md`
  - Updated to include baselines and real-data runs.
- `experiments/phase1/tests/smoke_strategy_start.py`
  - Updated to new mobility + delay params (still in-process smoke).

## Why `flwr run .` failed in the sandbox

- In this environment, Ray fails due to OS-level process/port permissions (psutil/sysctl/port bind).
- End-to-end simulation must be run in a normal terminal with OS permissions.

## How to run proper end-to-end sims

From `experiments/phase1`:

```bash
../venv/bin/pip install -e .
../venv/bin/flwr run . --run-config configs/cifar10_mavfl.toml
../venv/bin/flwr run . --run-config configs/gtsrb_mavfl.toml
```

Baselines:

```bash
../venv/bin/flwr run . --run-config "num-server-rounds=5 strategy='cbs' use-fake-data=false"
../venv/bin/flwr run . --run-config "num-server-rounds=5 strategy='rbs' use-fake-data=false"
../venv/bin/flwr run . --run-config "num-server-rounds=5 strategy='random' use-fake-data=false"
```

## What still needs confirmation

- Exact values for transmit power, compute cycles, and IDM parameters (paper formulas exist, but numeric defaults may be in Table I).
- Exact K0 (num-selected) and alpha used in the paper for the reported gains.
- Alignment of dataset sizes (paper says 600 IID samples/client).

## Current status

- Code now matches MAVFL structure and evaluation design in the paper.
- End-to-end validation should be done outside the sandbox and logs checked for:
  - `mavfl-success-ratio`
  - `mavfl-round-delay-s`
  - `mavfl-utility`
  - Accuracy vs. baselines at 60/80 km/h.
