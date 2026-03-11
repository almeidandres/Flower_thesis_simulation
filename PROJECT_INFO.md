# Project Info

This repo studies mobility-aware federated learning based on MAVFL using Flower and PyTorch.

## Main experiment

- Primary code lives in `experiments/phase1/`
- Main strategies: `mavfl`, `cbs`, `rbs`, `random`
- Main tasks: CIFAR-10 with ResNet-18 and GTSRB with LeNet
- Preferred workflow uses `uv`

## Current implementation status

- The codebase uses the Flower app-based API with `ServerApp` and `ClientApp`
- The main Phase 1 implementation targets paper-aligned MAVFL behavior plus CBS, RBS, and Random baselines
- Utility and success-ratio computation are handled on the server side
- Aggregation is intended to be equal-weight across successful uploads for MAVFL-style rounds
- The mobility model is an open one-way road segment with Poisson arrivals, not a circular loop
- Experiment outputs include `artifacts/final_model.pt` and `artifacts/metrics.csv`

## Key current design decisions

- `experiments/phase1/phase1_flower/mobility.py` uses an open-road `IDMRoadMobility` model with pending, active, and exited vehicles
- `experiments/phase1/phase1_flower/strategy_mavfl.py` contains MAVFL plus the CBS, RBS, and Random baselines
- `experiments/phase1/phase1_flower/server_app.py` constructs mobility and delay models from run config and writes round metrics to CSV
- `experiments/phase1/phase1_flower/client_app.py` returns training and evaluation metrics only; synthetic reward logic was removed from clients
- `experiments/phase1/pyproject.toml` holds the default run config, including `num-clients = 100`, `num-selected = 5`, `mavfl-alpha = 0.6`, `ucb-discount = 0.9`, `arrival-rate-hz = 0.05`, and `train-samples-per-client = 600`

## Guardrails from prior issues

These are the main failure modes surfaced by the earlier review and worklog. New changes should avoid regressing on them.

- Preserve equal-weight aggregation across successful uploads for paper-aligned MAVFL behavior
- Keep success ratio and utility computation on the server side
- Keep discounted UCB accounting per node/arm instead of as one global reward bucket
- Keep reward updates tied to actual selected nodes and actual round outcomes
- Use seeded randomness for selection and sampling when reproducibility matters
- Avoid shared-mutable-config bugs and iterator-consumption bugs in strategy code
- Call out clearly when real-data partitioning differs from the paper assumptions

## Current mobility and data caveats

- The current baseline is the simplified open-road mobility model in `experiments/phase1/`
- Do not reintroduce circular wrap-around or loop-based assumptions unless intentionally changing the experiment design
- The code config requests 600 train samples per client, but real-data partitioning is deterministic round-robin over the dataset size, so exact paper alignment still needs to be checked per dataset and client count
- With the current open-road arrival model, long runs may require a larger client pool if the pending pool exhausts

## Latest recorded verification state

- The smoke strategy test exists at `experiments/phase1/tests/smoke_strategy_start.py`
- Previous project verification notes recorded a passing smoke test for the strategy-start path
- Previous project verification notes also recorded a later 7-round CIFAR-10 run used to verify the zero-success-round UCB update fix
- Full end-to-end Flower simulation still needs to be run in a normal terminal environment when sandbox or Ray permissions block it

## How to run

From `experiments/phase1/`:

```bash
uv venv
uv pip install -e .
uv run flwr run . --run-config configs/cifar10_mavfl.toml
```

Other main runs:

```bash
uv run flwr run . --run-config configs/gtsrb_mavfl.toml
uv run flwr run . --run-config "num-server-rounds=5 strategy='cbs' use-fake-data=false"
uv run flwr run . --run-config "num-server-rounds=5 strategy='rbs' use-fake-data=false"
uv run flwr run . --run-config "num-server-rounds=5 strategy='random' use-fake-data=false"
```

## What to inspect in results

- `mavfl-success-ratio`
- `mavfl-round-delay-s`
- `mavfl-utility`
- Server accuracy across rounds
- `experiments/phase1/artifacts/metrics.csv`

## Open questions

- Exact paper values for transmit power, compute parameters, and IDM defaults may still need confirmation against the paper tables
- Exact paper-aligned values for selection size `K0` and `alpha` used in reported gains should be confirmed
- A dedicated convergence-benchmark script and cumulative-time comparison workflow were planned previously but are not yet part of the active project summary

## Source-of-truth policy

- Use current code in `experiments/phase1/` as the primary truth
- Use `AGENTS.md` for repo-wide working rules
- Use this file for current project status and implementation context
- Treat `docs/mavfl-ucb-strategy.md` as algorithm reference, not implementation status
