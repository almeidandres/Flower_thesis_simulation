# Flower Thesis Simulation

Research repository for a thesis project on mobility-aware federated learning with Flower and PyTorch.

## Main project

- `experiments/phase1/`: primary implementation of the Phase 1 experiments
- `experiments/phase1/phase1_flower/`: Flower server, client, strategy, mobility, data, and training code
- `experiments/phase1/configs/`: runnable experiment configs for CIFAR-10 and GTSRB

## Main experiment

The main experiment runs the MAVFL strategy from `experiments/phase1/` and compares it against the CBS, RBS, and Random baselines.

Using `uv`:

```bash
cd experiments/phase1
uv venv
uv pip install -e .
uv run flwr run . --run-config configs/cifar10_mavfl.toml
```

## Project layout

- `docs/`: algorithm and paper reference notes
- `experiments/`: runnable thesis experiments
- `references/`: reference material, papers, and example apps
- `.claude/`: optional local Claude configuration

## Repo guidance

- Canonical repo rules: `AGENTS.md`
- Stable project summary: `PROJECT_INFO.md`
- Algorithm reference: `docs/mavfl-ucb-strategy.md`

## Where to start

- Run guide: `experiments/phase1/README.md`
- Algorithm notes: `docs/mavfl-ucb-strategy.md`
- Project status: `PROJECT_INFO.md`
