# Flower AI Prime Coder Memory

## Project Structure
- Main implementation: `/Users/niki/Local_Docs/College/flower_test/phase_1_codex/`
- Quickstart reference: `/Users/niki/Local_Docs/College/flower_test/quickstart-pytorch/`
- Skills/agent config: `/Users/niki/Local_Docs/College/flower_test/skills/flower-ai-agent/`
- Project plan: `skills/flower-ai-agent/references/project-plan.md`

## Flower Version & API
- Flower 1.26.1 installed in `venv/`
- Uses NEW app-based API: `ServerApp`, `ClientApp`, `Grid`, `Message`, `RecordDict`
- Imports: `flwr.app`, `flwr.serverapp`, `flwr.clientapp`, `flwr.common`
- Strategy base: `flwr.serverapp.strategy.FedAvg` with `.start()` loop
- Aggregation: `aggregate_arrayrecords(reply_contents, weighted_by_key)`
- Key pattern: override `configure_train` and `aggregate_train` on FedAvg

## Paper: MAVFL (arXiv:2410.10451)
- Mobility-Aware Vehicular FL with MAB-based vehicle selection
- Utility: alpha * p^r - (1-alpha) * normalized_delay
- UCB: mean_reward + sqrt(2 * log(n) / M_k) with discount lambda
- Aggregation: equal-weight among successful vehicles (indicator-based)
- Success ratio: p^r = |successful| / |selected|
- Convergence depends on p^r explicitly
- Sim params: 1000m road, 20 zones, 25m BS, 3MHz BW, 600 IID samples/client

## Known Issues (from initial review 2026-03-07, updated 2026-03-08)
- See `review-issues.md` for detailed list
- FIXED: aggregation equal-weighting (sets weighted_by_key=1.0 for all replies)
- FIXED: GPU — `get_device()` now returns `mps` on Apple M-series; was returning `cpu` (only checked CUDA)
- HIGH: Failed-arm MAB reward — failed nodes get 0 utility, paper assigns full round utility to all selected arms (`strategy_mavfl.py:284`)
- HIGH: UCB `n(r,λ)` wrong — uses sum of arm pull counts across all arms, should be geometric sum `(1-λ^r)/(1-λ)` (`strategy_mavfl.py:268`)
- HIGH: Data partitioning — real CIFAR-10 gives ~5000 samples/client via round-robin, paper uses exactly 600 (`data.py:13-15`)
- Missing: federation config section in pyproject.toml (lines 84-90 commented out, no num-supernodes)

## Pending Work
- Convergence benchmarking plan: see `phase_1_codex/WORKLOG.md` (Session 2026-03-08)
