# Flower AI Prime Coder Memory

## Project Structure
- Main implementation: `/Users/niki/Local_Docs/College/flower_test/phase_1_codex/`
- Quickstart reference: `/Users/niki/Local_Docs/College/flower_test/quickstart-pytorch/`
- Skills/agent config: `/Users/niki/Local_Docs/College/flower_test/skills/flower-ai-agent/`
- Project plan: `skills/flower-ai-agent/references/project-plan.md`

## Flower Version & API
- Flower 1.26.1 installed in `venv/` (at `/Users/niki/Local_Docs/College/flower_test/venv/`)
- **Activate venv**: `cd /Users/niki/Local_Docs/College/flower_test && source venv/bin/activate`, then `cd phase_1_codex`
- Uses NEW app-based API: `ServerApp`, `ClientApp`, `Grid`, `Message`, `RecordDict`
- Imports: `flwr.app`, `flwr.serverapp`, `flwr.clientapp`, `flwr.common`
- Strategy base: `flwr.serverapp.strategy.FedAvg` with `.start()` loop
- Aggregation: `aggregate_arrayrecords(reply_contents, weighted_by_key)`
- Key pattern: override `configure_train` and `aggregate_train` on FedAvg
- **Simulation nodes are fixed at startup** (`_register_nodes` in vce_api.py). Cannot add/remove mid-run.
- `partition-id` assigned per node in `node_config` automatically by VCE. Each node gets unique partition.
- To simulate dynamic vehicle population: use large fixed pool (e.g., 100 nodes), control visibility via server-side mobility model.

## Paper: MAVFL (arXiv:2410.10451)
- Mobility-Aware Vehicular FL with MAB-based vehicle selection
- Utility: alpha * p^r - (1-alpha) * normalized_delay
- UCB: mean_reward + sqrt(2 * log(n) / M_k) with discount lambda
- Aggregation: equal-weight among successful vehicles (indicator-based)
- Success ratio: p^r = |successful| / |selected|
- Convergence depends on p^r explicitly
- Sim params: 1000m road, 20 zones, 25m BS, 3MHz BW, 600 IID samples/client

## Mobility Model (Open-Road, implemented 2026-03-09)
- **Three-set lifecycle**: _pending (not yet arrived), _active (on road), _exited (permanently gone)
- **No circular wrap-around** -- vehicles exit permanently at road_length_m
- **Poisson arrivals**: `arrival-rate-hz = 0.05` (lowered from 0.167; see below)
- Initial placement: `initial_active` vehicles uniformly on [0, road_length_m)
- New arrivals enter at position 0 with random speed ~[0.7, 1.1] * v_desired
- MAB handles new arrivals correctly: `node_stats.get(node_id)` returns None -> UCB = inf (exploration)
- `eligible_nodes` fallback changed: returns [] (skip round) instead of falling back to all nodes
- `num-clients = 100` in pyproject.toml; run with `num-supernodes=100` via CLI
- `road-loop-m` config key removed; replaced by `arrival-rate-hz` and `initial-active-vehicles`
- Smoke test updated: `tests/smoke_strategy_start.py` uses new constructor (no road_loop_m)

## Known Issues (from initial review 2026-03-07, updated 2026-03-08)
- See `review-issues.md` for detailed list
- FIXED: aggregation equal-weighting (sets weighted_by_key=1.0 for all replies)
- FIXED: GPU -- `get_device()` now returns `mps` on Apple M-series
- FIXED: circular wrap-around replaced with open-road model (2026-03-09)
- FIXED: `arrival-rate-hz` lowered from 0.167 to 0.05 -- old value gave E[arrivals/round]=5.0=K0, all slots taken by inf-UCB newcomers, exploitation never triggered. New: ~1.5/round, ~3-4 slots for finite-UCB exploitation.
- FIXED: Delay normalisation -- `_min_delay`/`_max_delay` dynamic tracking replaced with pre-computed `_tmin`/`_tmax` from channel model bounds (Paper Eq. 6). `_tmin` = delay for center-zone node; `_tmax` = delay for outer-zone node. Fixed in `strategy_mavfl.py:93-116` and `_utility`.
- FIXED: UCB `n(r,lambda)` -- now uses `self._n_r_lambda` geometric accumulator (Eq.9: n(r+1)=n(r)*λ+|S_r|), updated in `_post_round_update`, used in `_select_nodes`
- HIGH: Data partitioning -- real CIFAR-10 gives ~500 samples/client with 100 nodes via round-robin, paper uses exactly 600

## Pending Work
- Convergence benchmarking plan: see `phase_1_codex/WORKLOG.md` (Session 2026-03-08)
- Pool exhaustion: with 100 nodes and Poisson arrivals, pool depletes in ~30-40 rounds. For long runs (>50 rounds), either increase pool size or consider recycling exited nodes with fresh MAB stats.
