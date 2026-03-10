# MAVFL Phase 1 Work Log

## Status: Open-road continuous arrival model implemented. 100-vehicle pool. Circular wrap-around removed. Ready for real CIFAR-10 run.

---

## What was done

Implemented all bug fixes from the `flowerai-prime-coder` review of the MAVFL
federated learning reconstruction (paper: https://doi.org/10.48550/arXiv.2410.10451).

### Fixes applied

| # | File | Description |
|---|------|-------------|
| 1 | `server_app.py` | `_model_size_bits` return — **already correct**, no change needed |
| 2 | `server_app.py` | Duplicate `"mobility"` key — **already correct**, no change needed |
| 3 | `strategy_mavfl.py` | Equal-weight aggregation (Paper Eq. 3): set `num-examples=1.0` on each reply before calling `aggregate_arrayrecords`/`aggregate_metricrecords` |
| 4 | `strategy_mavfl.py` | Per-arm MAB reward: failed-but-selected nodes get `utility=0.0`, successful nodes get the actual round utility |
| 5 | `pyproject.toml`, both `configs/*.toml` | `train-samples-per-client` 256 → 600 (paper spec) |
| 6 | `client_app.py`, `server_app.py` | Safe `use-fake-data` bool parsing: `val if isinstance(val, bool) else str(val).lower() == "true"` |
| 7 | `strategy_mavfl.py`, `server_app.py` | Seeded RNG: `self._rng = random.Random(seed)` in `MobilitySelectionStrategy.__init__`; all `random.sample()` replaced with `self._rng.sample()`; `seed` passed from `context.run_config` |
| 8 | `server_app.py` | Per-round metric persistence: writes `artifacts/metrics.csv` after each run with columns `round, success-ratio, round-delay-s, utility, server-loss, server-accuracy` |

### Key design notes

- `aggregate_arrayrecords` / `aggregate_metricrecords` do **not** accept `None` for the
  weight key — equal weighting is achieved by normalizing `num-examples` to `1.0` in
  place on each `RecordDict` before aggregating.
- `result.train_metrics_clientapp` is a `dict[int, MetricRecord]` keyed by round number.
  MAVFL metrics (`mavfl-*`) are injected into this by `aggregate_train`.
- `result.evaluate_metrics_serverapp` holds the server-side loss/accuracy keyed by round.
- The smoke test (`tests/smoke_strategy_start.py`) uses a `MockGrid` and completes in
  ~0.02s. It does **not** exercise `ClientApp`/`ServerApp` wiring.

### Verification

Smoke test passes:
```
python tests/smoke_strategy_start.py
# → smoke_strategy_start: PASS
```

MAB utility values change across rounds (0.6 → 0.6 → 0.2), confirming per-arm updates
work correctly.

---

## Session 2026-03-08 — Fix road-loop-m dead zone

### Root cause identified

`road-loop-m = 2000` created a 2000 m circular loop with only the first 1000 m as the
coverage zone. After each ~66 s round, vehicles advanced ~1100 m and simultaneously
flushed into the dead zone. With 10 vehicles, this caused round 2 to have 0 eligible
vehicles (no model update) and round 4 to have only 1.

The paper describes a **straight 1000 m road segment** — no loop, no dead zone.

### Fix applied

| File | Change |
|------|--------|
| `pyproject.toml` line 48 | `road-loop-m = 2000.0` → `road-loop-m = 1000.0` |

With `road-loop-m = 1000`, vehicles wrap at 1000 m back to position 0, so all 10 nodes
remain eligible every round. Natural dropout still occurs via channel quality and
`time_to_exit`.

### Verification

- Smoke test: PASS (unchanged)
- Fake-data 5-round MAVFL run: every round `selected 5/10 eligible nodes (10 total)` —
  no dropout rounds. `success-ratio` varied 0.2–0.6 per round (natural, not total
  failure). Utility changed round-to-round confirming MAB updates. `artifacts/metrics.csv`
  written with 5 rows.

---

---

## Session 2026-03-09 — Open-road continuous arrival model

### Problem identified: circular wrap-around is unfaithful to the paper

Through analysis of the paper (arXiv:2410.10451) and the codebase, three issues were found with the previous `road-loop-m = 1000` circular model:

1. **MAB exploration permanently suppressed.** After round 1, all 10 node IDs had been seen, so `UCB = inf` (the exploration bonus for new arrivals) was never triggered again. New arrivals getting the exploration bonus is a core mechanism of MAVFL.
2. **Data reuse.** The same 10 fixed data partitions were trained on every single round — not how real vehicular FL works, and not what the paper models.
3. **The paper describes a one-way road.** Vehicles that exit coverage are gone. The wrap-around was a pragmatic hack, not a paper-faithful design.

**Root cause of the original hack:** The paper is underspecified. It never describes a vehicle arrival process, yet its results (3000s of CIFAR-10 training) are physically impossible without one — all vehicles at 60 km/h traverse 1000m in ~60s, depleting the entire pool in 1–2 rounds. The paper likely simulated a steady-state traffic stream (open road) but omitted the description.

### Solution: open-road model with continuous Poisson arrivals

**Arrival rate derived from Little's Law:**
```
Transit time  T  = L / v = 1000 m / 16.67 m/s ≈ 60 s
Arrival rate  λ  = N_target / T = 10 / 60 ≈ 0.167 vehicles/second
```
At steady state, ~10 vehicles are on the road at any time. Each `advance()` call draws from `Poisson(λ × duration_s)` to activate new vehicles.

### Changes applied

| File | Change |
|------|--------|
| `mobility.py` | Full rewrite. Replaced circular wrap with three-set lifecycle: `_pending` (not yet on road), `_active` (traversing), `_exited` (permanently departed). Poisson arrivals at rate 0.167/s. No `road_loop_m`. |
| `strategy_mavfl.py` | Fixed `eligible_nodes` fallback — no longer falls back to all 100 nodes when pool is empty; skips round instead. Added `pool_summary()` to per-round logs. |
| `server_app.py` | Replaced `road_loop_m=` with `arrival_rate_hz=` and `initial_active=` in `IDMRoadMobility` constructor. |
| `pyproject.toml` | `num-clients = 100`, removed `road-loop-m`, added `arrival-rate-hz = 0.167` and `initial-active-vehicles = 10`. |
| `tests/smoke_strategy_start.py` | Updated constructor call to match new parameters. |

### MAB behaviour with new arrivals

No MAB changes needed. When a pending node activates for the first time, `self.node_stats.get(node_id)` returns `None`, causing `_ucb_score` to return `float("inf")` — the exploration bonus is automatically applied to all new arrivals.

### Pool exhaustion warning

With 100 nodes and ~10 active at steady state, the pending pool supports roughly **~18 rounds** at 30s round delays before exhaustion. For longer runs (50+ rounds), increase `num-clients` and `num-supernodes` proportionally (e.g., 500 for long CIFAR-10 experiments). Do **not** recycle exited nodes — that reintroduces the data-reuse problem.

### Next steps updated

- `num-supernodes = 100` must be passed at CLI: `flwr run . --run-config "num-clients=100"` with appropriate federation config
- Smoke test constructor alignment done; full fake-data run should be verified next
- Benchmarking plan (Session 2026-03-08) still applies — strategies and metrics infrastructure unchanged

---

## Session 2026-03-10 — UCB MAB fix: update on zero-success rounds

### Problem identified

`aggregate_train` gated both the utility computation and `_post_round_update` inside
`if filtered_replies:`. When all selected nodes dropped out of coverage (zero successful
replies), the MAB update was silently skipped:

- Arms that caused the failure received no negative reward — their `mean_reward` stayed
  artificially high, so UCB kept re-selecting them.
- `_n_r_lambda` (the discounted total-selections counter, Paper Eq. 9) did not advance,
  distorting the exploration bonus for subsequent rounds.

### Fix applied

| File | Change |
|------|--------|
| `strategy_mavfl.py` | Moved `utility = self._utility(...)` before the `if filtered_replies:` block; moved `self._post_round_update(server_round, utility)` after it (unconditional). Metric dict population (`mavfl-*` fields) stays inside the block since metrics are only valid when results exist. |

### Verification

7-round CIFAR-10 real-data run (100 nodes, `initial-active=10`):

| Round | Old accuracy | New accuracy |
|-------|-------------|--------------|
| 1 | 0.1294 | 0.1024 |
| 2 | 0.2018 | 0.2441 |
| 3 | 0.2658 | 0.2824 |
| 4 | 0.2693 | 0.2743 |
| 5 | 0.2138 | 0.2524 |
| 6 | 0.2655 | 0.3399 |
| 7 | 0.3140 | **0.3471** |

Final accuracy improved from 31.4% → 34.7% (+3.3pp). Rounds 2–7 are strictly better or
equal. Round 1 had a loss spike (17.0) due to a bad initial round with `success-ratio=0.2`
immediately followed by evaluation — unrelated to the UCB fix, present in prior runs too.

---

## Next steps

### 1. Real CIFAR-10 run
```bash
flwr run . --config-overrides configs/cifar10_mavfl.toml
```
Check: `num-examples` per client is 600, `success-ratio` logged each round, utility
values change across rounds.

### 3. Real GTSRB run
```bash
flwr run . --config-overrides configs/gtsrb_mavfl.toml
```

### 4–5. Baseline comparison + convergence analysis
See Session 2026-03-08 (Convergence Benchmarking Plan) below.

---

---

## Session 2026-03-08 — GPU fix + Convergence Benchmarking Plan

### GPU fix applied

`task.py:get_device()` was returning `cpu` on Apple Silicon because it only checked
`torch.cuda.is_available()`. Fixed to check MPS first:

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Verified: `python -c "from phase1_flower.task import get_device; print(get_device())"` → `mps`

---

### Convergence Benchmarking Plan (not yet implemented)

**What the paper's "28% improvement" means:**

The paper measures **simulated wall-clock time to reach a target test accuracy**, not
number of rounds. Specifically: sum `round-delay-s` across rounds until
`server-accuracy ≥ 0.75` (CIFAR-10) or `≥ 0.90` (GTSRB). The 28% is MAVFL vs.
**Random** baseline. All 4 baselines already exist in `strategy_mavfl.py`.

| Strategy | Paper result vs MAVFL |
|---|---|
| Random | ~28% slower (primary baseline) |
| CBS | ~21% slower |
| RBS | ~12% slower |
| MAVFL | fastest (proposed method) |

**Three changes needed:**

**1. `server_app.py:161-189`** — Add `cumulative-time-s` and `strategy` columns to metrics.csv

- `cumulative-time-s`: running sum of `round-delay-s` — this is the paper's x-axis
- `strategy`: from `context.run_config["strategy"]` — enables concatenating CSVs from all runs

**2. `pyproject.toml:84-90`** — Uncomment federation config block, add `options.num-supernodes = 10`

**3. New `tests/benchmark_convergence.py`** — Reads multiple strategy CSVs, prints comparison table

```
strategy   rounds_to_target   sim_seconds_to_target   speedup_vs_random
random     42                 2940.1                  1.00x
cbs        35                 2320.5                  1.27x
rbs        31                 2100.2                  1.40x
mavfl      28                 1884.3                  1.56x
```

**Run workflow (once implemented):**
```bash
flwr run . --run-config "strategy='random'"  && cp artifacts/metrics.csv artifacts/metrics_random.csv
flwr run . --run-config "strategy='cbs'"     && cp artifacts/metrics.csv artifacts/metrics_cbs.csv
flwr run . --run-config "strategy='rbs'"     && cp artifacts/metrics.csv artifacts/metrics_rbs.csv
flwr run . --run-config "strategy='mavfl'"   && cp artifacts/metrics.csv artifacts/metrics_mavfl.csv
python tests/benchmark_convergence.py artifacts/metrics_*.csv --target-accuracy 0.75
```

**Pass/fail criteria:**
- PASS: MAVFL reaches 75% in fewer simulated seconds than Random, and ordering is MAVFL < RBS < CBS < Random
- FAIL: ordering wrong → strip IDM to constant speed and retest to isolate mobility model as confound

---

## Flower API notes (version 1.26.x)

- `strategy.start()` returns a `Result` with:
  - `result.arrays` — final model
  - `result.train_metrics_clientapp` — `dict[int, MetricRecord]` (round → metrics)
  - `result.evaluate_metrics_clientapp` — `dict[int, MetricRecord]`
  - `result.evaluate_metrics_serverapp` — `dict[int, MetricRecord]`
- `FedAvg.__init__` keyword args: `weighted_by_key="num-examples"`, `arrayrecord_key="arrays"`, `configrecord_key="config"`
- No `[tool.flwr.federations]` section in `pyproject.toml` is correct for 1.26.x (migration notice in file explains this)
