# MAVFL Phase 1 Work Log

## Status: road-loop-m fixed. Fake-data 5-round run clean. Ready for real CIFAR-10 run.

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
