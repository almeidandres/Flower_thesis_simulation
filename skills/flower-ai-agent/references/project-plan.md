# Project Plan Context (Persistent)

Use this as the default project intent for Flower-related work in this repo unless the user overrides it.

## Phase 1

### Week 1
Establish a working Flower + PyTorch baseline and pin a reproducible environment using Flower simulation installation paths and scaffolding patterns from Flower tutorials.
Start by scaffolding the PyTorch quickstart and verify a small number of rounds through `strategy.start(...)` with a basic strategy before adding client-selection logic.

### Week 2
Implement MAVFL as a custom Flower strategy by encoding:
- Select a subset at the beginning of each round.
- Distribute the global model to selected clients in `configure_train`.

Align with Flower message flow (`grid.send_and_receive`) and aggregation (`aggregate_train`) to collect per-round feedback for UCB updates.

### Week 3
Reproduce MAVFL benchmark setup as closely as feasible:
- CIFAR-10 with ResNet-18
- GTSRB with LeNet
- Simplified IDM road-segment setting before SUMO

Use acceptance checks tied to reported delay-to-target-accuracy and convergence-speed behavior.

## Phase 2

### Week 4
Implement a SUMO-based mobility scenario to replace or complement the abstracted road-segment model.
Prefer literature-grounded configurations (e.g. real-map downtown scenario from OpenStreetMap).

### Week 5
Connect mobility scenario to FL availability:
- Determine eligibility from coverage state.
- In `configure_train`, sample only from in-coverage clients.
- Apply MAVFL UCB within the eligible set.

### Week 6
Run side-by-side comparisons:
- Original simplified mobility
- SUMO-driven mobility

Keep learning tasks fixed to isolate the impact of mobility realism on convergence-delay outcomes.

## Phase 3

### Week 7
Add packet-level networking via a literature-backed coupling pattern (e.g., ns-3 + FL simulator, or SUMO + ns-3 co-simulation for V2X learning).
Objective: test whether MAVFL delay-based utility still holds under realistic latency/loss effects.

### Week 8
Run MAVFL with packet-level behavior enabled.
Measure:
- Selected-client upload success/failure rates
- Delay in target accuracy under packet-level networking
- Differences from abstracted network assumptions

### Weeks 9-10
Strengthen validity by cross-checking configuration patterns against prior ns-3 FL evaluation work, and optionally compare coupling assumptions against alternatives (e.g., Veins/SUMO including FLEXE).

## Phase 4

### Week 11
Execute main comparison grid with learning tasks held constant:
- Baseline selection strategy
- MAVFL under abstracted networking
- MAVFL under packet-level networking

Optionally include an additional reproducible vehicle-selection baseline with public implementation artifacts.

### Week 12
Produce core thesis figures:
- Convergence curves
- Time-to-target metrics (as in MAVFL)
- Networking-derived latency/failure statistics explaining deviations

### Weeks 13-14
Package reproducible artifacts:
- Pinned environment
- Scripts
- Experiment configurations

Goal: make simplified vs packet-level result differences reproducible by third parties.

## Paper Anchor

Target algorithm to imitate:
`https://doi.org/10.48550/arXiv.2410.10451`
