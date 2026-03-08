# Detailed Review Issues (2026-03-07)

## Critical (mathematical fidelity / correctness)
1. Aggregation weighting: uses FedAvg num-examples weighting, paper uses equal 1/|N^r|
2. UCB discount applied to ALL nodes globally, paper applies per-arm discount
3. UCB utility assigned identically to all selected nodes, not per-node
4. Missing 600 IID samples/client (paper spec), defaults to 256 train
5. Convergence bound depends on p^r but aggregation doesn't reflect this

## Moderate (architecture / Flower conventions)
6. pyproject.toml missing [tool.flwr.federations] for simulation num-supernodes
7. aggregate_train calls list(replies) consuming the iterable, then _check_and_log_replies also iterates
8. ConfigRecord mutated in configure_train (shared reference risk)
9. predict_position ignores IDM interactions (linear extrapolation only)
10. No seeding of random.sample in MAVFLStrategy._select_nodes

## Minor
11. data_bits_per_client uses raw pixel bits, paper may mean training data volume
12. path_loss formula sign: 128.1 + 37.6*log10(d_km) can go negative for d<1km
13. No logging of which nodes were filtered as unsuccessful
14. Smoke test MockGrid doesn't set src_node_id properly for success filtering
