# Agent Rules

## Goal

Work on the federated learning thesis project in this repo.
Prioritize correct, runnable code over speculative fidelity.

## Repo Structure

- Main experiment: `experiments/phase1/`
- Docs: `docs/`
- References: `references/`
- Local tool config: `.claude/`

## Working Rules

- Prefer existing repo conventions over inventing new structure.
- Ask questions only if ambiguity changes architecture, experiment behavior, or results.
- For normal code tasks, make the best reasonable choice and proceed.
- Prefer minimal changes over broad rewrites.
- Keep code consistent with the current app-based Flower structure already in the repo.
- Treat algorithm/reference docs as supporting context, not authority over current code.
- Use `uv` for local setup and run commands unless a file clearly requires something else.

## Source Priority

When instructions conflict, follow this order:

1. Current code in `experiments/phase1/`
2. This file
3. `PROJECT_INFO.md`
4. Official framework docs
5. `docs/mavfl-ucb-strategy.md` for paper/algorithm reference

## Flower Conventions

- Use the app-based Flower API already present in this repo.
- Do not introduce legacy Flower client/server patterns unless that file already uses them.
- Keep server/client responsibilities separated.
- Keep experiment changes reproducible through config files or scripts when possible.

## MAVFL Guardrails

- Preserve server-side success-ratio and utility computation for paper-aligned MAVFL changes.
- Preserve equal-weight aggregation across successful uploads unless the experiment explicitly changes that rule.
- Keep UCB accounting per node/arm, not as a single undifferentiated global update.
- Use seeded randomness for strategy sampling when reproducibility matters.
- Call out any mismatch between paper assumptions and the current data partitioning or mobility model.

## Validation

- Run the smallest useful verification available after changes.
- Prefer smoke tests, unit tests, or targeted commands before full simulations.
- If a full simulation cannot be run, say exactly what was not verified and why.

## Communication

- Be direct and concise.
- State assumptions briefly.
- Report changed files, verification, and remaining risks.
