# Tournament Workspace

This folder is the phase-1 workspace for the transformer deslopifier tournament.

Auth preflight has now passed for the required lightweight checks:

- `W&B`: verified working
- `Hugging Face`: verified for Pangram datasets, Pangram adapters, and Meta Llama config access

Use [IMPLEMENTATION_PLAN.md](/Users/jgold/STAT-4830-PromptLab-project/tournament/IMPLEMENTATION_PLAN.md) as the source of truth for:

- experiment coverage
- dataset schemas
- training and eval entrypoints
- run promotion logic
- tracking expectations

The next concrete step is to keep building out the scripts under `tournament/scripts/` and the shared package under `src/hill_climb/tournament/`.
