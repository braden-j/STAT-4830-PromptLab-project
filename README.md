# STAT 4830 PromptLab Project

This repository contains the PromptLab coursework codebase for studying AI slop,
deslopification, prompt optimization, and tournament-style transformer
fine-tuning experiments.

## Tournament Phase 1

The current tournament workflow lives in:

- `tournament/`
- `src/hill_climb/tournament/`

Key entrypoints:

- `tournament/scripts/verify_access.py`
- `tournament/scripts/build_source_pool.py`
- `tournament/scripts/build_p0_pairs.py`
- `tournament/scripts/train_sft.py`
- `tournament/scripts/train_dpo.py`
- `tournament/scripts/eval_checkpoint.py`
- `tournament/scripts/run_phase1_dual_gpu.sh`

Supporting documentation:

- `tournament/IMPLEMENTATION_PLAN.md`
- `tournament/PRIME_RUNBOOK.md`

## Install

For the tournament workflow:

```bash
pip install -e .[tournament]
```

## Notes

- Hugging Face and W&B credentials are expected via environment variables.
- Heavy model artifacts and checkpoints should be stored on remote compute or
  persistent disks rather than committed to GitHub.
