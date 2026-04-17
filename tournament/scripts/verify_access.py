#!/usr/bin/env python3
"""Verify W&B, HF, Pangram, and Meta Llama access before tournament runs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import wandb
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}. Install with `pip install .[tournament]`.")
        return 1

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")
    if not token:
        print("[error] Missing HF_TOKEN or HUGGINGFACE_HUB_TOKEN.")
        return 1
    if not wandb_key:
        print("[error] Missing WANDB_API_KEY.")
        return 1

    results: dict[str, dict[str, str]] = {"hf": {}, "wandb": {}}
    api = HfApi(token=token)
    try:
        who = api.whoami(token=token)
        results["hf"]["whoami"] = who.get("name", "ok")
    except Exception as exc:
        print(json.dumps({"hf": {"whoami": f"error:{type(exc).__name__}:{exc}"}}))
        return 1

    checks = [
        ("model", "meta-llama/Llama-3.2-3B", "config.json"),
        ("model", "meta-llama/Llama-3.2-3B", "generation_config.json"),
        ("model", "pangram/editlens_roberta-large", "config.json"),
        ("model", "pangram/editlens_Llama-3.2-3B", "adapter_config.json"),
        ("dataset", "pangram/editlens_iclr", "README.md"),
        ("dataset", "pangram/editlens_iclr_grammarly", "README.md"),
    ]
    ok = True
    for repo_type, repo_id, filename in checks:
        key = f"{repo_type}:{repo_id}:{filename}"
        try:
            if repo_type == "model":
                api.model_info(repo_id, token=token)
                hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            else:
                api.dataset_info(repo_id, token=token)
                hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
            results["hf"][key] = "ok"
        except Exception as exc:
            ok = False
            results["hf"][key] = f"error:{type(exc).__name__}:{str(exc)[:180]}"

    try:
        wandb.login(key=wandb_key, relogin=True, verify=True)
        wb_api = wandb.Api(api_key=wandb_key)
        viewer = wb_api.viewer
        results["wandb"]["viewer"] = getattr(viewer, "username", "ok")
        project = wb_api.project("jgold23-university-of-pennsylvania-model-united-nations-/PromptLab_STAT4830")
        results["wandb"]["project"] = getattr(project, "name", "ok")
    except Exception as exc:
        ok = False
        results["wandb"]["login"] = f"error:{type(exc).__name__}:{str(exc)[:180]}"

    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
