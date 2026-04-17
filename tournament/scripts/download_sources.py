#!/usr/bin/env python3
"""Fetch or validate the external data sources used by the tournament."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hill_climb.tournament.config import SourcePoolConfig, load_yaml_config
from hill_climb.tournament.data import download_defan
from hill_climb.tournament.io import ensure_parent, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download or validate tournament data sources")
    p.add_argument("--config", default=None, help="Optional source-pool YAML config")
    p.add_argument("--manifest-out", default="tournament/data/raw/source_manifest.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config, "source") if args.config else SourcePoolConfig()
    manifest: dict[str, str | bool] = {}

    kaggle_path = Path(cfg.kaggle_path)
    manifest["kaggle_path"] = str(kaggle_path)
    manifest["kaggle_exists"] = kaggle_path.exists()

    defan_path = download_defan(cfg.defan_path)
    manifest["defan_path"] = str(defan_path)
    manifest["defan_exists"] = defan_path.exists()

    out_path = ensure_parent(args.manifest_out)
    write_json(out_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
