#!/usr/bin/env python3
"""Run from repo root to verify hill_climb package is importable. Use:
  PYTHONPATH=<path-to-repo>/src python scripts/hill-climb_scripts/check_slop_import.py
  Or from repo root:  PYTHONPATH=$(pwd)/src python scripts/hill-climb_scripts/check_slop_import.py
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
src_dir = repo_root / "src"
pkg_dir = src_dir / "hill_climb"
print(f"Repo root: {repo_root}")
print(f"src exists: {src_dir.is_dir()}")
print(f"src/hill_climb exists: {pkg_dir.is_dir()}")
print(f"sys.path (first 5): {sys.path[:5]}")

if str(src_dir) not in sys.path:
    print("\nWARNING: src is not on sys.path. Add it with:")
    print(f"  export PYTHONPATH={repo_root}/src   # then run the script")
    print("  or run: PYTHONPATH=<repo>/src python scripts/hill-climb_scripts/check_slop_import.py")
    sys.path.insert(0, str(src_dir))

try:
    import hill_climb
    print("\nimport hill_climb ... OK")
except Exception as e:
    print(f"\n'import hill_climb' failed: {e}")
    print("-> Fix: run from repo root and set PYTHONPATH to <repo>/src")
    sys.exit(1)

try:
    from hill_climb.dataset_io import load_jsonl
    from hill_climb.tokenizer_utils import tokenize_and_align_labels
    print("from hill_climb.dataset_io import load_jsonl ... OK")
    print("from hill_climb.tokenizer_utils import tokenize_and_align_labels ... OK")
    print("\nAll hill_climb imports succeeded.")
except Exception as e:
    print(f"\nhill_climb import failed: {e}")
    if "No module named 'datasets'" in str(e) or "No module named 'torch'" in str(e):
        print("-> Fix: install deps, e.g. pip install torch transformers datasets")
    sys.exit(1)
