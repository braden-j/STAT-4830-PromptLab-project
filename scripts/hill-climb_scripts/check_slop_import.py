#!/usr/bin/env python3
"""Run from repo root to verify slop package is importable. Use:
  PYTHONPATH=<path-to-repo>/slop_src python slop_scripts/check_slop_import.py
  Or from repo root:  PYTHONPATH=$(pwd)/slop_src python slop_scripts/check_slop_import.py
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
slop_src = repo_root / "slop_src"
print(f"Repo root: {repo_root}")
print(f"slop_src exists: {slop_src.is_dir()}")
print(f"slop_src/slop exists: {(slop_src / 'slop').is_dir()}")
print(f"slop_src/slop exists: {(slop_src / 'slop').is_dir()}")
print(f"sys.path (first 5): {sys.path[:5]}")

if str(slop_src) not in sys.path:
    print("\nWARNING: slop_src is not on sys.path. Add it with:")
    print(f"  export PYTHONPATH={repo_root}/slop_src   # then run the script")
    print("  or run: PYTHONPATH=<repo>/slop_src python slop_scripts/check_slop_import.py")
    sys.path.insert(0, str(slop_src))

try:
    import slop
    print("\nimport slop ... OK")
except Exception as e:
    print(f"\n'import slop' failed: {e}")
    print("-> Fix: run from repo root and set PYTHONPATH to <repo>/slop_src")
    sys.exit(1)

try:
    from slop.dataset_io import load_jsonl
    from slop.tokenizer_utils import tokenize_and_align_labels
    print("from slop.dataset_io import load_jsonl ... OK")
    print("from slop.tokenizer_utils import tokenize_and_align_labels ... OK")
    print("\nAll slop imports succeeded.")
except Exception as e:
    print(f"\nslop import failed: {e}")
    if "No module named 'datasets'" in str(e) or "No module named 'torch'" in str(e):
        print("-> Fix: install deps, e.g. pip install torch transformers datasets")
    sys.exit(1)
