#!/usr/bin/env python3
"""Run from repo root:  PYTHONPATH=$(pwd)/src python scripts/hill-climb_scripts/debug_slop_import.py"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
src_dir = repo_root / "src"
pkg_dir = src_dir / "hill_climb"

print("1. Repo root:", repo_root)
print("2. src exists:", src_dir.is_dir())
print("3. src/hill_climb exists:", pkg_dir.is_dir())
print("4. sys.path (first 3):", sys.path[:3])
print("5. Is src on sys.path?", str(src_dir) in sys.path)

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    print("   -> Added src to sys.path")

print("\n6. Importing hill_climb...")
try:
    import hill_climb
    print("   hill_climb.__file__ =", getattr(hill_climb, "__file__", "N/A"))
    print("   hill_climb package location:", Path(hill_climb.__file__).parent if hasattr(hill_climb, "__file__") else "N/A")
except Exception as e:
    print("   FAILED:", e)
    sys.exit(1)

print("\n7. from hill_climb.dataset_io import load_jsonl...")
try:
    from hill_climb.dataset_io import load_jsonl
    print("   OK")
except Exception as e:
    print("   FAILED:", e)
    sys.exit(1)

print("\n8. from hill_climb.tokenizer_utils import tokenize_and_align_labels...")
try:
    from hill_climb.tokenizer_utils import tokenize_and_align_labels
    print("   OK")
except Exception as e:
    print("   FAILED:", e)
    sys.exit(1)

print("\nAll imports OK. hill_climb package is working.")
