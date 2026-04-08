#!/usr/bin/env python3
"""Run from repo root:  PYTHONPATH=$(pwd)/slop_src python slop_scripts/debug_slop_import.py"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
slop_src = repo_root / "slop_src"
slop_dir = slop_src / "slop"

print("1. Repo root:", repo_root)
print("2. slop_src exists:", slop_src.is_dir())
print("3. slop_src/slop exists:", slop_dir.is_dir())
print("4. sys.path (first 3):", sys.path[:3])
print("5. Is slop_src on sys.path?", str(slop_src) in sys.path)

# Add slop_src if not present (so we can at least test)
if str(slop_src) not in sys.path:
    sys.path.insert(0, str(slop_src))
    print("   -> Added slop_src to sys.path")

print("\n6. Importing slop...")
try:
    import slop
    print("   slop.__file__ =", getattr(slop, "__file__", "N/A"))
    print("   slop package location:", Path(slop.__file__).parent if hasattr(slop, "__file__") else "N/A")
except Exception as e:
    print("   FAILED:", e)
    sys.exit(1)

print("\n7. from slop.dataset_io import load_jsonl...")
try:
    from slop.dataset_io import load_jsonl
    print("   OK")
except Exception as e:
    print("   FAILED:", e)
    sys.exit(1)

print("\n8. from slop.tokenizer_utils import tokenize_and_align_labels...")
try:
    from slop.tokenizer_utils import tokenize_and_align_labels
    print("   OK")
except Exception as e:
    print("   FAILED:", e)
    sys.exit(1)

print("\nAll imports OK. Slop package is working.")
