#!/usr/bin/env python3
"""Generate concrete parent -> child examples for each semantic mutation helper.
No model or network required. Run from repo root:
  PYTHONPATH=<repo>/slop_src python slop_scripts/audit_semantic_mutations.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "slop_src"))

from slop.prompt_opt.templates import PromptSpec, get_seeds_for_task, prompt_spec_to_dict
from slop.prompt_opt.mutations import (
    mutate_constraints_semantically,
    mutate_anti_slop_semantically,
    mutate_output_format_semantically,
    mutate_reasoning_style_semantically,
)


def main() -> None:
    task = "Explain inflation clearly to a college student."
    seeds = get_seeds_for_task(task)
    parent = seeds[0].copy()

    examples = []

    # 1. Constraints
    child = parent.copy()
    mutate_constraints_semantically(child, __import__("random").Random(42))
    examples.append({
        "helper": "mutate_constraints_semantically",
        "parent_spec": prompt_spec_to_dict(parent),
        "child_spec": prompt_spec_to_dict(child),
        "fields_changed": [k for k in parent.__dataclass_fields__ if getattr(parent, k) != getattr(child, k)],
    })

    # 2. Anti-slop
    parent2 = seeds[0].copy()
    child2 = parent2.copy()
    mutate_anti_slop_semantically(child2, __import__("random").Random(43))
    examples.append({
        "helper": "mutate_anti_slop_semantically",
        "parent_spec": prompt_spec_to_dict(parent2),
        "child_spec": prompt_spec_to_dict(child2),
        "fields_changed": [k for k in parent2.__dataclass_fields__ if getattr(parent2, k) != getattr(child2, k)],
    })

    # 3. Output format (prose_preferred)
    parent3 = seeds[0].copy()
    child3 = parent3.copy()
    mutate_output_format_semantically(child3, __import__("random").Random(44))
    examples.append({
        "helper": "mutate_output_format_semantically",
        "parent_spec": prompt_spec_to_dict(parent3),
        "child_spec": prompt_spec_to_dict(child3),
        "fields_changed": [k for k in parent3.__dataclass_fields__ if getattr(parent3, k) != getattr(child3, k)],
    })

    # 4. Reasoning style
    parent4 = seeds[0].copy()
    child4 = parent4.copy()
    mutate_reasoning_style_semantically(child4, __import__("random").Random(45))
    examples.append({
        "helper": "mutate_reasoning_style_semantically",
        "parent_spec": prompt_spec_to_dict(parent4),
        "child_spec": prompt_spec_to_dict(child4),
        "fields_changed": [k for k in parent4.__dataclass_fields__ if getattr(parent4, k) != getattr(child4, k)],
    })

    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
