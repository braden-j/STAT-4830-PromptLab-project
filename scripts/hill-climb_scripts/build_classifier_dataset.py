#!/usr/bin/env python3
"""Build a large weakly supervised token-level slop dataset for the classifier.

Generates hundreds to thousands of examples: clean text (all 0 labels) and
rule-based slop (word-aligned 0/1 labels). Outputs train/val/test JSONL with
class-balanced splits. Labels align to whitespace-tokenized words.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from hill_climb.dataset_io import load_jsonl
from hill_climb.slop_gen import RuleSloppifier


# Built-in corpus of clean sentences to scale without external data
CLEAN_CORPUS = [
    "The report was clear and well structured.",
    "We need to improve efficiency in the process.",
    "The team agreed on the next steps.",
    "She presented the results at the meeting.",
    "The company announced its new policy.",
    "The study found a significant effect.",
    "The data shows a clear trend.",
    "The project will start next month.",
    "The solution addresses the main problem.",
    "The approach has several advantages.",
    "The method is widely used in practice.",
    "The results support our hypothesis.",
    "The analysis reveals important patterns.",
    "The model performs well on the task.",
    "The system handles most cases correctly.",
    "The framework provides a useful structure.",
    "The policy applies to all employees.",
    "The change will take effect in January.",
    "The update includes several improvements.",
    "The report summarizes the key findings.",
    "The meeting will cover three main topics.",
    "The plan requires approval from the board.",
    "The budget was approved last week.",
    "The deadline is the end of the quarter.",
    "The customer requested a full refund.",
    "The doctor recommended rest and fluids.",
    "The teacher explained the concept clearly.",
    "The student submitted the assignment on time.",
    "The computer crashed during the update.",
    "The phone rang several times.",
    "The car needs regular maintenance.",
    "The hospital has a new wing.",
    "The restaurant is closed on Mondays.",
    "The house was built in the nineties.",
    "The city has a population of one million.",
    "The country exports mainly machinery.",
    "The book is available in the library.",
    "The movie received positive reviews.",
    "The idea came up during the discussion.",
    "The result exceeded our expectations.",
    "The solution was implemented successfully.",
    "The problem was identified early.",
    "The employee joined the company last year.",
    "The customer service team responded quickly.",
    "The manager approved the request.",
    "The committee will meet next Tuesday.",
    "The department is responsible for quality.",
    "The office will reopen next week.",
    "The conference takes place in March.",
    "The workshop will run for two days.",
    "The training is mandatory for new staff.",
    "The survey was sent to all participants.",
    "The feedback was mostly positive.",
    "The response rate was high.",
    "The sample size was sufficient.",
    "The criteria are clearly defined.",
    "The requirements have been updated.",
    "The specification is now complete.",
    "The design was reviewed by the team.",
    "The implementation follows best practices.",
    "The test coverage has improved.",
    "The documentation is up to date.",
    "The version will be released soon.",
    "The fix was deployed yesterday.",
    "The issue was resolved quickly.",
    "The error message was helpful.",
    "The performance has increased.",
    "The cost was lower than expected.",
    "The time required was minimal.",
    "The effort was worthwhile.",
    "The outcome was satisfactory.",
    "The impact was significant.",
    "The benefit is obvious.",
    "The advantage is clear.",
    "The risk is low.",
    "The chance of success is high.",
    "The probability is estimated at fifty percent.",
    "The estimate seems reasonable.",
    "The assumption was correct.",
    "The conclusion is supported by the data.",
    "The evidence is compelling.",
    "The argument is persuasive.",
    "The reasoning is sound.",
    "The logic is straightforward.",
    "The approach is pragmatic.",
    "The strategy is effective.",
    "The tactic worked as planned.",
    "The technique is commonly used.",
    "The tool is easy to use.",
    "The resource is freely available.",
    "The material is suitable for the course.",
    "The content is relevant to the topic.",
    "The information was accurate.",
    "The details are in the appendix.",
    "The summary is on the first page.",
    "The introduction provides context.",
    "The conclusion restates the main points.",
    "The recommendation is to proceed.",
    "The suggestion was well received.",
    "The proposal will be reviewed.",
    "The request has been forwarded.",
    "The inquiry was answered promptly.",
    "The question was addressed in the report.",
    "The concern was raised by the client.",
    "The complaint was handled professionally.",
    "The dispute was settled out of court.",
    "The agreement was signed by both parties.",
    "The contract runs for three years.",
    "The terms are negotiable.",
    "The conditions apply in all cases.",
    "The rule was enforced strictly.",
    "The policy was updated recently.",
    "The guideline is available online.",
    "The standard is widely adopted.",
    "The practice is common in the industry.",
    "The convention is to use lowercase.",
    "The format is standard across the platform.",
    "The structure is consistent throughout.",
    "The pattern is easy to recognize.",
    "The trend is expected to continue.",
    "The forecast was accurate.",
    "The prediction was correct.",
    "The expectation was met.",
    "The goal was achieved.",
    "The target was exceeded.",
    "The objective is to reduce costs.",
    "The aim is to improve quality.",
    "The purpose is to simplify the process.",
    "The intent was clear from the start.",
    "The motivation was purely practical.",
    "The reason was never explained.",
    "The cause was identified later.",
    "The effect was immediate.",
    "The consequence was unintended.",
    "The implication is serious.",
    "The significance should not be overlooked.",
    "The relevance is obvious.",
    "The importance cannot be overstated.",
    "The value is significant.",
    "The benefit outweighs the cost.",
    "The advantage is substantial.",
    "The gain is marginal.",
    "The loss was minimal.",
    "The damage was repaired.",
    "The issue was fixed.",
    "The bug was reported and resolved.",
    "The feature was requested by users.",
    "The improvement was noticeable.",
    "The enhancement was welcome.",
    "The modification was minor.",
    "The revision was thorough.",
    "The update was applied successfully.",
    "The change was documented.",
    "The adjustment was necessary.",
    "The correction was made quickly.",
]


def load_clean_text(paths: list[str], text_key: str = "text", label_key: str = "labels", min_words: int = 3) -> list[str]:
    """Load clean (non-slop) text from JSONL or plain text files."""
    out = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            for item in load_jsonl(p):
                t = item.get(text_key)
                if not isinstance(t, str) or len(t.strip().split()) < min_words:
                    continue
                labels = item.get(label_key, [])
                if labels and isinstance(labels[0], (int, float)) and labels[0] == 1:
                    continue
                out.append(t.strip())
        else:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) >= min_words:
                        out.append(line)
    return out


def build_examples(
    clean_sentences: list[str],
    slop_per_text: int,
    seed: int,
    min_words: int = 3,
    sloppifier_kw: dict | None = None,
) -> list[dict]:
    """Build list of {"text": str, "labels": list[int], "difficulty": str} with clean and slop examples."""
    sloppifier_kw = sloppifier_kw or {}
    rng = random.Random(seed)
    easy_ratio = sloppifier_kw.pop("easy_ratio", 0.5)
    medium_ratio = sloppifier_kw.pop("medium_ratio", 0.3)
    hard_ratio = sloppifier_kw.pop("hard_ratio", 0.2)
    examples = []
    for sent in clean_sentences:
        words = sent.split()
        if len(words) < min_words:
            continue
        examples.append({"text": sent, "labels": [0] * len(words), "difficulty": "easy"})
        total_r = easy_ratio + medium_ratio + hard_ratio
        weights = [easy_ratio, medium_ratio, hard_ratio] if total_r > 0 else [1, 0, 0]
        for _ in range(slop_per_text):
            d = rng.choices(["easy", "medium", "hard"], weights=weights)[0]
            s = RuleSloppifier.from_difficulty(d, seed=rng.randint(0, 2**31 - 1))
            text, labels = s.sloppify_with_labels(sent)
            if len(text.split()) >= min_words and any(labels):
                examples.append({"text": text, "labels": labels, "difficulty": d})
    return examples


def stratified_split(examples: list[dict], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> tuple[list, list, list]:
    """Split by has_slop (any label 1) for class balance across splits."""
    rng = random.Random(seed)
    clean = [e for e in examples if not any(e["labels"])]
    slop = [e for e in examples if any(e["labels"])]
    rng.shuffle(clean)
    rng.shuffle(slop)
    n_clean, n_slop = len(clean), len(slop)
    n_train_c = max(1, int(n_clean * train_ratio))
    n_val_c = max(0, int(n_clean * val_ratio))
    n_train_s = max(1, int(n_slop * train_ratio))
    n_val_s = max(0, int(n_slop * val_ratio))
    train = clean[:n_train_c] + slop[:n_train_s]
    val = clean[n_train_c:n_train_c + n_val_c] + slop[n_train_s:n_train_s + n_val_s]
    test = clean[n_train_c + n_val_c:] + slop[n_train_s + n_val_s:]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build token-level slop classifier dataset at scale")
    p.add_argument("--input", nargs="*", default=[], help="Paths to good text (JSONL or .txt)")
    p.add_argument("--output-dir", default="data", help="Output directory for train/val/test.jsonl")
    p.add_argument("--target-total", type=int, default=3000, help="Target total examples (clean + slop)")
    p.add_argument("--slop-per-text", type=int, default=3, help="Slop variants per clean sentence")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--filler-prob", type=float, default=0.25)
    p.add_argument("--hedge-prob", type=float, default=0.2)
    p.add_argument("--repeat-prob", type=float, default=0.15)
    p.add_argument("--generic-noun-prob", type=float, default=0.3)
    p.add_argument("--template-prob", type=float, default=0.2)
    p.add_argument("--easy-ratio", type=float, default=0.5, help="Fraction of slop examples that are easy (curriculum)")
    p.add_argument("--medium-ratio", type=float, default=0.3, help="Fraction of slop examples that are medium")
    p.add_argument("--hard-ratio", type=float, default=0.2, help="Fraction of slop examples that are hard")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    clean = load_clean_text(args.input, min_words=args.min_words)
    if len(clean) < 10:
        clean = CLEAN_CORPUS.copy()
    while len(clean) < args.target_total // (1 + args.slop_per_text):
        clean.extend(rng.choices(CLEAN_CORPUS, k=min(100, args.target_total)))
    clean = list(dict.fromkeys(clean))
    rng.shuffle(clean)
    n_need = max(10, args.target_total // (1 + args.slop_per_text))
    clean = clean[:n_need]
    sloppifier_kw = {
        "filler_prob": args.filler_prob,
        "hedge_prob": args.hedge_prob,
        "repeat_sentence_prob": args.repeat_prob,
        "generic_noun_prob": args.generic_noun_prob,
        "template_prob": args.template_prob,
        "easy_ratio": args.easy_ratio,
        "medium_ratio": args.medium_ratio,
        "hard_ratio": args.hard_ratio,
    }
    examples = build_examples(
        clean,
        slop_per_text=args.slop_per_text,
        seed=args.seed,
        min_words=args.min_words,
        sloppifier_kw=sloppifier_kw,
    )
    extra_seed = args.seed + 9999
    while len(examples) < args.target_total:
        batch = rng.choices(clean, k=min(50, len(clean)))
        more = build_examples(batch, args.slop_per_text, extra_seed, args.min_words, sloppifier_kw)
        extra_seed += 1
        examples.extend(more)
        if not more:
            break
    if len(examples) > args.target_total:
        rng.shuffle(examples)
        examples = examples[:args.target_total]
    train, val, test = stratified_split(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(data)} examples to {path}")
    n_slop_tok = sum(sum(ex["labels"]) for ex in examples)
    n_tok = sum(len(ex["labels"]) for ex in examples)
    print(f"Total examples: {len(examples)} (train={len(train)}, val={len(val)}, test={len(test)})")
    print(f"Slop token fraction: {n_slop_tok / max(1, n_tok):.2%}")


if __name__ == "__main__":
    main()
