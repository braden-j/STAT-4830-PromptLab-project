"""Rule-based sloppifier: filler phrases, hedging, repetition, generic nouns, templates."""

from __future__ import annotations

import random
import re
from typing import List, Tuple

# Difficulty presets for curriculum: easy (obvious), medium (lower probs), hard (subtle only)
DIFFICULTY_PRESETS: dict[str, dict] = {
    "easy": {
        "filler_prob": 0.25,
        "hedge_prob": 0.2,
        "repeat_sentence_prob": 0.15,
        "generic_noun_prob": 0.3,
        "template_prob": 0.2,
        "use_only_subtle": False,
    },
    "medium": {
        "filler_prob": 0.12,
        "hedge_prob": 0.1,
        "repeat_sentence_prob": 0.08,
        "generic_noun_prob": 0.15,
        "template_prob": 0.1,
        "use_only_subtle": False,
    },
    "hard": {
        "filler_prob": 0.0,
        "hedge_prob": 0.12,
        "repeat_sentence_prob": 0.0,
        "generic_noun_prob": 0.2,
        "template_prob": 0.0,
        "use_only_subtle": True,
    },
}

# Natural vague phrases (harder to detect, more realistic)
NATURAL_VAGUE_PHRASES_WORDS: List[List[str]] = [
    ["in", "practice"], ["in", "general"], ["in", "many", "cases"],
    ["to", "some", "extent"], ["in", "some", "sense"], ["to", "a", "degree"],
    ["by", "and", "large"], ["for", "the", "most", "part"], ["in", "principle"],
]
NATURAL_VAGUE_PHRASES = [" " + " ".join(p) for p in NATURAL_VAGUE_PHRASES_WORDS]

# Filler phrases to inject (with optional leading space for insertion)
# Word-list form for label tracking (sloppify_with_labels)
FILLER_PHRASES_WORDS: List[List[str]] = [
    ["you", "know"], ["like"], ["basically"], ["I", "mean"], ["kind", "of"], ["sort", "of"],
    ["um"], ["uh"], ["to", "be", "honest"], ["at", "the", "end", "of", "the", "day"],
    ["well"], ["so"], ["actually"], ["literally"], ["right"], ["anyway"], ["or", "whatever"],
    ["and", "stuff"], ["things", "like", "that"],
]
FILLER_PHRASES = [
    " you know", " like", " basically", " I mean", " kind of", " sort of",
    " um", " uh", " to be honest", " at the end of the day",
    " well", " so", " actually", " literally", " right", " anyway",
    " or whatever", " and stuff", " things like that",
]

# Hedging words to add before verbs/adjectives
HEDGING_WORDS = [
    "might", "perhaps", "possibly", "somewhat", "generally", "often",
    "could be", "tends to", "usually", "typically", "maybe", "sometimes",
    "fairly", "rather", "quite", "relatively", "broadly", "largely",
]

# Sentence-initial templates (string and word-list for labels)
SLOP_TEMPLATES_WORDS: List[List[str]] = [
    "In today's world,".split(), "It's important to note that".split(),
    "At the end of the day,".split(), "When you think about it,".split(),
    "The reality is that".split(), "It goes without saying that".split(),
    "As we all know,".split(), "In this day and age,".split(),
    "Needless to say,".split(), "Having said that,".split(), "That being said,".split(),
    "When it comes down to it,".split(), "The fact of the matter is".split(),
]
SLOP_TEMPLATES = [" " + " ".join(t) for t in SLOP_TEMPLATES_WORDS]

# Noun -> more generic term (lower specificity)
GENERIC_NOUN_MAP = {
    "car": "vehicle",
    "cars": "vehicles",
    "doctor": "professional",
    "doctors": "professionals",
    "hospital": "facility",
    "hospitals": "facilities",
    "company": "organization",
    "companies": "organizations",
    "employee": "person",
    "employees": "people",
    "customer": "person",
    "customers": "people",
    "student": "individual",
    "students": "individuals",
    "teacher": "professional",
    "teachers": "professionals",
    "computer": "device",
    "computers": "devices",
    "phone": "device",
    "phones": "devices",
    "problem": "situation",
    "problems": "situations",
    "solution": "approach",
    "solutions": "approaches",
    "result": "outcome",
    "results": "outcomes",
    "idea": "concept",
    "ideas": "concepts",
    "book": "resource",
    "books": "resources",
    "movie": "content",
    "movies": "content",
    "restaurant": "place",
    "restaurants": "places",
    "house": "place",
    "houses": "places",
    "city": "area",
    "cities": "areas",
    "country": "region",
    "countries": "regions",
}


class RuleSloppifier:
    """Configurable rule-based sloppifier."""

    def __init__(
        self,
        filler_prob: float = 0.25,
        hedge_prob: float = 0.2,
        repeat_sentence_prob: float = 0.15,
        generic_noun_prob: float = 0.3,
        template_prob: float = 0.2,
        use_only_subtle: bool = False,
        seed: int | None = None,
    ):
        self.filler_prob = filler_prob
        self.hedge_prob = hedge_prob
        self.repeat_sentence_prob = repeat_sentence_prob
        self.generic_noun_prob = generic_noun_prob
        self.template_prob = template_prob
        self.use_only_subtle = use_only_subtle
        self._rng = random.Random(seed)

    @classmethod
    def from_difficulty(cls, difficulty: str, seed: int | None = None) -> "RuleSloppifier":
        """Create a sloppifier from a curriculum difficulty preset: easy, medium, hard."""
        preset = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["easy"]).copy()
        use_only_subtle = preset.pop("use_only_subtle", False)
        return cls(use_only_subtle=use_only_subtle, seed=seed, **preset)

    def _inject_fillers(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        result = []
        for i, w in enumerate(words):
            result.append(w)
            if i < len(words) - 1 and self._rng.random() < self.filler_prob:
                result.append(self._rng.choice(FILLER_PHRASES).strip())
        return " ".join(result)

    def _add_hedging(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        result = []
        for i, w in enumerate(words):
            if self._rng.random() < self.hedge_prob and w.lower() not in {"the", "a", "an", "and", "or", "but"}:
                result.append(self._rng.choice(HEDGING_WORDS))
            result.append(w)
        return " ".join(result)

    def _repeat_sentence(self, text: str) -> str:
        sents = re.split(r"(?<=[.!?])\s+", text)
        sents = [s.strip() for s in sents if s.strip()]
        if len(sents) < 2:
            return text
        idx = self._rng.randint(0, len(sents) - 1)
        dup = sents[idx]
        insert_pos = self._rng.randint(0, len(sents))
        sents.insert(insert_pos, dup)
        return " ".join(sents)

    def _lower_specificity(self, text: str) -> str:
        words = text.split()
        result = []
        for w in words:
            key = w.lower().rstrip(".,;:!?")
            if key in GENERIC_NOUN_MAP and self._rng.random() < self.generic_noun_prob:
                repl = GENERIC_NOUN_MAP[key]
                if w[0].isupper():
                    repl = repl.capitalize()
                if not key[-1].isalnum() and key[-1] in ".,;:!?":
                    repl += w[len(key):]
                result.append(repl)
            else:
                result.append(w)
        return " ".join(result)

    def _add_template(self, text: str) -> str:
        if not text.strip():
            return text
        if self._rng.random() >= self.template_prob:
            return text
        template = self._rng.choice(SLOP_TEMPLATES)
        return template + text.strip()[0].lower() + text.strip()[1:] if len(text) > 1 else template + text

    def sloppify(self, text: str) -> str:
        """Apply all rules in a random order for variety."""
        if not text or not text.strip():
            return text
        t = text.strip()
        ops = [
            self._inject_fillers,
            self._add_hedging,
            self._repeat_sentence,
            self._lower_specificity,
            self._add_template,
        ]
        self._rng.shuffle(ops)
        for op in ops:
            t = op(t)
        return t

    def __call__(self, text: str) -> str:
        return self.sloppify(text)

    def _tokens_to_text_and_labels(self, tokens: List[Tuple[str, int]]) -> Tuple[str, List[int]]:
        if not tokens:
            return "", []
        text = " ".join(w for w, _ in tokens)
        labels = [l for _, l in tokens]
        return text, labels

    def _inject_fillers_tokens(self, tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        if len(tokens) < 3:
            return tokens
        out: List[Tuple[str, int]] = []
        for i, (w, l) in enumerate(tokens):
            out.append((w, l))
            if i < len(tokens) - 1 and self._rng.random() < self.filler_prob:
                phrase = self._rng.choice(FILLER_PHRASES_WORDS)
                for pw in phrase:
                    out.append((pw, 1))
        return out

    def _inject_natural_vague_tokens(self, tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Inject natural vague phrases (hard difficulty)."""
        if len(tokens) < 3:
            return tokens
        out: List[Tuple[str, int]] = []
        for i, (w, l) in enumerate(tokens):
            if i < len(tokens) - 1 and self._rng.random() < 0.15:
                phrase = self._rng.choice(NATURAL_VAGUE_PHRASES_WORDS)
                for pw in phrase:
                    out.append((pw, 1))
            out.append((w, l))
        return out

    def _add_hedging_tokens(self, tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        if not tokens:
            return tokens
        out: List[Tuple[str, int]] = []
        for w, l in tokens:
            if self._rng.random() < self.hedge_prob and w.lower() not in {"the", "a", "an", "and", "or", "but"}:
                out.append((self._rng.choice(HEDGING_WORDS), 1))
            out.append((w, l))
        return out

    def _repeat_sentence_tokens(self, tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        if len(tokens) < 5:
            return tokens
        text, _ = self._tokens_to_text_and_labels(tokens)
        sents = re.split(r"(?<=[.!?])\s+", text)
        sents = [s.strip() for s in sents if s.strip()]
        if len(sents) < 2:
            return tokens
        words = text.split()
        boundaries = [0]
        idx = 0
        for s in sents:
            n = len(s.split())
            idx += n
            boundaries.append(idx)
        sent_idx = self._rng.randint(0, len(sents) - 1)
        start, end = boundaries[sent_idx], boundaries[sent_idx + 1]
        segment = tokens[start:end]
        insert_pos = self._rng.randint(0, len(boundaries) - 1)
        insert_idx = boundaries[insert_pos]
        dup = [(w, 1) for w, _ in segment]
        return tokens[:insert_idx] + dup + tokens[insert_idx:]

    def _lower_specificity_tokens(self, tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for w, l in tokens:
            key = w.lower().rstrip(".,;:!?")
            if key in GENERIC_NOUN_MAP and self._rng.random() < self.generic_noun_prob:
                repl = GENERIC_NOUN_MAP[key]
                if w[0].isupper():
                    repl = repl.capitalize()
                if not key[-1].isalnum() and key[-1] in ".,;:!?":
                    repl += w[len(key):]
                out.append((repl, 1))
            else:
                out.append((w, l))
        return out

    def _add_template_tokens(self, tokens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        if self._rng.random() >= self.template_prob or not tokens:
            return tokens
        template = self._rng.choice(SLOP_TEMPLATES_WORDS)
        prefix = [(w, 1) for w in template]
        if tokens:
            w0, l0 = tokens[0]
            if len(w0) > 0 and w0[0].isupper():
                w0 = w0[0].lower() + w0[1:]
            tokens = [(w0, l0)] + tokens[1:]
        return prefix + tokens

    def sloppify_with_labels(self, text: str) -> Tuple[str, List[int]]:
        """Apply rules and return (text, labels) with labels aligned to whitespace-tokenized words.
        Label 1 = slop (filler, hedge, repeated, generic replacement, template); 0 = clean.
        When use_only_subtle is True, only hedging, generic nouns, and natural vague phrases are used.
        """
        if not text or not text.strip():
            return text.strip(), []
        tokens: List[Tuple[str, int]] = [(w, 0) for w in text.strip().split()]
        if self.use_only_subtle:
            ops = [
                self._inject_natural_vague_tokens,
                self._add_hedging_tokens,
                self._lower_specificity_tokens,
            ]
        else:
            ops = [
                self._inject_fillers_tokens,
                self._add_hedging_tokens,
                self._repeat_sentence_tokens,
                self._lower_specificity_tokens,
                self._add_template_tokens,
            ]
        self._rng.shuffle(ops)
        for op in ops:
            tokens = op(tokens)
        return self._tokens_to_text_and_labels(tokens)


def sloppify(
    text: str,
    filler_prob: float = 0.25,
    hedge_prob: float = 0.2,
    repeat_sentence_prob: float = 0.15,
    generic_noun_prob: float = 0.3,
    template_prob: float = 0.2,
    seed: int | None = None,
) -> str:
    """One-shot sloppify with default rules."""
    s = RuleSloppifier(
        filler_prob=filler_prob,
        hedge_prob=hedge_prob,
        repeat_sentence_prob=repeat_sentence_prob,
        generic_noun_prob=generic_noun_prob,
        template_prob=template_prob,
        seed=seed,
    )
    return s.sloppify(text)


def sloppify_with_labels(
    text: str,
    filler_prob: float = 0.25,
    hedge_prob: float = 0.2,
    repeat_sentence_prob: float = 0.15,
    generic_noun_prob: float = 0.3,
    template_prob: float = 0.2,
    seed: int | None = None,
) -> Tuple[str, List[int]]:
    """One-shot sloppify returning (text, word-level labels). Labels align to text.split()."""
    s = RuleSloppifier(
        filler_prob=filler_prob,
        hedge_prob=hedge_prob,
        repeat_sentence_prob=repeat_sentence_prob,
        generic_noun_prob=generic_noun_prob,
        template_prob=template_prob,
        seed=seed,
    )
    return s.sloppify_with_labels(text)
