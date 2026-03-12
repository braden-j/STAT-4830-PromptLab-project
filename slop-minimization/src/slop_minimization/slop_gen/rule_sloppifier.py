"""Rule-based sloppifier: filler phrases, hedging, repetition, generic nouns, templates."""

from __future__ import annotations

import random
import re

# Filler phrases to inject (with optional leading space for insertion)
FILLER_PHRASES = [
    " you know",
    " like",
    " basically",
    " I mean",
    " kind of",
    " sort of",
    " um",
    " uh",
    " to be honest",
    " at the end of the day",
]

# Hedging words to add before verbs/adjectives
HEDGING_WORDS = [
    "might",
    "perhaps",
    "possibly",
    "somewhat",
    "generally",
    "often",
    "could be",
    "tends to",
    "usually",
    "typically",
]

# Sentence-initial templates
SLOP_TEMPLATES = [
    "In today's world, ",
    "It's important to note that ",
    "At the end of the day, ",
    "When you think about it, ",
    "The reality is that ",
    "It goes without saying that ",
    "As we all know, ",
]

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
        seed: int | None = None,
    ):
        self.filler_prob = filler_prob
        self.hedge_prob = hedge_prob
        self.repeat_sentence_prob = repeat_sentence_prob
        self.generic_noun_prob = generic_noun_prob
        self.template_prob = template_prob
        self._rng = random.Random(seed)

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
