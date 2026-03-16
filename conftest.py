"""Pytest configuration."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def _make_offline_tokenizer():
    """Build a minimal tokenizer that works offline (no HuggingFace Hub)."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<|endoftext|>": 0, "<|pad|>": 1, "[UNK]": 2}
    for i, w in enumerate(
        "the a is you know like kind of basically hi there yo so what i mean clear and".split(), 3
    ):
        vocab[w] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="[UNK]",
    )
    return fast


def get_tokenizer():
    """Return a tokenizer: from cache (gpt2) if possible, else minimal offline tokenizer."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception:
        return _make_offline_tokenizer()
