"""Factory for creating token classifier with optional LoRA (PEFT) and Unsloth."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from hill_climb.config import ModelConfig


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def _freeze_base_except_lora_and_head(model: nn.Module) -> None:
    """Set requires_grad=False for all parameters except LoRA and classifier head."""
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "classifier" in name or "modules_to_save" in name:
            continue
        param.requires_grad = False


def _create_with_unsloth(config: ModelConfig):
    """Create model + tokenizer using Unsloth (causal LMs only). Returns (model, tokenizer) or None."""
    if not _unsloth_available():
        return None
    backbone_name = getattr(config, "backbone_name", "distilbert-base-uncased")
    # Unsloth supports Llama, Mistral, etc. - typically causal LMs
    if "distilbert" in backbone_name.lower() or "bert" in backbone_name.lower():
        return None
    try:
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer

        max_seq_length = int(getattr(config, "max_length", 512))
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=backbone_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        r = int(getattr(config, "lora_r", 16))
        target_modules = getattr(config, "lora_target_modules", None) or ["q_proj", "v_proj", "k_proj", "o_proj"]
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=int(getattr(config, "lora_alpha", 32)),
            lora_dropout=float(getattr(config, "lora_dropout", 0.05)),
        )
        hidden_size = model.config.hidden_size
        num_labels = int(getattr(config, "num_labels", 2))
        dropout = float(getattr(config, "dropout", 0.1))
        # Wrap in a classifier interface (same as EncoderSlopClassifier)
        wrapper = _UnslothSlopClassifierWrapper(model, hidden_size, num_labels, dropout)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return wrapper, tokenizer
    except Exception:
        return None


class _UnslothSlopClassifierWrapper(nn.Module):
    """Wraps Unsloth causal LM with a token classification head (same interface as EncoderSlopClassifier)."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        from transformers.modeling_outputs import TokenClassifierOutput

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs[0]
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)

    def score_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1)
            return probs[..., 1]

    def doc_slop_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slop_probs = self.score_tokens(input_ids, attention_mask)
        if attention_mask is not None:
            mask = (attention_mask == 1).float()
            masked = slop_probs * mask
            count = mask.sum(dim=1).clamp(min=1)
            return masked.sum(dim=1) / count
        return slop_probs.mean(dim=1)


def create_classifier_and_tokenizer(config: ModelConfig):
    """Create token classifier and tokenizer from config.

    Applies LoRA via PEFT (or Unsloth when use_unsloth=True and supported).
    Optionally freezes base weights except LoRA + classifier head.
    Returns (model, tokenizer). Rest of training code is unchanged.
    """
    from transformers import AutoModel, AutoTokenizer

    from hill_climb.models.token_classifier import EncoderSlopClassifier

    backbone_name = getattr(config, "backbone_name", "distilbert-base-uncased")
    backbone_type = getattr(config, "backbone_type", "encoder")
    use_unsloth = getattr(config, "use_unsloth", False)
    use_lora = getattr(config, "use_lora", True)
    freeze_base = getattr(config, "freeze_base", True)

    if use_unsloth and _unsloth_available() and backbone_type == "causal":
        result = _create_with_unsloth(config)
        if result is not None:
            return result

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EncoderSlopClassifier(
        backbone_name=backbone_name,
        num_labels=int(getattr(config, "num_labels", 2)),
        dropout=float(getattr(config, "dropout", 0.1)),
        max_length=int(getattr(config, "max_length", 512)),
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        target_modules = getattr(config, "lora_target_modules", None) or ["q_proj", "v_proj"]
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        if "distilbert" in backbone_name.lower() or "bert" in backbone_name.lower():
            if target_modules == ["q_proj", "v_proj"]:
                target_modules = ["q_lin", "k_lin", "v_lin"]
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(getattr(config, "lora_r", 16)),
            lora_alpha=int(getattr(config, "lora_alpha", 32)),
            lora_dropout=float(getattr(config, "lora_dropout", 0.05)),
            target_modules=target_modules,
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        if freeze_base:
            _freeze_base_except_lora_and_head(model)

    return model, tokenizer
