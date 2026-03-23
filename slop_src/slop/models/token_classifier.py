"""Token-level slop classifier with transformer backbone and per-token head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class EncoderSlopClassifier(nn.Module):
    """Token-level slop classifier using encoder backbone (e.g. DistilBERT) + linear head."""

    def __init__(
        self,
        backbone_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        max_length: int = 512,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def score_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-token slop probabilities (class 1)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(out.logits, dim=-1)
            slop_probs = probs[..., 1]
        return slop_probs

    def doc_slop_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Document-level slop score: mean of per-token slop prob over non-padding tokens."""
        slop_probs = self.score_tokens(input_ids, attention_mask)
        if attention_mask is not None:
            mask = (attention_mask == 1).float()
            masked = slop_probs * mask
            count = mask.sum(dim=1).clamp(min=1)
            return (masked.sum(dim=1) / count)
        return slop_probs.mean(dim=1)


class SlopTokenClassifier(nn.Module):
    """Per-token slop classifier using LM backbone + linear classification head."""

    def __init__(
        self,
        backbone_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        max_length: int = 512,
    ):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        self.num_labels = num_labels
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def score_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-token slop probabilities (class 1)."""
        self.eval()
        with torch.no_grad():
            out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(out.logits, dim=-1)
            slop_probs = probs[..., 1]
        return slop_probs
