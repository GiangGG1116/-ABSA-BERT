import torch
import torch.nn as nn
from transformers import BertModel


class ATEBert(nn.Module):
    """BERT + linear head for Aspect Term Extraction (sequence labeling)."""

    def __init__(self, model_name: str, num_labels: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)  # (B, T, num_labels)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits


class ATSCBert(nn.Module):
    """BERT + linear head for Aspect-Term Sentiment Classification."""

    def __init__(self, model_name: str, num_labels: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).pooler_output  # [CLS] representation  (B, hidden_size)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_labels)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits