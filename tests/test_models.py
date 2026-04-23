import torch
import pytest
from unittest.mock import patch, MagicMock
from absa.models import ATEBert, ATSCBert


def _mock_bert_output(hidden_size=768, seq_len=5, batch=2, is_seq=True):
    """Return a mock BertModel that produces deterministic outputs."""
    mock_model = MagicMock()
    mock_model.config.hidden_size = hidden_size
    if is_seq:
        mock_model.return_value.last_hidden_state = torch.zeros(batch, seq_len, hidden_size)
    else:
        mock_model.return_value.pooler_output = torch.zeros(batch, hidden_size)
    return mock_model


# ── ATEBert ───────────────────────────────────────────────────────────────────

@patch("absa.models.BertModel.from_pretrained")
def test_ate_bert_forward_no_labels(mock_bert):
    B, T, H = 2, 5, 768
    mock_bert.return_value.config.hidden_size = H
    mock_bert.return_value.return_value.last_hidden_state = torch.zeros(B, T, H)

    model = ATEBert("bert-base-uncased", num_labels=3)
    ids   = torch.zeros(B, T, dtype=torch.long)
    masks = torch.ones(B, T, dtype=torch.long)

    logits = model(ids, masks)
    assert logits.shape == (B, T, 3)


@patch("absa.models.BertModel.from_pretrained")
def test_ate_bert_forward_with_labels(mock_bert):
    B, T, H = 2, 5, 768
    mock_bert.return_value.config.hidden_size = H
    mock_bert.return_value.return_value.last_hidden_state = torch.zeros(B, T, H)

    model = ATEBert("bert-base-uncased", num_labels=3)
    ids    = torch.zeros(B, T, dtype=torch.long)
    masks  = torch.ones(B, T, dtype=torch.long)
    labels = torch.zeros(B, T, dtype=torch.long)

    loss, logits = model(ids, masks, labels)
    assert loss.ndim == 0           # scalar
    assert logits.shape == (B, T, 3)


# ── ATSCBert ──────────────────────────────────────────────────────────────────

@patch("absa.models.BertModel.from_pretrained")
def test_atsc_bert_forward_no_labels(mock_bert):
    B, T, H = 2, 5, 768
    mock_bert.return_value.config.hidden_size = H
    mock_bert.return_value.return_value.pooler_output = torch.zeros(B, H)

    model = ATSCBert("bert-base-uncased", num_labels=3)
    ids  = torch.zeros(B, T, dtype=torch.long)
    masks = torch.ones(B, T, dtype=torch.long)
    seg  = torch.zeros(B, T, dtype=torch.long)

    logits = model(ids, masks, seg)
    assert logits.shape == (B, 3)


@patch("absa.models.BertModel.from_pretrained")
def test_atsc_bert_forward_with_labels(mock_bert):
    B, T, H = 2, 5, 768
    mock_bert.return_value.config.hidden_size = H
    mock_bert.return_value.return_value.pooler_output = torch.zeros(B, H)

    model = ATSCBert("bert-base-uncased", num_labels=3)
    ids    = torch.zeros(B, T, dtype=torch.long)
    masks  = torch.ones(B, T, dtype=torch.long)
    seg    = torch.zeros(B, T, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.long)

    loss, logits = model(ids, masks, seg, labels)
    assert loss.ndim == 0
    assert logits.shape == (B, 3)
