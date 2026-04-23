import ast
import torch
import pytest
from unittest.mock import MagicMock
from absa.data import ATEDataset, ATSCDataset, collate_ate, collate_atsc, _parse_str_list


# ── _parse_str_list ───────────────────────────────────────────────────────────

def test_parse_str_list_basic():
    assert _parse_str_list("['the', 'bread', 'is']") == ["the", "bread", "is"]


def test_parse_str_list_ints():
    assert _parse_str_list("[0, 1, -1]") == ["0", "1", "-1"]


# ── ATEDataset ────────────────────────────────────────────────────────────────

def _mock_tokenizer(vocab=None):
    tk = MagicMock()
    tk.pad_token_id = 0
    tk.cls_token_id = 101
    tk.sep_token_id = 102
    # tokenize: return single piece per word
    tk.tokenize = lambda word: [word]
    tk.convert_tokens_to_ids = lambda pieces: [hash(p) % 1000 + 1 for p in pieces]
    return tk


def _make_df(n=3):
    import pandas as pd
    rows = {
        "tokens": ["['food', 'is', 'great']"] * n,
        "tags":   ["[1, 0, 0]"] * n,
        "pols":   ["[2, -1, -1]"] * n,
    }
    return pd.DataFrame(rows)


def test_ate_dataset_len():
    df = _make_df(5)
    ds = ATEDataset(df, _mock_tokenizer())
    assert len(ds) == 5


def test_ate_dataset_item_shapes():
    df = _make_df(1)
    ds = ATEDataset(df, _mock_tokenizer())
    ids, labels = ds[0]
    assert ids.shape == labels.shape
    # CLS + 3 tokens + SEP = 5
    assert ids.shape[0] == 5


def test_ate_dataset_cls_sep_ignored():
    df = _make_df(1)
    ds = ATEDataset(df, _mock_tokenizer())
    _, labels = ds[0]
    assert labels[0].item() == -100   # CLS
    assert labels[-1].item() == -100  # SEP


# ── ATSCDataset ───────────────────────────────────────────────────────────────

def test_atsc_dataset_len():
    df = _make_df(4)
    ds = ATSCDataset(df, _mock_tokenizer())
    assert len(ds) == 4


def test_atsc_dataset_item_types():
    df = _make_df(1)
    ds = ATSCDataset(df, _mock_tokenizer())
    ids, seg, label = ds[0]
    assert ids.dtype == torch.long
    assert seg.dtype == torch.long
    assert label.dtype == torch.long


# ── collate functions ─────────────────────────────────────────────────────────

def test_collate_ate_pads():
    t1 = (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 0]))
    t2 = (torch.tensor([1, 2]), torch.tensor([0, 1]))
    ids, labels, masks = collate_ate([t1, t2], pad_id=0)
    assert ids.shape == (2, 3)
    assert ids[1, -1].item() == 0       # padded
    assert masks[1, -1].item() == 0     # masked out
    assert labels[1, -1].item() == -100  # ignored in loss


def test_collate_atsc_stacks_y():
    t1 = (torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor(2))
    t2 = (torch.tensor([1]),     torch.tensor([0]),    torch.tensor(0))
    ids, seg, masks, y = collate_atsc([t1, t2], pad_id=0)
    assert y.shape == (2,)
    assert y[0].item() == 2
    assert y[1].item() == 0
