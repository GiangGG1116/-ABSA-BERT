import ast
from functools import partial
from typing import List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _parse_str_list(raw: str) -> List[str]:
    """Safely parse a Python-list-like string from CSV into a list of strings."""
    return [str(x) for x in ast.literal_eval(raw)]


class ATEDataset(Dataset):
    """Dataset for Aspect Term Extraction (token-level sequence labeling).

    Expected CSV columns:
      tokens – Python list string of raw words, e.g. "['the', 'bread', 'is']"
      tags   – Python list string of BIO int labels per word
      pols   – Python list string of polarity ints per word (unused in ATE loss)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        self.df = df
        self.tk = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_raw, tags_raw, _ = self.df.iloc[idx, :3].values
        words = _parse_str_list(tokens_raw)
        tags = [int(t) for t in _parse_str_list(tags_raw)]

        bert_tokens: List[str] = []
        bert_tags: List[int] = []
        for word, tag in zip(words, tags):
            pieces = self.tk.tokenize(word)
            bert_tokens.extend(pieces)
            # First sub-token keeps the original label; continuations use -100 (ignored by loss)
            bert_tags.extend([tag] + [-100] * (len(pieces) - 1))

        # Truncate to max_length - 2 to leave room for [CLS] and [SEP]
        max_content = self.max_length - 2
        bert_tokens = bert_tokens[:max_content]
        bert_tags = bert_tags[:max_content]

        input_ids = (
            [self.tk.cls_token_id]
            + self.tk.convert_tokens_to_ids(bert_tokens)
            + [self.tk.sep_token_id]
        )
        labels = [-100] + bert_tags + [-100]  # CLS and SEP are ignored

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_ate(batch: List[Tuple], pad_id: int) -> Tuple[torch.Tensor, ...]:
    ids_list, labels_list = zip(*batch)
    ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    masks = (ids != pad_id).long()
    return ids, labels, masks


def make_ate_collate(pad_id: int):
    """Return a picklable collate function for ATEDataset (safe for DataLoader workers)."""
    return partial(collate_ate, pad_id=pad_id)


class ATSCDataset(Dataset):
    """Dataset for Aspect-Term Sentiment Classification.

    Input format: [CLS] sentence [SEP] aspect

    Expected CSV columns:
      tokens – Python list string of raw words
      tags   – Python list string of BIO int labels per word (unused in ATSC loss)
      pols   – Python list string of polarity ints per word (-1 = not an aspect token)
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        self.df = df
        self.tk = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_raw, _, pols_raw = self.df.iloc[idx, :3].values
        words = _parse_str_list(tokens_raw)
        pols = [int(p) for p in _parse_str_list(pols_raw)]

        sent_pieces: List[str] = []
        aspect_pieces: List[str] = []
        label = 0
        for word, pol in zip(words, pols):
            pieces = self.tk.tokenize(word)
            sent_pieces.extend(pieces)
            if pol != -1:
                aspect_pieces.extend(pieces)
                label = pol

        # Truncate sentence so the full sequence fits within max_length
        # Structure: [CLS] sent [SEP] aspect  →  2 special tokens + sent + aspect
        max_sent = self.max_length - 2 - len(aspect_pieces)
        sent_pieces = sent_pieces[:max(max_sent, 0)]

        word_pieces = [self.tk.cls_token] + sent_pieces + [self.tk.sep_token] + aspect_pieces
        token_type_ids = [0] * (1 + len(sent_pieces) + 1) + [1] * len(aspect_pieces)
        input_ids = self.tk.convert_tokens_to_ids(word_pieces)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def collate_atsc(batch: List[Tuple], pad_id: int) -> Tuple[torch.Tensor, ...]:
    ids_list, seg_list, y_list = zip(*batch)
    ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
    seg = pad_sequence(seg_list, batch_first=True, padding_value=0)
    masks = (ids != pad_id).long()
    y = torch.stack(y_list)
    return ids, seg, masks, y


def make_atsc_collate(pad_id: int):
    """Return a picklable collate function for ATSCDataset (safe for DataLoader workers)."""
    return partial(collate_atsc, pad_id=pad_id)