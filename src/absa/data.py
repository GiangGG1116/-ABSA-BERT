from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class ABSAATEDataset(Dataset):
    """Dataset for ATE (token-level). Columns: tokens, tags, pols
    tokens: python-list-like string e.g., "['the','bread','is']"
    tags:   python-list-like string of ints per token
    pols:   python-list-like string of ints per token (aligned, but unused for loss)
    """
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.tk = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values
        tokens = tokens.replace("'", "").strip("][").split(", ")
        tags = tags.strip("][ ").split(", ")
        pols = pols.strip("][ ").split(", ")

        bert_tokens: List[str] = []
        bert_tags: List[int] = []
        for i in range(len(tokens)):
            ts = self.tk.tokenize(tokens[i])
            bert_tokens += ts
            bert_tags += [int(tags[i])] * len(ts)

        input_ids = self.tk.convert_tokens_to_ids(bert_tokens)
        ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(bert_tags, dtype=torch.long)
        return ids, labels


def collate_ate(batch, pad_id: int):
    ids_list, labels_list = zip(*batch)
    ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)  # -100 ignored in CE
    masks = (ids != pad_id).long()
    return ids, labels, masks


class ABSAATSCdataset(Dataset):
    """Dataset for ATSC (sentence + aspect) from same CSV. Uses first aspect whose pol != -1.
    """
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.tk = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens, tags, pols = self.df.iloc[idx, :3].values
        tokens = tokens.replace("'", "").strip("][").split(", ")
        pols = pols.strip("][ ").split(", ")

        sent_tokens: List[str] = []
        aspect_tokens: List[str] = []
        label = 0
        for i in range(len(tokens)):
            tks = self.tk.tokenize(tokens[i])
            sent_tokens += tks
            if int(pols[i]) != -1:
                aspect_tokens += tks
                label = int(pols[i])
        # [CLS] sent [SEP] aspect
        word_pieces = [self.tk.cls_token] + sent_tokens + [self.tk.sep_token] + aspect_tokens
        token_type_ids = [0] * (1 + len(sent_tokens) + 1) + [1] * len(aspect_tokens)
        input_ids = self.tk.convert_tokens_to_ids(word_pieces)

        ids = torch.tensor(input_ids, dtype=torch.long)
        seg = torch.tensor(token_type_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return ids, seg, y


def collate_atsc(batch, pad_id: int):
    ids_list, seg_list, y_list = zip(*batch)
    ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
    seg = pad_sequence(seg_list, batch_first=True, padding_value=0)
    masks = (ids != pad_id).long()
    y = torch.stack(y_list)
    return ids, seg, masks, y