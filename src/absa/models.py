import torch
import torch.nn as nn
from transformers import BertModel

class ATEBert(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.classifier(outputs.last_hidden_state)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits

class ABSABert(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        pooled = out.pooler_output  # [CLS]
        logits = self.classifier(pooled)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits