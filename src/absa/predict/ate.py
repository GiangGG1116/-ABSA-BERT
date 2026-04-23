import argparse

import torch
from transformers import BertTokenizer

from absa.models import ATEBert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ATE inference on a single sentence")
    p.add_argument("--ckpt", required=True, help="Path to saved model checkpoint")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--sentence", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = ATEBert(args.model_name).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    word_pieces = tokenizer.tokenize(args.sentence)
    # Add [CLS] and [SEP] to match the training format
    input_ids = (
        [tokenizer.cls_token_id]
        + tokenizer.convert_tokens_to_ids(word_pieces)
        + [tokenizer.sep_token_id]
    )
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    masks = (ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        logits = model(ids, masks)          # (1, T, num_labels)
        preds = logits.argmax(dim=-1)[0].tolist()

    # Strip CLS and SEP predictions (they were labelled -100 during training)
    token_preds = preds[1:-1]
    print({"tokens": word_pieces, "preds": token_preds})


if __name__ == "__main__":
    main()