import argparse

import torch
from transformers import BertTokenizer

from absa.models import ATSCBert

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ATSC inference on a sentence + aspect pair")
    p.add_argument("--ckpt", required=True, help="Path to saved model checkpoint")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--sentence", required=True)
    p.add_argument("--aspect", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = ATSCBert(args.model_name).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    sent_pieces = tokenizer.tokenize(args.sentence)
    aspect_pieces = tokenizer.tokenize(args.aspect)
    word_pieces = [tokenizer.cls_token] + sent_pieces + [tokenizer.sep_token] + aspect_pieces
    token_type_ids = [0] * (1 + len(sent_pieces) + 1) + [1] * len(aspect_pieces)

    ids = torch.tensor([tokenizer.convert_tokens_to_ids(word_pieces)], dtype=torch.long, device=device)
    seg = torch.tensor([token_type_ids], dtype=torch.long, device=device)
    masks = (ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        logits = model(ids, masks, seg)
        pred = int(logits.argmax(dim=1).item())

    print({
        "sentence": args.sentence,
        "aspect": args.aspect,
        "pred": pred,
        "sentiment": LABEL_MAP.get(pred, str(pred)),
    })


if __name__ == "__main__":
    main()