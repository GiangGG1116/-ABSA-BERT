import argparse
import torch
from transformers import BertTokenizer
from absa.models import ABSABert


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--sentence", required=True)
    p.add_argument("--aspect", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = BertTokenizer.from_pretrained(args.model_name)

    t1 = tk.tokenize(args.sentence)
    t2 = tk.tokenize(args.aspect)
    word_pieces = [tk.cls_token] + t1 + [tk.sep_token] + t2
    seg = [0] * (1 + len(t1) + 1) + [1] * len(t2)

    ids = torch.tensor([tk.convert_tokens_to_ids(word_pieces)], dtype=torch.long).to(device)
    seg = torch.tensor([seg], dtype=torch.long).to(device)
    masks = (ids != tk.pad_token_id).long()

    model = ABSABert(args.model_name).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(ids, masks, seg)
        pred = int(logits.argmax(dim=1).item())
    print({"sentence": args.sentence, "aspect": args.aspect, "pred": pred})

if __name__ == "__main__":
    main()