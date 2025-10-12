import argparse
import torch
from transformers import BertTokenizer
from absa.models import ATEBert


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--sentence", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = BertTokenizer.from_pretrained(args.model_name)
    model = ATEBert(args.model_name).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    word_pieces = tk.tokenize(args.sentence)
    ids = torch.tensor([tk.convert_tokens_to_ids(word_pieces)], dtype=torch.long).to(device)
    masks = (ids != tk.pad_token_id).long()

    with torch.no_grad():
        logits = model(ids, masks)
        preds = logits.argmax(dim=2)[0].tolist()
    print({"tokens": word_pieces, "preds": preds})

if __name__ == "__main__":
    main()