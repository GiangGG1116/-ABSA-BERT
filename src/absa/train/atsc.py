import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from absa.config import TrainConfig
from absa.data import ATSCDataset, make_atsc_collate
from absa.models import ATSCBert
from absa.utils import avg, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT for Aspect-Term Sentiment Classification")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--train_csv", default="./data/raw/restaurants_train.csv")
    p.add_argument("--valid_csv", default="./data/raw/restaurants_test.csv")
    p.add_argument("--save_dir", default="./models/atsc")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_length=args.max_length,
        seed=args.seed,
    ).ensure()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)

    train_ds = ATSCDataset(train_df, tokenizer, max_length=cfg.max_length)
    valid_ds = ATSCDataset(valid_df, tokenizer, max_length=cfg.max_length)

    collate_fn = make_atsc_collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ATSCBert(cfg.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_loss = float("inf")
    ckpt = Path(cfg.save_dir) / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        # --- training ---
        model.train()
        train_losses = []
        for ids, seg, masks, y in tqdm(train_loader, desc=f"[ATSC Train] epoch {epoch}"):
            ids, seg, masks, y = ids.to(device), seg.to(device), masks.to(device), y.to(device)
            optimizer.zero_grad()
            loss, _ = model(ids, masks, seg, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        # --- validation ---
        model.eval()
        eval_losses = []
        correct = total = 0
        with torch.no_grad():
            for ids, seg, masks, y in tqdm(valid_loader, desc=f"[ATSC Eval]  epoch {epoch}"):
                ids, seg, masks, y = ids.to(device), seg.to(device), masks.to(device), y.to(device)
                loss, logits = model(ids, masks, seg, y)
                eval_losses.append(loss.item())
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.numel()

        avg_train = avg(train_losses)
        avg_eval = avg(eval_losses)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f}  valid_loss={avg_eval:.4f}  acc={acc:.4f}")
        if avg_eval < best_loss:
            best_loss = avg_eval
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Saved best checkpoint -> {ckpt}")

    print(f"\nTraining complete. Best valid_loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()