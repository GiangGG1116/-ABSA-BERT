import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from absa.config import TrainConfig
from absa.utils import set_seed
from absa.data import ABSAATEDataset, collate_ate
from absa.models import ATEBert


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--train_csv", default="./data/restaurants_train.csv")
    p.add_argument("--valid_csv", default="./data/restaurants_test.csv")
    p.add_argument("--save_dir", default="./models/ate")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(model_name=args.model_name, train_csv=args.train_csv, valid_csv=args.valid_csv,
                      save_dir=args.save_dir, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                      seed=args.seed).ensure()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = BertTokenizer.from_pretrained(cfg.model_name)

    train_df = pd.read_csv(cfg.train_csv)
    valid_df = pd.read_csv(cfg.valid_csv)

    train_ds = ABSAATEDataset(train_df, tk)
    valid_ds = ABSAATEDataset(valid_df, tk)

    collate_fn = lambda b: collate_ate(b, pad_id=tk.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = ATEBert(cfg.model_name).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_loss = float("inf")
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.save_dir) / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for ids, labels, masks in tqdm(train_loader, desc=f"[Train] epoch {epoch}"):
            ids, labels, masks = ids.to(device), labels.to(device), masks.to(device)
            loss, _ = model(ids, masks, labels)
            train_losses.append(loss.item())
            loss.backward()
            optim.step(); optim.zero_grad()

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for ids, labels, masks in tqdm(valid_loader, desc=f"[Eval] epoch {epoch}"):
                ids, labels, masks = ids.to(device), labels.to(device), masks.to(device)
                loss, _ = model(ids, masks, labels)
                eval_losses.append(loss.item())
        avg_train, avg_eval = sum(train_losses)/len(train_losses), sum(eval_losses)/len(eval_losses)
        print(f"Epoch {epoch}: train_loss={avg_train:.4f} valid_loss={avg_eval:.4f}")
        if avg_eval < best_loss:
            best_loss = avg_eval
            torch.save(model.state_dict(), ckpt)
            print(f"Saved new best to {ckpt}")

    print("Training done. Best valid loss:", best_loss)

if __name__ == "__main__":
    main()