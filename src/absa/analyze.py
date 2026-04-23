import argparse
import ast
import json
from pathlib import Path
from statistics import mean, median

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze ABSA dataset CSV files")
    p.add_argument("--train_csv", default="./data/raw/restaurants_train.csv")
    p.add_argument("--valid_csv", default="./data/raw/restaurants_test.csv")
    p.add_argument("--out_json", default="./outputs/data_analysis.json")
    return p.parse_args()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise ValueError(f"Missing required column. Expected one of: {candidates}")


def _parse_list_cell(cell: str) -> list:
    if not isinstance(cell, str):
        return []
    value = ast.literal_eval(cell)
    return value if isinstance(value, list) else []


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * q))
    return float(s[idx])


def analyze_split(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    tokens_col = _find_col(df, ["tokens"])
    tags_col = _find_col(df, ["tags"])
    pols_col = _find_col(df, ["pols", "polarities"])

    tokens_list = [_parse_list_cell(v) for v in df[tokens_col]]
    tags_list = [_parse_list_cell(v) for v in df[tags_col]]
    pols_list = [_parse_list_cell(v) for v in df[pols_col]]

    sent_lens = [len(x) for x in tokens_list]
    aspect_counts = [sum(1 for t in tags if int(t) > 0) for tags in tags_list]

    polarity_counts: dict[str, int] = {"0": 0, "1": 0, "2": 0, "other": 0}
    for row_pols in pols_list:
        for pol in row_pols:
            p = int(pol)
            if p == -1:
                continue
            if str(p) in polarity_counts:
                polarity_counts[str(p)] += 1
            else:
                polarity_counts["other"] += 1

    unique_tokens = set()
    for row_tokens in tokens_list:
        unique_tokens.update(str(tok).lower() for tok in row_tokens)

    return {
        "csv_path": csv_path,
        "num_rows": int(len(df)),
        "num_duplicate_rows": int(df.duplicated().sum()),
        "sentence_length": {
            "min": int(min(sent_lens)) if sent_lens else 0,
            "max": int(max(sent_lens)) if sent_lens else 0,
            "mean": float(mean(sent_lens)) if sent_lens else 0.0,
            "median": float(median(sent_lens)) if sent_lens else 0.0,
            "p90": _percentile(sent_lens, 0.9),
        },
        "aspect_tokens_per_row": {
            "mean": float(mean(aspect_counts)) if aspect_counts else 0.0,
            "max": int(max(aspect_counts)) if aspect_counts else 0,
            "rows_without_aspect": int(sum(1 for n in aspect_counts if n == 0)),
        },
        "polarity_token_counts": polarity_counts,
        "vocab_size_lower": int(len(unique_tokens)),
    }


def main() -> None:
    args = parse_args()
    report = {
        "train": analyze_split(args.train_csv),
        "valid": analyze_split(args.valid_csv),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Saved analysis report to:", out_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
