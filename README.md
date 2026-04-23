# ABSA-BERT

ABSA-BERT is an Aspect-Based Sentiment Analysis project built with BERT and PyTorch. It includes two tasks:
- `ATE` (Aspect Term Extraction): extract aspect tokens in a sentence.
- `ATSC` (Aspect Term Sentiment Classification): predict sentiment for a target aspect.

## 1. Requirements

- Python `>=3.10`
- `uv` is recommended for environment and dependency management

## 2. Installation

From the project root (`ABSA-BERT`):

```bash
uv sync
```

If you want editable install plus development dependencies:

```bash
uv sync --extra dev
```

## 3. Data

Default dataset paths:
- `./data/raw/restaurants_train.csv`
- `./data/raw/restaurants_test.csv`

Download data with:

```bash
bash scripts/download_data.sh
```

Note: training and prediction expect CSV files with 3 list-string columns:
- `Tokens`
- `Tags`
- `Polarities`

## 4. Train

### Train ATE

```bash
uv run python -m absa.train.ate \
  --train_csv ./data/raw/restaurants_train.csv \
  --valid_csv ./data/raw/restaurants_test.csv \
  --save_dir ./outputs/ate
```

### Train ATSC

```bash
uv run python -m absa.train.atsc \
  --train_csv ./data/raw/restaurants_train.csv \
  --valid_csv ./data/raw/restaurants_test.csv \
  --save_dir ./outputs/atsc
```

The best checkpoint is saved as `best.pt` inside `save_dir`.

## 5. Inference

### Predict ATE

```bash
uv run python -m absa.predict.ate \
  --ckpt ./outputs/ate/best.pt \
  --sentence "The food was great but the service was slow."
```

### Predict ATSC

```bash
uv run python -m absa.predict.atsc \
  --ckpt ./outputs/atsc/best.pt \
  --sentence "The food was great but the service was slow." \
  --aspect "food"
```

## 6. Data Analysis (EDA)

Built-in data analysis command:

```bash
uv run absa-analyze-data
```

Statistics are printed to the console and saved to:
- `./outputs/data_analysis.json`

You can pass custom paths:

```bash
uv run absa-analyze-data \
  --train_csv ./data/raw/restaurants_train.csv \
  --valid_csv ./data/raw/restaurants_test.csv \
  --out_json ./outputs/my_analysis.json
```

## 7. Test and Lint

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check src tests
```

Format code:

```bash
uv run ruff format src tests
```

## 8. Docker

Build image:

```bash
docker build -t absa-bert:latest .
```

Run ATE training in Docker (if your machine has GPU support):

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  absa-bert:latest make train-ate
```

## 9. Main Project Structure

```text
ABSA-BERT/
|- configs/
|- data/
|  |- raw/
|  |- processed/
|- outputs/
|- scripts/
|- src/absa/
|  |- analyze.py
|  |- config.py
|  |- data.py
|  |- models.py
|  |- predict/
|  |- train/
|- tests/
|- pyproject.toml
|- Makefile
```