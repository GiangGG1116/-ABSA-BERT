# 🧠 ABSA-BERT — README.md

## 📘 Overview

This project implements **Aspect-Based Sentiment Analysis (ABSA)** using **BERT** in PyTorch. It includes two tasks:

* **ATE (Aspect Term Extraction):** Identify aspect terms in sentences.
* **ATSC (Aspect Term Sentiment Classification):** Determine the sentiment polarity of the extracted aspects.

The pipeline includes dataset building, tokenization, model training, and prediction modules.

---

## ⚙️ Installation

### 1️⃣ Clone and setup environment

```bash
git clone https://github.com/<your-repo>/ML_nlp.git
cd ML_nlp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ (Optional) Using Conda

```bash
conda create -n mlnlp python=3.10 -y
conda activate mlnlp
pip install -r requirements.txt
```

---

## ⬇️ Download dataset

### 1️⃣ Ensure gdown is installed

If you see error `gdown: command not found`, install it:

```bash
pip install gdown
```

### 2️⃣ Run the script to download data

From the **project root (`ML_nlp/`)**:

```bash
bash scripts/download_data.sh
```

This will:

* Download a Google Drive file (ID inside the script)
* Save it as `data.zip`
* You can then extract it:

```bash
unzip data.zip -d data
```

### 3️⃣ Verify data structure

Ensure you have:

```
./data/restaurants_train.csv
./data/restaurants_test.csv
```

> ⚠️ Note: If your files are in `data/data/...`, move them up one level to `data/`.

---

## 🚀 Training

### 1️⃣ Set up PYTHONPATH

Run once per terminal session:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 2️⃣ Train ATE (Aspect Term Extraction)

```bash
python -m absa.train_ate \
  --model_name bert-base-uncased \
  --train_csv data/restaurants_train.csv \
  --valid_csv data/restaurants_test.csv \
  --epochs 5 --batch_size 32 --lr 1e-5 \
  --save_dir models/ate
```

### 3️⃣ Train ATSC (Aspect Term Sentiment Classification)

```bash
python -m absa.train_atsc \
  --model_name bert-base-uncased \
  --train_csv data/restaurants_train.csv \
  --valid_csv data/restaurants_test.csv \
  --epochs 5 --batch_size 32 --lr 1e-5 \
  --save_dir models/atsc
```

Both scripts automatically save the **best checkpoint** by validation loss.

---

## 🔮 Prediction

### 1️⃣ Predict Aspect Terms (ATE)

```bash
python -m absa.predict_ate --ckpt models/ate/best.pt \
  --model_name bert-base-uncased \
  --sentence "the bread is top notch as well"
```

**Output:**

```json
{"tokens": ["the", "bread", "is", "top", "notch", "as", "well"], "preds": [0, 1, 0, 0, 0, 0, 0]}
```

### 2️⃣ Predict Sentiment (ATSC)

```bash
python -m absa.predict_atsc --ckpt models/atsc/best.pt \
  --model_name bert-base-uncased \
  --sentence "The bread is top notch as well" \
  --aspect "bread"
```

**Output:**

```json
{"sentence": "The bread is top notch as well", "aspect": "bread", "pred": 2}
```

---

## 🧩 Expected Results

| Task | Accuracy |
| ---- | -------- |
| ATE  | ~92.1%   |
| ATSC | ~81.4%   |

---

## 📂 Folder structure

```
ABSA-BERT/
├── data/                      # datasets
├── models/                    # trained checkpoints
├── scripts/
│   └── download_data.sh        # dataset downloader
├── src/absa/                   # source code
│   ├── data.py
│   ├── models.py
│   ├── train_ate.py
│   ├── train_atsc.py
│   ├── predict_ate.py
│   └── predict_atsc.py
└── requirements.txt

```

---

## 📜 License

MIT © 2025 — Trịnh Văn Giang