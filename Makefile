.PHONY: install install-dev data train-ate train-atsc predict-ate predict-atsc \
        test lint fmt type-check docker-build docker-run clean

# ── Installation ──────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"

# ── Data ──────────────────────────────────────────────────────────────────────
data:
	bash scripts/download_data.sh

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_CSV ?= ./data/raw/restaurants_train.csv
VALID_CSV ?= ./data/raw/restaurants_test.csv

train-ate:
	python -m absa.train.ate \
		--train_csv $(TRAIN_CSV) \
		--valid_csv $(VALID_CSV) \
		--save_dir  ./outputs/ate

train-atsc:
	python -m absa.train.atsc \
		--train_csv $(TRAIN_CSV) \
		--valid_csv $(VALID_CSV) \
		--save_dir  ./outputs/atsc

# ── Prediction (example usage) ────────────────────────────────────────────────
predict-ate:
	python -m absa.predict.ate \
		--ckpt     ./outputs/ate/best.pt \
		--sentence "The food was great but the service was slow."

predict-atsc:
	python -m absa.predict.atsc \
		--ckpt     ./outputs/atsc/best.pt \
		--sentence "The food was great but the service was slow." \
		--aspect   "food"

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ --cov=src/absa --cov-report=term-missing

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

# ── Docker ────────────────────────────────────────────────────────────────────
IMAGE ?= absa-bert:latest

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/outputs:/app/outputs \
		$(IMAGE) make train-ate

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
