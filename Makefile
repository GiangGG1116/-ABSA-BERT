.PHONY: install ate atsc fmt
install:
	pip install --upgrade pip
	pip install -r requirements.txt

ate:
	python -m absa.train_ate

atsc:
	python -m absa.train_atsc

fmt:
	python -m pip install ruff
	ruff format src