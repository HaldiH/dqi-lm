CONFIG ?= configs/config.yaml
PYTHON = .venv/bin/python

.PHONY: env data train

env:
	uv sync

split:
	$(PYTHON) scripts/split_dataset.py --config $(CONFIG)

preprocess:
	$(PYTHON) scripts/data_prep.py --config $(CONFIG)

data: split preprocess

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)
