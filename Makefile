CONFIG ?= configs/config.yaml
PYTHON = .venv/bin/python

.PHONY: env data train

env:
	uv sync

data:
	$(PYTHON) scripts/data_prep.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)
