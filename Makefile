CONFIG ?= configs/config.yaml
PYTHON = .venv/bin/python
CHECKPOINT_DIR = .checkpoints

# Create checkpoint directory
$(CHECKPOINT_DIR):
	mkdir -p $(CHECKPOINT_DIR)

env:
	uv sync

# Merge target - creates checkpoint after successful execution
merge: $(CHECKPOINT_DIR)/merge.done

$(CHECKPOINT_DIR)/merge.done: | $(CHECKPOINT_DIR)
	$(PYTHON) scripts/merge_us_debate_data.py --config $(CONFIG)
	touch $(CHECKPOINT_DIR)/merge.done

# Split target - depends on merge checkpoint
split: $(CHECKPOINT_DIR)/split.done

$(CHECKPOINT_DIR)/split.done: $(CHECKPOINT_DIR)/merge.done | $(CHECKPOINT_DIR)
	$(PYTHON) scripts/split_dataset.py --config $(CONFIG)
	touch $(CHECKPOINT_DIR)/split.done

# Preprocess target - depends on split checkpoint
preprocess: $(CHECKPOINT_DIR)/preprocess.done

$(CHECKPOINT_DIR)/preprocess.done: $(CHECKPOINT_DIR)/split.done | $(CHECKPOINT_DIR)
	$(PYTHON) scripts/data_prep.py --config $(CONFIG)
	touch $(CHECKPOINT_DIR)/preprocess.done

# Data target - depends on preprocess checkpoint
data: $(CHECKPOINT_DIR)/preprocess.done

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

.PHONY: env train evaluate
