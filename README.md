# dqi-lm

A fine-tuning framework for training language models on debate quality indicators (DQI) using US Congressional debate data.

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager (install separately)

### Installation

1. **Set up the Python environment:**
   ```bash
   make env
   ```
   This command uses `uv` to install all project dependencies and create a virtual environment.

## Data Setup

The project requires US Congressional debate data from the Stanford Congress Text dataset.

### 1. Download Data

Download the data from: https://data.stanford.edu/congress_text

Extract the archive and place:
- `hein-daily/` directory → `data/raw/hein-daily/`
- `USfinal-clean.csv` → `data/raw/USfinal-clean.csv`

**Note:** The `USfinal-clean.csv` file is not publicly available. If you need access, please contact the project author.

### 2. Directory Structure

After downloading, your `data/raw/` directory should look like:
```
data/
└── raw/
    ├── hein-daily/
    │   ├── 097_SpeakerMap.txt
    │   ├── 098_SpeakerMap.txt
    │   └── ... (more speaker maps)
    └── USfinal-clean.csv
```

## Usage

The project uses a Makefile-based workflow for data processing and model training. Commands can be chained to create a complete pipeline.

### Available Commands

| Command         | Description                           |
| --------------- | ------------------------------------- |
| `make env`      | Install Python environment with `uv`  |
| `make data`     | Merge, split, and preprocess all data |
| `make train`    | Fine-tune the language model          |
| `make evaluate` | Evaluate the fine-tuned model         |
| `make publish`  | Publish/export the model              |

### Complete Workflow

```bash
# 1. Set up environment
make env

# 2. Process data (merges, splits, and preprocesses)
make data

# 3. Fine-tune the model
make train

# 4. Evaluate the model
make evaluate

# 5. Publish the model
make publish
```

### Individual Data Processing Steps

The `make data` command handles the following steps automatically, but you can also run them individually:

```bash
# Merge datasets
make merge

# Split into train/val/test
make split

# Preprocess data
make preprocess
```

### Configuration

To use a different config file, specify it via the `CONFIG` variable:

```bash
make train CONFIG=configs/config_mistral-7b-instruct-v0.3.yaml
make evaluate CONFIG=configs/config_mistral-7b-instruct-v0.3.yaml
```

Default config: `configs/config.yaml`

## Configuration Files

Pre-configured models are available in the `configs/` directory:

- `config_llama3.1-8b-instruct.yaml` - Llama 3.1 8B
- `config_mistral-7b-instruct-v0.3.yaml` - Mistral 7B
- `config_deepseek-r1-distill-llama-70b-bnb-4bit.yaml` - DeepSeek Llama 70B (4-bit)
- `config_gemma3-12b-it.yaml` - Gemma 3 12B
- And more...

## Project Structure

```
.
├── data/                    # Data directory (raw and processed)
├── configs/                 # Model configuration files
├── scripts/                 # Python scripts for data/training/evaluation
├── outputs/                 # Fine-tuned model outputs
├── prompts/                 # Evaluation prompt templates
├── slurm/                   # SLURM job submission scripts
├── Makefile                 # Workflow automation
└── pyproject.toml          # Project dependencies
```

## Notes

- The project includes Jupyter notebooks (`finetuning.ipynb`, `graphs.ipynb`) for detailed analysis
- Fine-tuned models are saved to `outputs/`
- Experiment tracking is handled via Weights & Biases (wandb)

## Support

For questions about the `USfinal-clean.csv` dataset or other inquiries, please contact the project author.
