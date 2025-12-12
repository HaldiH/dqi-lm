# scripts/train.py
import yaml
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import wandb
import os

# 1. Load config
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 2. Setup WandB
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project=cfg['wandb']['project'], name=cfg['wandb']['run_name'])

# 3. Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = cfg['model']['name'],
    max_seq_length = cfg['model']['max_seq_length'],
    dtype = None,
    load_in_4bit = cfg['model']['load_in_4bit'],
)

model = FastLanguageModel.get_peft_model(
    model,
    r = cfg['lora']['r'],
    target_modules = cfg['lora']['target_modules'],
    lora_alpha = cfg['lora']['lora_alpha'],
    lora_dropout = cfg['lora']['lora_dropout'],
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = cfg['training']['seed'],
)

# 4. Load Dataset
dataset = load_dataset("json", data_files=cfg['data']['processed_path'], split="train")

# 5. Configure Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = cfg['model']['max_seq_length'],
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps = cfg['training']['gradient_accumulation_steps'],
        warmup_steps = cfg['training']['warmup_steps'],
        max_steps = cfg['training']['max_steps'],
        learning_rate = float(cfg['training']['learning_rate']),
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = cfg['training']['logging_steps'],
        optim = cfg['training']['optim'],
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = cfg['training']['seed'],
        output_dir = cfg['training']['output_dir'],
        report_to = "wandb",
    ),
)

# 6. Train
print("Starting training...")
trainer_stats = trainer.train()

# 7. Save
print("Saving model...")
# model.save_pretrained_gguf(cfg['training']['output_dir'] + "_gguf", tokenizer, quantization_method = "q4_k_m")
model.save_pretrained(cfg['training']['output_dir'])

wandb.finish()
print("Finished.")