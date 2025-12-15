import yaml
import torch
from typing import cast
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
import weave
import wandb
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def train(cfg):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )

    train_dataset = cast(
        Dataset,
        load_dataset(
            "json", data_files=cfg["data"]["processed_train_path"], split="train"
        ),
    )
    eval_dataset = cast(
        Dataset,
        load_dataset(
            "json", data_files=cfg["data"]["processed_val_path"], split="train"
        ),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=cfg["model"]["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            warmup_steps=cfg["training"]["warmup_steps"],
            max_steps=cfg["training"]["max_steps"],
            learning_rate=float(cfg["training"]["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=cfg["training"]["logging_steps"],
            optim=cfg["training"]["optim"],
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            load_best_model_at_end=True,
            lr_scheduler_type="linear",
            seed=cfg["training"]["seed"],
            output_dir=cfg["training"]["output_dir"],
            report_to="wandb",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    print("Saving model...")
    # model.save_pretrained_gguf(cfg['training']['output_dir'] + "_gguf", tokenizer, quantization_method = "q4_k_m")
    model.save_pretrained(cfg["training"]["output_dir"])

    wandb.finish()
    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
