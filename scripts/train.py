import yaml
import torch
from typing import cast
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from datasets import load_dataset, Dataset
import weave
import wandb
import os
import argparse


def train(cfg):
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
        use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        random_state=cfg["training"]["seed"],
    )

    tokenizer = get_chat_template(tokenizer)

    def formatting_prompts_func(examples: dict):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    train_dataset = cast(
        Dataset,
        load_dataset(
            "json", data_files=cfg["data"]["processed_train_path"], split="train"
        ),
    )
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = cast(
        Dataset,
        load_dataset(
            "json", data_files=cfg["data"]["processed_val_path"], split="train"
        ),
    )
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_length=cfg["model"]["max_seq_length"],
            dataset_num_proc=2,
            packing=cfg["training"].get("packing", False),
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
            max_steps=cfg["training"]["max_steps"],
            learning_rate=float(cfg["training"]["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=cfg["training"]["logging_steps"],
            optim=cfg["training"]["optim"],
            weight_decay=cfg["training"].get("weight_decay", 0.01),
            eval_strategy="steps",
            eval_steps=cfg["training"].get("eval_steps", 50),
            save_strategy="steps",
            save_steps=cfg["training"].get("save_steps", 100),
            save_total_limit=cfg["training"].get("save_total_limit", 3),
            save_safetensors=cfg["training"].get("save_safetensors", True),
            load_best_model_at_end=cfg["training"].get("load_best_model_at_end", True),
            metric_for_best_model=cfg["training"].get(
                "metric_for_best_model", "eval_loss"
            ),
            greater_is_better=cfg["training"].get("greater_is_better", False),
            lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "linear"),
            max_grad_norm=cfg["training"].get("max_grad_norm", 0.3),
            neftune_noise_alpha=cfg["training"].get("neftune_noise_alpha"),
            seed=cfg["training"]["seed"],
            output_dir=cfg["training"]["output_dir"],
            report_to="wandb",
            tf32=cfg["training"].get("tf32", True),
            dataloader_num_workers=cfg["training"].get("dataloader_num_workers", 4),
        ),
    )

    print("Starting training...")
    resume_from_checkpoint = cfg["training"].get("resume_from_checkpoint", None)
    if resume_from_checkpoint is True:
        resume_from_checkpoint = "latest"
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("Saving model...")
    output_dir = cfg["training"]["output_dir"]
    # model.save_pretrained_gguf(cfg['training']['output_dir'] + "_gguf", tokenizer, quantization_method = "q4_k_m")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    trainer.save_state()
    return model, tokenizer


def main(config):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    run = wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"])
    train(cfg)
    run.finish()

    # with wandb.init(
    #     project=cfg["wandb"]["project"], id=run.id, resume="allow"
    # ) as run:
    #     artifact = wandb.Artifact(
    #         name=cfg["wandb"].get("artifact_name", "finetuned-model"),
    #         type=cfg["wandb"].get("artifact_type", "model"),
    #         description=cfg["wandb"].get(
    #             "artifact_description", "Finetuned language model"
    #         ),
    #         metadata=cfg["wandb"].get("artifact_metadata", {}),
    #     )
    #     artifact.add_dir(cfg["training"]["output_dir"])
    #     run.log_artifact(artifact)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(**vars(args))
