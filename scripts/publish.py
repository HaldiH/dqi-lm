"""
Model publishing utilities for uploading trained models to various platforms.
"""

import os
import yaml
import argparse
import wandb
from unsloth import FastLanguageModel


def publish_model(model, tokenizer, cfg, run: wandb.Run):
    """Publish the trained model to various platforms."""
    publish_config = cfg.get("publish", {})

    if not publish_config.get("enabled", False):
        print("Model publishing is disabled in config.")
        return

    output_dir = cfg["training"]["output_dir"]

    # Publish to HuggingFace Hub
    if publish_config.get("huggingface", {}).get("enabled", False):
        hf_config = publish_config["huggingface"]
        repo_id = hf_config.get("repo_id")

        if not repo_id:
            print("Error: HuggingFace repo_id not specified in config.")
        else:
            print(f"Publishing model to HuggingFace Hub: {repo_id}")
            try:
                # Check if HF token is set
                hf_token = os.getenv("HF_TOKEN") or hf_config.get("token")

                model.push_to_hub(
                    repo_id=repo_id,
                    token=hf_token,
                    private=hf_config.get("private", False),
                    commit_message=hf_config.get(
                        "commit_message", "Upload finetuned model"
                    ),
                )
                tokenizer.push_to_hub(
                    repo_id=repo_id,
                    token=hf_token,
                    private=hf_config.get("private", False),
                    commit_message=hf_config.get(
                        "commit_message", "Upload finetuned tokenizer"
                    ),
                )
                model.push_to_hub_merged(
                    repo_id=repo_id,
                    token=hf_token,
                    private=hf_config.get("private", False),
                    save_method="merged_16bit",
                    commit_message=hf_config.get(
                        "commit_message", "Upload finetuned model (merged)"
                    ),
                )
                print(f"✓ Successfully published to HuggingFace Hub: {repo_id}")
            except Exception as e:
                print(f"Error publishing to HuggingFace Hub: {e}")

    # Publish to WandB as artifact
    if publish_config.get("wandb", {}).get("enabled", False):
        wandb_config = publish_config["wandb"]
        artifact_name = wandb_config.get("artifact_name", "finetuned-model")
        artifact_type = wandb_config.get("artifact_type", "model")

        print(f"Publishing model to WandB as artifact: {artifact_name}")
        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=wandb_config.get("description", "Finetuned language model"),
                metadata=wandb_config.get("metadata", {}),
            )
            artifact.add_dir(output_dir)
            run.log_artifact(artifact)
            print(f"✓ Successfully published to WandB: {artifact_name}")
        except Exception as e:
            print(f"Error publishing to WandB: {e}")


def main(config, model_dir):
    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Determine model directory
    model_dir = model_dir or cfg["training"]["output_dir"]

    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return

    print(f"Loading model from: {model_dir}")

    # Initialize WandB if publishing to WandB
    if cfg.get("publish", {}).get("wandb", {}).get("enabled", False):
        run = wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["run_name"] + "-publish",
            job_type="model-publish",
        )

    # Load model and tokenizer
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=cfg["model"]["max_seq_length"],
            dtype=None,
            load_in_4bit=cfg["model"].get("load_in_4bit", False),
        )
        print("✓ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Publish the model
    publish_model(model, tokenizer, cfg, run)

    if cfg.get("publish", {}).get("wandb", {}).get("enabled", False):
        run.finish()

    print("Publishing complete.")


if __name__ == "__main__":
    """CLI for publishing a trained model."""
    import dotenv

    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Publish a trained model to HuggingFace Hub and/or WandB."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file used for training.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the trained model directory. If not provided, uses output_dir from config.",
    )
    args = parser.parse_args()
    main(**vars(args))
