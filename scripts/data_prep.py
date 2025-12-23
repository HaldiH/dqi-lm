import pandas as pd
import yaml
import os
import argparse
from datasets import Dataset


def load_prompt_content(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def process_split(
    df, template, system_prompt_content, col_speech, col_label, output_path
):
    formatted_data = []
    print(f"Formatting {len(df)} entries...")
    for _, row in df.iterrows():
        # Constructing the final text for the LLM
        full_text = template.format(
            system_prompt=system_prompt_content,
            input_text=row[col_speech],
            output_score=str(row[col_label]),
        )
        formatted_data.append({"text": full_text})

    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset.to_json(output_path)
    print(f"Success! Formatted dataset saved to: {output_path}")
    return formatted_data


def format_data(cfg):
    print(f"--- Preparing data ---")

    prompt_path = cfg["prompts"]["system_prompt_path"]
    print(f"Loading prompt from: {prompt_path}")
    system_prompt_content = load_prompt_content(prompt_path)

    template = cfg["prompts"]["format_template"]
    col_label = cfg["data"]["col_label"]
    col_speech = cfg["data"]["col_speech"]

    # Load speeches from merged_debates file
    merged_debates_path = cfg["data"]["merged_debates_path"]
    print(f"Reading speeches from: {merged_debates_path}")
    df_speeches = pd.read_csv(merged_debates_path)
    print(f"Loaded {len(df_speeches)} speeches")

    # 1. Process Train
    train_path = cfg["data"]["train_path"]
    print(f"Reading train dataset: {train_path}")
    df_train = pd.read_csv(train_path)

    # Merge train data with speeches on speech_id
    print("Merging train data with speeches...")
    df_train = df_train.merge(
        df_speeches[["speech_id", "speech"]],
        left_on="speech_id_1",
        right_on="speech_id",
        how="left",
    )
    print(f"Train dataset after merge: {len(df_train)} rows")

    processed_train_path = cfg["data"]["processed_train_path"]
    train_data = process_split(
        df_train,
        template,
        system_prompt_content,
        col_speech,
        col_label,
        processed_train_path,
    )
    print(f"Sample generated (Train):\n{train_data[0]['text'][:300]}...")

    # 2. Process Validation
    val_path = cfg["data"]["val_path"]
    print(f"Reading val dataset: {val_path}")
    df_val = pd.read_csv(val_path)

    # Merge val data with speeches on speech_id
    print("Merging val data with speeches...")
    df_val = df_val.merge(
        df_speeches[["speech_id", "speech"]],
        left_on="speech_id_1",
        right_on="speech_id",
        how="left",
    )
    print(f"Val dataset after merge: {len(df_val)} rows")

    processed_val_path = cfg["data"]["processed_val_path"]
    process_split(
        df_val,
        template,
        system_prompt_content,
        col_speech,
        col_label,
        processed_val_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    format_data(config)
