import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import argparse


def split_data(config_path):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_path = config["data"]["raw_path"]
    df = pd.read_csv(input_path)

    col_label = config["data"]["col_label"]

    # Clean data: drop rows with missing labels
    df = df.dropna(subset=[col_label])

    print(f"Total dataset : {len(df)} rows")

    # 1. Split Train (80%) vs Temp (20%)
    # stratify=df[col_label] ensures classes are balanced
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[col_label]
    )

    # 2. Split Temp into Validation (10%) and Test (10%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df[col_label]
    )

    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    # Save
    train_path = config["data"]["train_path"]
    val_path = config["data"]["val_path"]
    test_path = config["data"]["test_path"]

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Files saved to {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train, val, and test sets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    split_data(args.config)
