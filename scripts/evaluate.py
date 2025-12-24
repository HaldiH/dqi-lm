import torch
from unsloth import FastLanguageModel
import pandas as pd
from tqdm import tqdm
import yaml
import re
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def extract_score(generated_text, num_classes):
    """
    Extracts the first digit found in the generated text that is a valid class label.
    """
    # Create pattern for valid class indices (0 to num_classes-1)
    valid_digits = "|" .join(str(i) for i in range(num_classes))
    pattern = f"[{valid_digits}]"
    match = re.search(pattern, generated_text)
    if match:
        return int(match.group(0))
    else:
        return -1  # Parsing error


def evaluate(config_path):
    # --- CONFIGURATION ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load system prompt
    with open(cfg["prompts"]["system_prompt_path"], "r") as f:
        system_prompt = f.read().strip()

    TEMPLATE = cfg["prompts"]["format_template"]

    # 1. Load fine-tuned model
    model_path = cfg["training"]["output_dir"]

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode (faster)

    # 2. Load Test Set and Speeches
    merged_debates_path = cfg["data"]["merged_debates_path"]
    print(f"Reading speeches from: {merged_debates_path}")
    df_speeches = pd.read_csv(merged_debates_path)
    print(f"Loaded {len(df_speeches)} speeches")

    test_path = cfg["data"]["test_path"]
    print(f"Reading test dataset: {test_path}")
    df_test = pd.read_csv(test_path)

    # Merge test data with speeches on speech_id
    print("Merging test data with speeches...")
    df_test = df_test.merge(
        df_speeches[["speech_id", "speech"]],
        left_on="speech_id_1",
        right_on="speech_id",
        how="left",
    )
    print(f"Test dataset after merge: {len(df_test)} rows")

    col_label = cfg["data"]["col_label"]
    num_classes = len(cfg["data"]["labels"])
    results_dir = cfg["evaluation"]["results_dir"]
    print(f"Number of classes: {num_classes}")
    print(f"Results will be saved to: {results_dir}")
    
    y_true = df_test[col_label].tolist()
    y_pred = []

    print("Starting inference on Test Set...")

    # 3. Prediction loop
    for text in tqdm(df_test["speech"]):
        # Prepare prompt
        prompt = TEMPLATE.format(
            system_prompt=system_prompt,
            input_text=text,
            output_score="",  # Leave empty for model completion
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generation
        outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True)

        # Decode only the new tokens
        new_tokens = outputs[0, inputs.input_ids.shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred = extract_score(decoded, num_classes)
        y_pred.append(pred)

    # Save predictions to CSV
    results_df = pd.DataFrame(
        {"speech": df_test["speech"], "true_label": y_true, "predicted_label": y_pred}
    )
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(f"{results_dir}/predictions.csv", index=False)
    print(f"\nPredictions saved to '{results_dir}/predictions.csv'")

    # 4. Calculate metrics
    # Filter parsing errors (-1)
    valid_indices = [i for i, x in enumerate(y_pred) if x != -1]
    y_true_clean = [y_true[i] for i in valid_indices]
    y_pred_clean = [y_pred[i] for i in valid_indices]

    parsing_errors = len(y_pred) - len(valid_indices)
    print(f"\n--- RESULTS ---")
    print(f"LLM formatting errors : {parsing_errors}/{len(y_pred)}")
    print(f"Accuracy : {accuracy_score(y_true_clean, y_pred_clean):.4f}")
    print(
        f"Mean Absolute Error : {mean_absolute_error(y_true_clean, y_pred_clean):.4f}"
    )

    # Generate target names dynamically
    target_names = [f"Level {i}" for i in range(num_classes)]
    
    print("\nClassification Report :")
    print(
        classification_report(
            y_true_clean,
            y_pred_clean,
            target_names=target_names,
        )
    )

    # 5. Visualization : Confusion Matrix
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)

    # Generate dynamic tick labels
    pred_labels = [f"Pred {i}" for i in range(num_classes)]
    true_labels = [f"True {i}" for i in range(num_classes)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=pred_labels,
        yticklabels=true_labels,
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix - DQI Justification")

    # Save graph
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    print(f"\nGraph saved to '{results_dir}/confusion_matrix.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    evaluate(args.config)
