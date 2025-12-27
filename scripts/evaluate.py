import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
from tqdm import tqdm
import yaml
import re
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import weave
import wandb


def extract_score(generated_text, num_classes):
    """
    Extracts the first digit found in the generated text that is a valid class label.
    """
    # Create pattern for valid class indices (0 to num_classes-1)
    valid_digits = "|".join(str(i) for i in range(num_classes))
    pattern = f"[{valid_digits}]"
    match = re.search(pattern, generated_text)
    if match:
        return int(match.group(0))
    else:
        return -1  # Parsing error


PREDICT_CONTEXT = {}


@weave.op()
def predict_text(text: str):
    ctx = PREDICT_CONTEXT
    model = ctx["model"]
    text_tokenizer = ctx["text_tokenizer"]
    chat_tokenizer = ctx["chat_tokenizer"]
    system_prompt = ctx["system_prompt"]
    num_classes = ctx["num_classes"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text to analyse: '{text}'"},
    ]

    prompt = chat_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = text_tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True)
    new_tokens = outputs[0, inputs.input_ids.shape[1] :]
    decoded = text_tokenizer.decode(new_tokens, skip_special_tokens=True)
    pred = extract_score(decoded, num_classes)

    return {"prediction": pred, "decoded": decoded}


def evaluate(config_path):
    # --- CONFIGURATION ---
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_run = wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{cfg['wandb']['run_name']}-eval",
        job_type="evaluation",
    )
    weave.init(cfg["wandb"]["project"])

    # Load system prompt
    with open(cfg["prompts"]["system_prompt_path"], "r") as f:
        system_prompt = f.read().strip()

    # 1. Load fine-tuned model via W&B artifact to keep eval reproducible
    artifact_name = cfg["wandb"].get(
        "model_artifact", f"{cfg['wandb']['run_name']}-model:latest"
    )
    model_artifact = wandb_run.use_artifact(artifact_name, type="model")
    model_path = model_artifact.download()

    print(f"Loading model from {model_path}...")
    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Enable inference mode (faster)
    # Some models (e.g., Pixtral/vision-capable variants) return a Processor
    # which expects image inputs. For text-only inference, unwrap to the
    # underlying text tokenizer when available.
    text_tokenizer = (
        tokenizer_or_processor.tokenizer
        if hasattr(tokenizer_or_processor, "tokenizer")
        else tokenizer_or_processor
    )
    chat_tokenizer = get_chat_template(text_tokenizer)

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

    y_true = df_test[col_label].astype(int).tolist()
    y_pred = []

    PREDICT_CONTEXT.update(
        {
            "model": model,
            "text_tokenizer": text_tokenizer,
            "chat_tokenizer": chat_tokenizer,
            "system_prompt": system_prompt,
            "num_classes": num_classes,
        }
    )

    print("Starting inference on Test Set...")

    # 3. Prediction loop
    for text in tqdm(df_test["speech"]):
        prediction = predict_text(text)
        y_pred.append(prediction["prediction"])

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
    accuracy = accuracy_score(y_true_clean, y_pred_clean)
    print(f"Accuracy : {accuracy:.4f}")
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    print(f"Mean Absolute Error : {mae:.4f}")
    print(f"Mean Squared Error : {mse:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, average="macro", zero_division=0
    )
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

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

    metrics_payload = {
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "parsing_errors": parsing_errors,
        "evaluated_samples": len(y_true_clean),
    }
    wandb_run.log(metrics_payload)

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
