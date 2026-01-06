import torch

from unsloth import FastLanguageModel
import yaml
import re
import argparse
import weave
from datasets import load_dataset
from typing import Any
from pydantic import PrivateAttr
import wandb
import weave
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn.metrics import confusion_matrix


def extract_score(generated_text):
    """
    Extracts the first digit found in the generated text that is a valid class label.
    """
    valid_digits = r"\d+"
    pattern = f"[{valid_digits}]"
    match = re.search(pattern, generated_text)
    if match:
        return int(match.group(0))
    else:
        return -1  # Parsing error


@weave.op()
def exact_match_scorer(expected: int, output: dict) -> dict:
    """Scorer that checks if the predicted label exactly matches the expected label."""
    return {"match": expected == output["predicted_label"]}


@weave.op()
def accuracy_score_op(expected: int, output: dict) -> dict:
    """Computes accuracy (1 if match, 0 otherwise)"""
    pred = output["predicted_label"]
    return {"accuracy": 1 if expected == pred else 0}


@weave.op()
def mse_score_op(expected: int, output: dict) -> dict:
    """Computes Mean Squared Error between expected and predicted"""
    pred = output["predicted_label"]
    return {"mse": (expected - pred) ** 2}


@weave.op()
def mae_score_op(expected: int, output: dict) -> dict:
    """Computes Mean Absolute Error between expected and predicted"""
    pred = output["predicted_label"]
    return {"mae": abs(expected - pred)}


class ConfusionMatrixScorer(weave.Scorer):
    class_names: list[str]

    @weave.op()
    def score(
        self,
        expected: int,
        output: dict,
    ):
        return {
            "y_pred": output["predicted_label"],
            "y_true": expected,
        }

    @weave.op()
    def summarize(self, score_rows):
        # Cette fonction reçoit la liste de TOUS les résultats de score()

        # 1. Extraction des listes complètes
        # Note : score_rows est une liste de dictionnaires (ceux retournés par score)
        y_pred = [row["y_pred"] for row in score_rows if row.get("y_pred") is not None]
        y_true = [row["y_true"] for row in score_rows if row.get("y_true") is not None]

        # 2. Calcul de la matrice (Sklearn)
        cm = confusion_matrix(y_true, y_pred)

        # 3. Création du graphique (Matplotlib/Seaborn)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Prédiction")
        plt.ylabel("Vérité")
        plt.title("Matrice de Confusion")

        # 4. Conversion en image pour Weave
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        image = Image.open(buf)

        # 5. Retourner l'image (elle s'affichera dans le tableau récapitulatif Weave)
        return {"confusion_plot": image}


class DQIModel(weave.Model):
    """
    Define an extra ChatModel class to store and version more parameters than just the model name.
    This enables fine-tuning on specific parameters.
    """

    chat_model: str
    cm_temperature: float
    cm_max_new_tokens: int
    cm_quantize: bool
    inference_batch_size: int
    dtype: Any
    device: str
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def model_post_init(self, __context):
        # unsloth version (enable native 2x faster inference)
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.chat_model,
            max_seq_length=self.cm_max_new_tokens,
            dtype=self.dtype,
            load_in_4bit=self.cm_quantize,
        )
        FastLanguageModel.for_inference(self._model)

    @weave.op()
    async def predict(self, query: list[str]) -> dict:
        # add_generation_prompt = true - Must add for generation
        input_ids = self._tokenizer.apply_chat_template(
            query,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        output_ids = self._model.generate(
            input_ids=input_ids,
            max_new_tokens=self.cm_max_new_tokens,
            use_cache=True,
            temperature=self.cm_temperature,
            min_p=0.1,
        )

        decoded_outputs = self._tokenizer.batch_decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        generated_text = "".join(decoded_outputs).strip()
        predicted_label = extract_score(generated_text)

        return {
            "predicted_label": predicted_label,
            "generated_text": generated_text,
        }


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    weave.init(cfg["wandb"]["project"])
    global all_preds, all_targets
    all_preds = []
    all_targets = []

    model = DQIModel(
        name=f"{cfg["wandb"]["run_name"]}",
        chat_model=cfg["training"]["output_dir"],
        cm_temperature=1.0,
        cm_max_new_tokens=cfg["model"]["max_seq_length"],
        cm_quantize=cfg["model"]["load_in_4bit"],
        inference_batch_size=cfg["model"]["max_seq_length"],
        dtype=None,
        device="auto",
    )
    # Load HuggingFace dataset and convert to list for Weave
    hf_dataset = load_dataset(
        "json", data_files=cfg["data"]["processed_test_path"], split="train"
    )
    # Convert to list of dicts for Weave Evaluation
    # Extract messages (without answer) and keep expected output separate
    dataset = [
        {
            "query": example["messages"][:-1],
            "expected": int(example["messages"][-1]["content"]),
        }
        for example in hf_dataset
    ]

    cm_scorer = ConfusionMatrixScorer(
        class_names=[str(i) for i in range(len(cfg["data"]["labels"]))]
    )

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[
            exact_match_scorer,
            accuracy_score_op,
            mse_score_op,
            mae_score_op,
            cm_scorer,
        ],
    )

    results = asyncio.run(evaluation.evaluate(model))

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        job_type="evaluation",
    )
    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_targets,
                preds=all_preds,
                class_names=[str(i) for i in range(len(cfg["data"]["labels"]))],
            )
        }
    )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args)
