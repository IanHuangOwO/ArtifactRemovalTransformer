# Standard library imports
import argparse
import json
import os

# Third-party imports
import numpy as np
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments

# Local application imports
from huggingface.hf_config import ArtifactRemovalTransformerConfig
from huggingface.hf_dataset import build_hf_datasets
from huggingface.hf_model import ArtifactRemovalTransformerForConditionalGeneration


def load_json_config(path: str) -> dict:
    """Loads a JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(eval_pred):
    """Computes MSE and MAE for evaluation."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Ensure logits and labels are numpy arrays
    if not isinstance(logits, np.ndarray):
        logits = logits.cpu().numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.cpu().numpy()

    mse = np.mean((logits - labels) ** 2)
    mae = np.mean(np.abs(logits - labels))
    
    return {"mse": mse, "mae": mae}


def main():
    parser = argparse.ArgumentParser(description="Train ArtifactRemovalTransformer with Hugging Face Trainer")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hf_checkpoints",
        help="Directory to save checkpoints and training outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # --- 1. Load Project Configuration ---
    config = load_json_config(args.config_path)
    train_config = config.get("train", {})
    model_config_dict = config.get("model", {})
    data_loader_config = train_config.get("data_loader", {})

    # --- 2. Build Hugging Face Datasets ---
    datasets: DatasetDict = build_hf_datasets(config=config, seed=args.seed)
    train_dataset = datasets["train"]
    eval_dataset = datasets["val"]

    # --- 3. Create Model Configuration ---
    # Merge model-specific and training-specific loss settings into one config
    model_config_dict["loss_zscore"] = train_config.get("loss", {}).get("zscore", True)
    model_config = ArtifactRemovalTransformerConfig(**model_config_dict)

    # --- 4. Instantiate the Model ---
    model = ArtifactRemovalTransformerForConditionalGeneration(config=model_config)

    # 5. Define Training Arguments
    # The Noam scheduler is custom and not directly supported by `TrainingArguments`.
    # We will use AdamW with a linear warmup, which is a standard and effective alternative.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=train_config.get("epochs", 60),
        per_device_train_batch_size=data_loader_config.get("batch_size", 32),
        per_device_eval_batch_size=data_loader_config.get("batch_size", 32),
        warmup_steps=train_config.get("scheduler", {}).get("warmup", 400),
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_num_workers=data_loader_config.get("num_workers", 4),
        seed=args.seed,
        fp16=True, # Use mixed-precision training if a GPU is available
    )

    # --- 6. Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 7. Train the Model ---
    print("Starting training with Hugging Face Trainer...")
    trainer.train()

    # --- 8. Save the final model ---
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    main()