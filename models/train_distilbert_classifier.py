"""
SafeMedTwin — DistilBERT Risk Classifier Training
=====================================================
Fine-tunes distilbert-base-uncased for 3-class text classification
(low / moderate / high risk) on the text dataset built by
models/text_dataset.py. This is DistilBERT doing what it's actually
built for — encoding text and classifying it — as the complementary
counterpart to TinyLlama's generative role in the SLM comparison.

Requires: pip install -r requirements-slm.txt (transformers, torch) —
NOT run automatically as part of the main app setup, since it's a
genuinely separate ~268MB model download plus a training step (a few
minutes on CPU for this dataset size, faster with a GPU).

Run:
    python models/train_distilbert_classifier.py

Saves to: models/distilbert_risk_classifier/
"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.text_dataset import build_all_text_datasets

MODEL_DIR = os.path.join(os.path.dirname(__file__), "distilbert_risk_classifier")
BASE_MODEL = "distilbert-base-uncased"
LABELS = ["low", "moderate", "high"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def main():
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            Trainer, TrainingArguments, DataCollatorWithPadding,
        )
    except ImportError:
        print("Missing dependencies. Run: pip install -r requirements-slm.txt")
        sys.exit(1)

    print("Building text dataset from all 5 diseases...")
    df = build_all_text_datasets()
    df["label_id"] = df["label"].map(LABEL2ID)

    # 90/10 split, stratified by label so the small "moderate" class isn't
    # starved in either split
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label_id"]
    )
    print(f"Train: {len(train_df)}  Eval: {len(eval_df)}")
    print("Label distribution (train):", train_df["label"].value_counts().to_dict())

    print(f"\nLoading tokenizer and base model ({BASE_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    class RiskTextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(list(texts), truncation=True, max_length=128)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_dataset = RiskTextDataset(train_df["text"], train_df["label_id"])
    eval_dataset = RiskTextDataset(eval_df["text"], eval_df["label_id"])

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=int(os.environ.get("SAFEMEDTWIN_DISTILBERT_EPOCHS", "5")),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to=[],  # no wandb/tensorboard — keep this self-contained
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    final_path = os.path.join(MODEL_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved fine-tuned classifier to {final_path}")


if __name__ == "__main__":
    main()
