import argparse
from pathlib import Path
import inspect
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pii_guard.config import MODELS


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train_classifier.json")
    ap.add_argument("--model", default=MODELS.classifier_base)
    ap.add_argument("--out", default="models/classifier/pytorch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}. Run scripts/build_train_classifier_json.py first.")

    ds_all = load_dataset("json", data_files=str(data_path), split="train")
    train_ds = ds_all.filter(lambda x: x["split"] == "train")
    val_ds = ds_all.filter(lambda x: x["split"] == "validation")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ("label",)])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=[c for c in val_ds.column_names if c not in ("label",)])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16 = torch.cuda.is_available()

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    tr_kwargs = dict(
        output_dir=str(out_dir / "_runs"),
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        fp16=fp16,
        report_to="none",
    )
    tr_kwargs[eval_key] = "epoch"

    args_tr = TrainingArguments(**tr_kwargs)

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"✅ Saved classifier to: {out_dir}")


if __name__ == "__main__":
    main()