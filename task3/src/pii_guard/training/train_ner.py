import argparse
from pathlib import Path
import inspect
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from pii_guard.config import MODELS


LABELS = [
    "O",
    "B-FIN", "I-FIN",
    "B-CARD", "I-CARD",
    "B-PHONE", "I-PHONE",
    "B-PERSON", "I-PERSON",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def align_labels_with_tokens(tokenizer, tokens, tags):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, return_offsets_mapping=True)
    word_ids = enc.word_ids()

    labels = []
    prev_word = None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        else:
            tag = tags[wid]
            if wid != prev_word:
                labels.append(LABEL2ID[tag])
            else:
                if tag.startswith("B-"):
                    labels.append(LABEL2ID["I-" + tag[2:]])
                else:
                    labels.append(LABEL2ID[tag])
        prev_word = wid
    enc.pop("offset_mapping", None)
    enc["labels"] = labels
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic/ner_bio.jsonl")
    ap.add_argument("--model", default=MODELS.ner_base)
    ap.add_argument("--out", default="models/ner/pytorch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=24)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing synthetic NER data: {data_path}. Run scripts/synthetic_ner_generator.py first.")

    ds = load_dataset("json", data_files=str(data_path), split="train")
    ds = ds.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, val_ds = ds["train"], ds["test"]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    train_ds = train_ds.map(lambda x: align_labels_with_tokens(tok, x["tokens"], x["tags"]), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda x: align_labels_with_tokens(tok, x["tokens"], x["tags"]), remove_columns=val_ds.column_names)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []
        for pred_row, lab_row in zip(preds, labels):
            row_labels = []
            row_preds = []
            for pr, lb in zip(pred_row, lab_row):
                if lb == -100:
                    continue
                row_labels.append(ID2LABEL[int(lb)])
                row_preds.append(ID2LABEL[int(pr)])
            true_labels.append(row_labels)
            true_preds.append(row_preds)

        m = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": m["overall_precision"],
            "recall": m["overall_recall"],
            "f1": m["overall_f1"],
            "accuracy": m["overall_accuracy"],
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16 = torch.cuda.is_available()

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    tr_kwargs = dict(
        output_dir=str(out_dir / "_runs"),
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        seed=args.seed,
        fp16=fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    tr_kwargs[eval_key] = "epoch"

    tr_args = TrainingArguments(**tr_kwargs)

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Eval:", trainer.evaluate())

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved NER model to: {out_dir}")


if __name__ == "__main__":
    main()