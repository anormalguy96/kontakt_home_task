from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import (
  Wav2Vec2ForCTC,
  Wav2Vec2Processor,
  TrainingArguments,
  Trainer,
)
import evaluate


@dataclass
class DataCollatorCTCWithPadding:
  processor: Wav2Vec2Processor
  padding: bool = True

  def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_features = [{"input_values": f["input_values"]} for f in features]
    label_features = [{"input_ids": f["labels"]} for f in features]

    batch = self.processor.pad(
      input_features,
      padding=self.padding,
      return_tensors="pt",
    )

    with self.processor.as_target_processor():
      labels_batch = self.processor.pad(
        label_features,
        padding=self.padding,
        return_tensors="pt",
      )

    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch


def find_text_column(cols: List[str]) -> str:
  candidates = ["sentence", "text", "transcript", "transcription"]
  for c in candidates:
    if c in cols:
      return c
  raise ValueError(f"Could not find text column. Available columns: {cols}")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--model_name", default="facebook/wav2vec2-large-xlsr-53")
  p.add_argument("--dataset_name", default="ysdede/khanacademy-turkish")
  p.add_argument("--train_percent", type=float, default=0.1)
  p.add_argument("--eval_percent", type=float, default=0.05)
  p.add_argument("--epochs", type=int, default=1)
  p.add_argument("--output_dir", default="models/checkpoint")
  args = p.parse_args()

  # Load dataset
  ds = load_dataset(args.dataset_name)
  if "train" not in ds:
    raise ValueError("Dataset has no 'train' split.")

  # pick a subset (engineering-centric: fast)
  train = ds["train"].shuffle(seed=42)
  n_train = max(50, int(len(train) * args.train_percent))
  n_eval = max(20, int(len(train) * args.eval_percent))
  train = train.select(range(n_train))
  eval_ds = ds["train"].shuffle(seed=123).select(range(n_eval))

  # Ensure audio column
  cols = train.column_names
  if "audio" not in cols:
    raise ValueError(f"No 'audio' column. Available columns: {cols}")

  text_col = find_text_column(cols)

  train = train.cast_column("audio", Audio(sampling_rate=16_000))
  eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

  processor = Wav2Vec2Processor.from_pretrained(args.model_name)
  model = Wav2Vec2ForCTC.from_pretrained(
    args.model_name,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
  )

  def prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=16_000)
    batch["input_values"] = inputs.input_values[0]
    with processor.as_target_processor():
      batch["labels"] = processor(batch[text_col]).input_ids
    return batch

  train_proc = train.map(prepare, remove_columns=train.column_names, num_proc=1)
  eval_proc = eval_ds.map(prepare, remove_columns=eval_ds.column_names, num_proc=1)

  wer_metric = evaluate.load("wer")

  def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    # replace -100 in labels as pad_token_id
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

  collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

  training_args = TrainingArguments(
    output_dir=args.output_dir,
    group_by_length=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    num_train_epochs=args.epochs,
    fp16=torch.cuda.is_available(),
    report_to=["tensorboard"],
    run_name="turkish_asr_task2",
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_proc,
    eval_dataset=eval_proc,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
  )

  trainer.train()

  # Save model + processor
  trainer.save_model(args.output_dir)
  processor.save_pretrained(args.output_dir)

  print(f"[train] done. checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
  main()