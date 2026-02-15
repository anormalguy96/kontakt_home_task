import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer


def file_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def bench_pytorch_classifier(model_dir: str, text: str, n: int) -> float:
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=192)

    with torch.inference_mode():
        for _ in range(10):
            _ = model(**inputs)
        t0 = time.perf_counter()
        for _ in range(n):
            _ = model(**inputs)
        t1 = time.perf_counter()
    return (t1 - t0) * 1000 / n


def bench_onnx_classifier(onnx_path: str, tok_dir: str, text: str, n: int) -> float:
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    enc = tok(text, return_tensors="np", truncation=True, max_length=192)
    feed = {k: v for k, v in enc.items()}

    for _ in range(10):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, feed)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / n


def bench_pytorch_ner(model_dir: str, text: str, n: int) -> float:
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=192, return_offsets_mapping=False)

    with torch.inference_mode():
        for _ in range(10):
            _ = model(**inputs)
        t0 = time.perf_counter()
        for _ in range(n):
            _ = model(**inputs)
        t1 = time.perf_counter()
    return (t1 - t0) * 1000 / n


def bench_onnx_ner(onnx_path: str, tok_dir: str, text: str, n: int) -> float:
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    enc = tok(text, return_tensors="np", truncation=True, max_length=192)
    feed = {k: v for k, v in enc.items()}

    for _ in range(10):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, feed)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000 / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/benchmarks/bench.json")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_safe = "Salam, sabah görüşərik."
    sample_unsafe = "Mənim adım Elvin Aliyev, FİN 94FMDDD və telefon +994 50 123 45 67."

    c_pt = "models/classifier/pytorch"
    n_pt = "models/ner/pytorch"

    c_onnx = Path("models/classifier/onnx/model.onnx")
    c_q = Path("models/classifier/onnx/model.int8.onnx")

    n_onnx = Path("models/ner/onnx/model.onnx")
    n_q = Path("models/ner/onnx/model.int8.onnx")

    results = {
        "sizes_mb": {
            "classifier_onnx": file_mb(c_onnx) if c_onnx.exists() else None,
            "classifier_onnx_int8": file_mb(c_q) if c_q.exists() else None,
            "ner_onnx": file_mb(n_onnx) if n_onnx.exists() else None,
            "ner_onnx_int8": file_mb(n_q) if n_q.exists() else None,
        },
        "latency_ms_avg": {
            "classifier_pytorch_safe": bench_pytorch_classifier(c_pt, sample_safe, args.n),
            "classifier_pytorch_unsafe": bench_pytorch_classifier(c_pt, sample_unsafe, args.n),
            "classifier_onnx_safe": bench_onnx_classifier(str(c_onnx), "models/classifier/onnx", sample_safe, args.n) if c_onnx.exists() else None,
            "classifier_onnx_int8_safe": bench_onnx_classifier(str(c_q), "models/classifier/onnx", sample_safe, args.n) if c_q.exists() else None,

            "ner_pytorch": bench_pytorch_ner(n_pt, sample_unsafe, args.n),
            "ner_onnx": bench_onnx_ner(str(n_onnx), "models/ner/onnx", sample_unsafe, args.n) if n_onnx.exists() else None,
            "ner_onnx_int8": bench_onnx_ner(str(n_q), "models/ner/onnx", sample_unsafe, args.n) if n_q.exists() else None,
        },
    }

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote benchmark report: {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()