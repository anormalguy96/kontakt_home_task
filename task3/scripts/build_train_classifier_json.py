import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

from pii_guard.config import DATA


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+994|0)\s*\d{2}[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")  # loose card-like
FIN_RE = re.compile(r"\b[A-Z0-9]{7}\b")  # loose FIN-like


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def looks_like_pii(s: str) -> bool:
    return bool(EMAIL_RE.search(s) or PHONE_RE.search(s) or CARD_RE.search(s) or FIN_RE.search(s))


def iter_sentences(text: str, min_chars: int, max_chars: int) -> Iterable[str]:
    text = normalize_ws(text)
    if not text:
        return
    for sent in SENT_SPLIT_RE.split(text):
        sent = normalize_ws(sent)
        if min_chars <= len(sent) <= max_chars:
            yield sent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/train_classifier.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_unsafe", type=int, default=0, help="0 = use all unsafe rows")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--min_chars", type=int, default=25)
    ap.add_argument("--max_chars", type=int, default=280)
    ap.add_argument("--filter_safe_pii", action="store_true", help="Recommended: filter SAFE sentences with PII-like patterns")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unsafe = load_dataset(DATA.unsafe_dataset, split="train")
    unsafe_texts = [normalize_ws(x[DATA.unsafe_text_col]) for x in unsafe if normalize_ws(x[DATA.unsafe_text_col])]
    if args.max_unsafe and args.max_unsafe > 0:
        unsafe_texts = random.sample(unsafe_texts, k=min(args.max_unsafe, len(unsafe_texts)))

    n_unsafe = len(unsafe_texts)
    if n_unsafe == 0:
        raise RuntimeError("No UNSAFE samples found. Check dataset columns / access permissions.")

    safe_ds = load_dataset(DATA.safe_dataset, split="train")

    safe_sents = []
    for row in tqdm(safe_ds, desc="Collecting SAFE sentences"):
        for sent in iter_sentences(row.get(DATA.safe_text_col, ""), args.min_chars, args.max_chars):
            if args.filter_safe_pii and looks_like_pii(sent):
                continue
            safe_sents.append(sent)
            if len(safe_sents) >= n_unsafe:
                break
        if len(safe_sents) >= n_unsafe:
            break

    if len(safe_sents) < max(1000, int(0.6 * n_unsafe)):
        raise RuntimeError(
            f"Could not collect enough SAFE sentences. Got={len(safe_sents)} need~={n_unsafe}. "
            "Try disabling --filter_safe_pii or lowering --min_chars."
        )

    safe_sents = safe_sents[:n_unsafe]

    records = [{"text": t, "label": 1} for t in unsafe_texts] + [{"text": t, "label": 0} for t in safe_sents]
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val = records[:n_val]
    train = records[n_val:]

    with out_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps({"text": r["text"], "label": r["label"], "split": "train"}, ensure_ascii=False) + "\n")
        for r in val:
            f.write(json.dumps({"text": r["text"], "label": r["label"], "split": "validation"}, ensure_ascii=False) + "\n")

    meta = {
        "unsafe_dataset": DATA.unsafe_dataset,
        "safe_dataset": DATA.safe_dataset,
        "counts": {
            "train": len(train),
            "validation": len(val),
            "unsafe_total_used": n_unsafe,
            "safe_total_used": len(safe_sents),
        },
        "val_ratio": args.val_ratio,
        "filter_safe_pii": bool(args.filter_safe_pii),
        "seed": args.seed,
    }
    (out_path.parent / "train_classifier_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"train={len(train)} val={len(val)} (unsafe={n_unsafe} safe={len(safe_sents)})")


if __name__ == "__main__":
    main()
