import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


LABELS = [
    "O",
    "B-FIN", "I-FIN",
    "B-CARD", "I-CARD",
    "B-PHONE", "I-PHONE",
    "B-PERSON", "I-PERSON",
]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

# simple post-mask cleanup: avoid "****" glued to punctuation weirdly
MULTI_STAR_RE = re.compile(r"\*{4,}")


@dataclass
class CascadeResult:
    is_unsafe: bool
    masked_text: str
    guardrail_score: float
    entities: List[Tuple[str, str]]  # (entity_label, entity_text)


class PiiCascade:
    """
    Efficient cascade:
      1) ONNX INT8 classifier
      2) if unsafe -> ONNX INT8 token classifier + mask spans
    """

    def __init__(
        self,
        clf_dir: str = "models/classifier/onnx",
        ner_dir: str = "models/ner/onnx",
        clf_onnx: str = "models/classifier/onnx/model.int8.onnx",
        ner_onnx: str = "models/ner/onnx/model.int8.onnx",
        max_len: int = 96,  # keep small for speed
        threshold: float = 0.5,
    ):
        self.max_len = max_len
        self.threshold = threshold

        self.clf_tok = AutoTokenizer.from_pretrained(clf_dir, use_fast=True)
        self.ner_tok = AutoTokenizer.from_pretrained(ner_dir, use_fast=True)

        self.clf_sess = ort.InferenceSession(clf_onnx, providers=["CPUExecutionProvider"])
        self.ner_sess = ort.InferenceSession(ner_onnx, providers=["CPUExecutionProvider"])

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _clf_predict_proba_unsafe(self, text: str) -> float:
        enc = self.clf_tok(text, return_tensors="np", truncation=True, max_length=self.max_len)
        logits = self.clf_sess.run(None, dict(enc))[0]  # (1,2)
        # logits[0,1] => unsafe logit vs safe logit; use softmax
        exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exps / exps.sum(axis=-1, keepdims=True)
        return float(probs[0, 1])

    def _ner_predict(self, text: str) -> Dict:
        enc = self.ner_tok(
            text,
            return_tensors="np",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()  # list of [start,end]
        logits = self.ner_sess.run(None, dict(enc))[0][0]  # (seq, num_labels)
        pred_ids = logits.argmax(axis=-1).tolist()
        tokens = self.ner_tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        return {"tokens": tokens, "pred_ids": pred_ids, "offsets": offsets}

    def _extract_spans(self, text: str, pred) -> List[Tuple[str, int, int]]:
        spans = []
        current = None  # (label, start, end)
        for pid, (s, e) in zip(pred["pred_ids"], pred["offsets"]):
            if s == 0 and e == 0:
                continue  # special tokens
            label = ID2LABEL.get(pid, "O")
            if label == "O":
                if current:
                    spans.append(current)
                    current = None
                continue

            if label.startswith("B-"):
                if current:
                    spans.append(current)
                current = (label[2:], s, e)
            elif label.startswith("I-"):
                ent = label[2:]
                if current and current[0] == ent:
                    current = (current[0], current[1], e)
                else:
                    # broken I- without B-: start new
                    if current:
                        spans.append(current)
                    current = (ent, s, e)

        if current:
            spans.append(current)

        # merge overlapping / adjacent spans
        spans.sort(key=lambda x: (x[1], x[2]))
        merged = []
        for ent, s, e in spans:
            if not merged:
                merged.append([ent, s, e])
            else:
                last = merged[-1]
                if s <= last[2] + 1:  # overlap/adjacent
                    last[2] = max(last[2], e)
                    # keep entity label generic if conflict
                    if last[0] != ent:
                        last[0] = "PII"
                else:
                    merged.append([ent, s, e])
        return [(m[0], m[1], m[2]) for m in merged]

    def _mask_text(self, text: str, spans: List[Tuple[str, int, int]]) -> Tuple[str, List[Tuple[str, str]]]:
        if not spans:
            return text, []
        out = []
        last = 0
        extracted = []
        for ent, s, e in spans:
            out.append(text[last:s])
            out.append("****")
            extracted.append((ent, text[s:e]))
            last = e
        out.append(text[last:])
        masked = "".join(out)
        masked = MULTI_STAR_RE.sub("****", masked)
        return masked, extracted

    def run(self, text: str) -> CascadeResult:
        p_unsafe = self._clf_predict_proba_unsafe(text)
        is_unsafe = p_unsafe >= self.threshold
        if not is_unsafe:
            return CascadeResult(is_unsafe=False, masked_text=text, guardrail_score=p_unsafe, entities=[])

        pred = self._ner_predict(text)
        spans = self._extract_spans(text, pred)
        masked, entities = self._mask_text(text, spans)
        return CascadeResult(is_unsafe=True, masked_text=masked, guardrail_score=p_unsafe, entities=entities)