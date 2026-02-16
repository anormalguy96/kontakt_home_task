import argparse, json, re
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def tokenize_with_offsets(text: str):
    tokens, spans = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans

def entities_to_bio(tokens, spans, entities):
    tags = ["O"] * len(tokens)

    # sort by start to keep BIO stable
    entities = sorted(entities, key=lambda e: (e["start"], e["end"]))

    for ent in entities:
        label = ent["label"].upper()
        start, end = int(ent["start"]), int(ent["end"])

        idxs = []
        for i, (s, e) in enumerate(spans):
            # overlap between token span and entity span
            if not (e <= start or s >= end):
                idxs.append(i)

        if not idxs:
            continue

        # BIO tagging
        tags[idxs[0]] = f"B-{label}"
        for i in idxs[1:]:
            tags[i] = f"I-{label}"

    return tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="ner_train_data.json")
    ap.add_argument("--out", default="data/synthetic/ner_bio.jsonl")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(inp.read_text(encoding="utf-8"))

    n = 0
    with outp.open("w", encoding="utf-8") as f:
        for row in data:
            text = row["text"]
            entities = row.get("entities", [])

            tokens, spans = tokenize_with_offsets(text)
            tags = entities_to_bio(tokens, spans, entities)

            f.write(json.dumps({"tokens": tokens, "tags": tags, "text": text}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Converted -> {outp} (n={n})")

if __name__ == "__main__":
    main()