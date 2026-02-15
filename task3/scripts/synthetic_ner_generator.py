import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


FIN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"

FIRST_NAMES = ["Elvin", "Aysel", "Nigar", "Orxan", "Leyla", "Kamal", "Zehra", "Murad", "Gunel", "Ramil"]
LAST_NAMES = ["Aliyev", "Mammadov", "Huseynov", "Hasanli", "Quliyev", "Abdullayeva", "Ismayilov", "Mustafayev"]

TEMPLATES = [
    "Salam, mənim adım {PERSON} və FİN kodum {FIN}-dir.",
    "Zəhmət olmasa {PERSON} üçün nömrəmi yeniləyin: {PHONE}.",
    "Kart məlumatım budur: {CARD}. Ad: {PERSON}.",
    "Müştəri: {PERSON}, əlaqə: {PHONE}, FİN: {FIN}.",
    "Ödəniş üçün kart: {CARD}, telefon: {PHONE}.",
    "Mən {PERSON}-am, FİN {FIN}, kart {CARD}.",
]

def gen_fin(rng: random.Random) -> str:
    return "".join(rng.choice(FIN_ALPHABET) for _ in range(7))

def gen_phone(rng: random.Random) -> str:
    if rng.random() < 0.5:
        # +994 XX XXX XX XX
        op = rng.choice(["50", "51", "55", "70", "77", "99"])
        return f"+994 {op} {rng.randint(100,999)} {rng.randint(10,99)} {rng.randint(10,99)}"
    else:
        # 0XX-XXX-XX-XX
        op = rng.choice(["050", "051", "055", "070", "077", "099"])
        return f"{op}-{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(10,99)}"

def gen_card(rng: random.Random) -> str:
    parts = [str(rng.randint(1000, 9999)) for _ in range(4)]
    sep = rng.choice([" ", "-"])
    return sep.join(parts)

def gen_person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

def bio_tag_tokens(tokens: List[str], entity_tokens: List[str], entity_label: str) -> List[str]:
    tags = ["O"] * len(tokens)
    for i in range(0, len(tokens) - len(entity_tokens) + 1):
        if tokens[i : i + len(entity_tokens)] == entity_tokens:
            tags[i] = f"B-{entity_label}"
            for j in range(1, len(entity_tokens)):
                tags[i + j] = f"I-{entity_label}"
            return tags
    return tags

def merge_tags(base: List[str], add: List[str]) -> List[str]:
    out = base[:]
    for i, t in enumerate(add):
        if t != "O":
            out[i] = t
    return out

def tokenize_simple(text: str) -> List[str]:
    return text.split()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/ner_bio.jsonl")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            person = gen_person(rng)
            fin = gen_fin(rng)
            phone = gen_phone(rng)
            card = gen_card(rng)

            text = rng.choice(TEMPLATES).format(PERSON=person, FIN=fin, PHONE=phone, CARD=card)
            tokens = tokenize_simple(text)

            tags = ["O"] * len(tokens)
            tags = merge_tags(tags, bio_tag_tokens(tokens, tokenize_simple(person), "PERSON"))
            tags = merge_tags(tags, bio_tag_tokens(tokens, tokenize_simple(fin), "FIN"))
            tags = merge_tags(tags, bio_tag_tokens(tokens, tokenize_simple(phone), "PHONE"))
            tags = merge_tags(tags, bio_tag_tokens(tokens, tokenize_simple(card), "CARD"))

            f.write(json.dumps({"tokens": tokens, "tags": tags, "text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote synthetic NER data: {out_path} (n={args.n})")


if __name__ == "__main__":
    main()
