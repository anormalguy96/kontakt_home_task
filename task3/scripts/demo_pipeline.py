from pii_guard.inference.pipeline import PiiCascade

pipe = PiiCascade(
    clf_onnx="models/classifier/onnx/model.int8.onnx",
    ner_onnx="models/ner/onnx/model.int8.onnx",
    max_len=96,
    threshold=0.5,
)

examples = [
    "Salam, sabah görüşərik.",
    "Mənim adım Elvin Aliyev, FİN 94FMDDD və telefon +994 50 123 45 67.",
    "Kartım 4169 1234 5678 9012, adım Aysel Mammadova.",
]

for t in examples:
    r = pipe.run(t)
    print("\nTEXT:", t)
    print("UNSAFE:", r.is_unsafe, "score:", round(r.guardrail_score, 3))
    print("MASKED:", r.masked_text)
    if r.entities:
        print("ENTITIES:", r.entities)