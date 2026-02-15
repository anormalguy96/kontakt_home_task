# Task 3 — Hybrid PII Protection System (Guardrail + Extractor)

Bu repozitoriya müştərilərin çatbotla yazışarkən paylaşa bildiyi həssas məlumatları (PII) avtomatik aşkar edib maskalamaq üçün qurulmuş **iki mərhələli (cascading)** bir NLP pipeline-dır.

Məqsəd çox sadədir:
- Mesaj təhlükəsizdirsə → gecikmədən buraxılsın (latency < 20ms)
- Mesajda PII varsa → PII hissələr tapılıb `****` ilə maskalansın

Bu işdə əsas fokus mükəmməl modeldən çox **tam işlək pipeline + optimizasiya + ölçüm** üzərindədir. Açığı, əlçatan GPU imkanım olmadığı və CPU mühitində işlədiyim üçün

![alt text](https://i.pinimg.com/236x/bf/fd/0e/bffd0e24f35c7e61f0b8bc022b94a213.jpg)

training hissəsində “ən yaxşı nəticə” üçün günlərlə gözləmək əvəzinə pipeline-ı real şəkildə işlək edib sonra da ONNX + INT8 quantization ilə sürəti/ölçünü ciddi şəkildə yaxşılaşdırmağa fokuslandım. Bu yanaşma həm praktikdir, həm də tapşırığın ruhuna daha uyğundur.

---

## Arxitektura (2 mərhələ)

### 1) The Guardrail (Binary Classifier)
Birinci mərhələdə məqsəd mesajın “SAFE/UNSAFE” olduğunu sürətli şəkildə müəyyən etməkdir.

- **UNSAFE** (`Label=1`) - `LocalDoc/pii_ner_azerbaijani` dataseti. Bu datasetdəki cümlələri olduğu kimi götürüb `label=1` verilir.
- **SAFE** (`Label=0`) - `aznlp/azerbaijani-blogs` dataseti. Blog mətnləri cümlələrə bölünür və UNSAFE sayına yaxın miqdarda nümunə götürülür (balans üçün).

Çıxış:
- SAFE → dərhal return (Extractor işə düşmür)
- UNSAFE → Extractor mərhələsinə ötürülür

### 2) The Extractor (NER / TokenClassification)
Guardrail “UNSAFE” dedikdə, ikinci model işə düşür və PII hissələri BIO formatında tapır:

- `PERSON`
- `FIN`
- `PHONE`
- `CARD`

Bu mərhələdə real data limitləri və format spesifikliyi səbəbilə sintetik data istifadə olunur:
- `scripts/synthetic_ner_generator.py` minlərlə BIO etiketli nümunə yaradır

#### Not: Sözügedən `scripts/synthetic_ner_generator.py` skriptini manual olaraq deklarasiya edib yazmağımın səbəbi tapşırıq sənədində bu skriptin olmamasıdır.

---

## Niyə “full dataset + uzun training” etmədim?

Tapşırıqda “pipeline və düşüncə tərzi” önə çıxır — mən də eyni yanaşmanı seçdim.

- CPU mühitində 120k+ nümunə ilə transformer fine-tune etmək saatlarla (bəzən günlərlə) vaxt aparır.
- Praktik olaraq bu, məhsuldar deyil: sizə lazım olan əsas şey işlək arxitektura + optimizasiya + benchmark nəticələridir.
- Ona görə Guardrail üçün:
  - **az sayda nümunə** (8k unsafe + 8k safe, lakin sonra bu sayı da benchmark hissəsində 5k-ya qədər azaltmalı oldum)
  - **yüksək batch** (ilk öncə 32, sonra 96, sonda isə worst-case scenario ilə rastlaşdığım üçün **128**)
  - **qısa max_len**
  istifadə etdim ki, iterasiya sürətli olsun və pipeline tam bağlansın.

Bu arada, skriptlər “full dataset”lə işləməyi də dəstəkləyir — istəsəniz `--max_unsafe 0` ilə bütün UNSAFE dataseti götürüb SAFE-ni də ona uyğun balanslaya bilərsiniz.

---

## Quraşdırma

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

> NER metric üçün `seqeval` modulunu quraşdırmaq lazımdır:

```powershell
pip install seqeval
```

---

## 1) Guardrail dataset yaratmaq (`train_classifier.json`)

Sürətli iterasiya üçün (tövsiyə olunan):

```powershell
python scripts/build_train_classifier_json.py --filter_safe_pii --max_unsafe 8000
```

Tam dataset üçün:

```powershell
python scripts/build_train_classifier_json.py --filter_safe_pii --max_unsafe 0
```

Nəticə faylı:

* `data/processed/train_classifier.json` (JSONL formatında)

---

## 2) Guardrail modelini train etmək (sürətli)

CPU üçün praktik parametrlər:

```powershell
python -m pii_guard.training.train_classifier --epochs 1 --batch 32 --max_len 96
```

Model burada saxlanır:

* `models/classifier/pytorch/`

---

## 3) Synthetic NER data yaratmaq

Sürətli demo üçün:

```powershell
python scripts/synthetic_ner_generator.py --n 5000
```

Daha çox data üçün:

```powershell
python scripts/synthetic_ner_generator.py --n 20000
```

Output:

* `data/synthetic/ner_bio.jsonl`

---

## 4) NER modelini train etmək (sürətli)

```powershell
python -m pii_guard.training.train_ner --epochs 1 --batch 128
```

Model burada saxlanır:

* `models/ner/pytorch/`

---

## 5) ONNX export + INT8 Quantization

### ONNX export

```powershell
python -m pii_guard.optimization.export_onnx
```

### INT8 quantization

```powershell
python -m pii_guard.optimization.quantize
```

Nəticə faylları:

* `models/classifier/onnx/model.onnx`
* `models/classifier/onnx/model.int8.onnx`
* `models/ner/onnx/model.onnx`
* `models/ner/onnx/model.int8.onnx`

---

## 6) Benchmark (PyTorch vs ONNX vs INT8)

```powershell
python -m pii_guard.optimization.benchmark --n 80
```

Report:

* `reports/benchmarks/bench.json`

### Nümunə nəticələr (CPU)

Bu repoda alınmış real ölçü nəticələri (sizin sistemdə bir az fərqli ola bilər):

**Guardrail (Classifier)**

* PyTorch: ~47–55 ms
* ONNX: ~12 ms
* ONNX INT8: ~4 ms  ✅ (latency < 20ms rahatlıqla)

**Extractor (NER)**

* PyTorch: ~44 ms
* ONNX: ~22 ms
* ONNX INT8: ~7 ms

Model ölçüləri də təxminən **4x kiçilir** (ONNX → INT8).

---

## Qısa yekun

Bu tapşırıqda məqsəd “ən böyük model + ən uzun training” deyil, **real sistem kimi işləyən pipeline** qurmaqdır:

* dataset hazırlığı
* iki mərhələli qərar mexanizmi (cascading)
* ONNX export
* INT8 quantization
* real benchmark