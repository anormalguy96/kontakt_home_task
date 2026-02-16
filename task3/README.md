# Task 3 — Hybrid PII Protection System (Guardrail + Extractor)

Bu repo müştərilərin çatbotla yazışanda paylaşa bildiyi həssas məlumatları (PII) avtomatik tapıb maskalamaq üçün hazırlanmış **iki mərhələli (cascading)** NLP pipeline-dır.

Məqsəd çox aydındır:
- Mesaj **təhlükəsizdirsə** (SAFE) → gecikmədən buraxılsın  
- Mesajda **PII varsa** (UNSAFE) → PII hissələr tapılıb `****` ilə maskalansın

Bu tapşırıqda məqsəd maksimum nəticəli model qurmaqdan çox, pipeline-ın real işləməsini və optimizasiyanı göstərmək idi.

---

## Ümumi arxitektura

### 1) Guardrail - Binary Classifier (SAFE / UNSAFE)
Birinci mərhələ mesajın təhlükəsiz olub-olmadığını **sürətli** müəyyən edir.

- **UNSAFE (Label=1):** `LocalDoc/pii_ner_azerbaijani`  
  Bu datasetdə PII olan cümlələr var — olduğu kimi götürülüb `label=1` verilir.
- **SAFE (Label=0):** `aznlp/azerbaijani-blogs`  
  Blog mətnləri cümlələrə bölünür və balans üçün UNSAFE sayına yaxın nümunə götürülür.

Axın:
- SAFE → dərhal cavab (Extractor işə düşmür)
- UNSAFE → Extractor mərhələsinə ötürülür

> Qeyd: UNSAFE tərəfdə 120k+ sətir görməyinizin səbəbi “sample” yox, dataseti bütöv split kimi load etməyimizdir. Sürətli test üçün isə `--max_unsafe` ilə limitləyirik.

---

### 2) Extractor - NER / TokenClassification (PII tapmaq)
Guardrail “UNSAFE” dedikdə ikinci model işə düşür və mətndə PII hissələri tapır:

- `PERSON`
- `FIN`
- `PHONE`
- `CARD`

Burada NER modeli BIO formatında öyrədilir:
- `B-...` → entity başlayır
- `I-...` → entity davam edir
- `O` → entity deyil

---

## NER datası: generator + BIO-ya çevirmə

`scripts/fake_dataset_generator.py` skripti sintetik data yaradır və nəticəni **span formatında** saxlayır:
- `ner_train_data.json` (text + entities[start/end/label])

Bizim NER training pipeline isə BIO formatı ilə işlədiyi üçün arada çevirmə addımı var:
- `scripts/convert_fake_to_bio.py` → `data/synthetic/ner_bio.jsonl` (tokens + tags)

Yəni əsas axın belədir:
1) fake generator JSON yaradır  
2) convert script onu BIO-ya çevirir  
3) train_ner BIO faylından modeli train edir

> Repo-da əlavə olaraq `scripts/synthetic_ner_generator.py` də saxlanılıb. Bu “fallback” üçündür: internet/faker problemi olanda, eyni pipeline-ı yenə də ayağa qaldırmaq mümkün olsun.

---

## Niyə full dataset ilə uzun training etmədim?

Bu tapşırıqda əsas gözlənti “ultimate model” yox, **pipeline + optimizasiya + düşüncə tərzi**dir.

CPU mühitində böyük dataset ilə transformer training iterasiya müddətini əhəmiyyətli dərəcədə artırır. Bu səbəbdən ilkin mərhələdə limitli dataset ilə sürətli iterasiya strategiyası seçilmişdir.

Bu sayədə həm end-to-end axın bağlandı, həm də ONNX/INT8 optimizasiya hissəsinə keçmək mümkün oldu.

---

## Quraşdırma (Windows / PowerShell)

```powershell
cd D:\github_repos\kontakt_home_task\task3

python -m venv .venv
.\.venv\Scripts\activate

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install faker
python -m pip install seqeval
```

`pii_guard` modulunun tapılması üçün:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

---

## 1) Guardrail dataset yaratmaq

Sürətli (tövsiyə olunan):

```powershell
python scripts/build_train_classifier_json.py --filter_safe_pii --max_unsafe 8000
```

Tam dataset:

```powershell
python scripts/build_train_classifier_json.py --filter_safe_pii --max_unsafe 0
```

Output:

* `data/processed/train_classifier.json`

---

## 2) Guardrail modeli train etmək

CPU üçün praktik run:

```powershell
python -m pii_guard.training.train_classifier --epochs 1 --batch 32 --max_len 96
```

Model:

* `models/classifier/pytorch/`

---

## 3) NER datasını yaratmaq və BIO-ya çevirmək

```powershell
python scripts/fake_dataset_generator.py
python scripts/convert_fake_to_bio.py --in ner_train_data.json --out data/synthetic/ner_bio.jsonl
```

Output:

* `data/synthetic/ner_bio.jsonl`

---

## 4) NER modeli train etmək

```powershell
python -m pii_guard.training.train_ner --epochs 1 --batch 128 --data data/synthetic/ner_bio.jsonl
```

Model:

* `models/ner/pytorch/`

> Qeyd: Sintetik data çox “təmiz” olduğu üçün metrikalar bəzən çox yüksək görünür. Burada məqsəd real dünyadakı performansı sübut etməkdən çox, pipeline-ın işləməsini və optimizasiya addımlarını göstərməkdir.

---

## 5) ONNX export + INT8 quantization + benchmark

Bir komand zənciri ilə:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path

python -m pii_guard.optimization.export_onnx
python -m pii_guard.optimization.quantize
python -m pii_guard.optimization.benchmark --n 80
```

Nəticə faylları:

* `models/classifier/onnx/model.onnx`
* `models/classifier/onnx/model.int8.onnx`
* `models/ner/onnx/model.onnx`
* `models/ner/onnx/model.int8.onnx`
* `reports/benchmarks/bench.json`

---

## Benchmark nəticələri

**Model ölçüləri**

* Classifier ONNX: ~516.34 MB → INT8: ~129.43 MB
* NER ONNX: ~514.10 MB → INT8: ~128.87 MB

**Orta gecikmə (ms, n=80)**

Classifier:

* PyTorch (SAFE): ~41.66 ms
* PyTorch (UNSAFE): ~54.41 ms
* ONNX (SAFE): ~20.77 ms
* ONNX INT8 (SAFE): ~4.17 ms (əla)

NER:

* PyTorch: ~44.22 ms
* ONNX: ~22.85 ms
* ONNX INT8: ~7.80 ms (əla)

Yekun:

* INT8 quantization ölçünü təxminən 4x kiçildir
* ONNX + INT8 CPU mühitində gecikməni ciddi şəkildə aşağı salır

---

## `fake_dataset_generator.py` faylı haqqında

Tapşırıq mətnində qeyd olunan `synthetic_data_generator.py` əvəzinə `fake_dataset_generator.py` faylı göndərilmişdir.

Bu skript:
- `ner_train_data.json` faylı yaradır
- Format: `text` + `entities (start, end, label)` (span-based annotation)

Lakin NER training pipeline BIO formatı ilə işlədiyi üçün arada çevirmə addımı əlavə edilmişdir:

```powershell
python scripts/fake_dataset_generator.py
python scripts/convert_fake_to_bio.py --in ner_train_data.json --out data/synthetic/ner_bio.jsonl
```

Beləliklə:

Generator → raw span data

convert_fake_to_bio.py → BIO format

train_ner.py → BIO ilə model training

Repo-da əlavə olaraq scripts/synthetic_ner_generator.py də saxlanılıb. Bu alternativ generator yalnız fallback məqsədi daşıyır (məsələn, internet/faker problemi olduqda pipeline-ın yenə də işləməsi üçün).


---

## Qısa yekun

Bu repo tapşırığın istədiyi kimi “real sistem” axınını tam göstərir:

* dataset hazırlığı (SAFE/UNSAFE balansı)
* 2 mərhələli qərar mexanizmi (cascading)
* NER üçün generator + BIO çevirmə
* ONNX export
* INT8 quantization
* benchmark ilə real ölçmə

## Diqqətiniz üçün təşəkkürlər!