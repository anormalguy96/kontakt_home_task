# Task 2 - Turkish ASR Fine-tuning & Optimization Pipeline

Türk dilli ASR modeli qurmağı ehtiva edən bu tapşırığın əsas məqsədi yalnız aşağı WER almaq deyil, həm də modeli **production mühitinə uyğun optimizasiya etmək** (ölçünü kiçiltmək, inference sürətini artırmaq, deploy edilə bilən servis qurmaq) idi.

Dataset: `ysdede/khanacademy-turkish` (Hugging Face)

---

## Fokus seçimi

Bu tapşırıqda “Engineering Centric” (MLOps/Engineer) yanaşmasını seçmişəm.

**Niyə bu seçim?**
- Qısa müddətdə həm mükəmməl fine-tuning (aşağı WER), həm də tam MLOps/optimizasiya pipeline-ı yüksək keyfiyyətdə hazırlamaq çətindir.
- Bu repo-nun əsas dəyəri odur ki, avtomatlaşdırılmış pipeline, ONNX + INT8 quantization, FastAPI servis, Docker, və load testingi (Locust) özündə birləşdirsin.

**Nəticə gözləntisi:**
- Model bəzi hallarda səhv yazsa belə (WER yüksək ola bilər), sistem “production-ready” istiqamətdə hazırlanıb. Struktur səliqəlidir, ölçü/latency müqayisəsi var, servis konteynerdə işləyir və load testing də uğurla tamamlanıb.

---

## 1) Repo strukturu

- `scripts/train.py` – (opsional) dataset yükləmə + preprocess + qısa fine-tuning (subset ilə)
- `scripts/export.py` – modeli ONNX formatına export edir və INT8 quantization tətbiq edir
- `scripts/benchmark.py` – PyTorch vs ONNX ölçü və inference time müqayisəsi üçün report yaradır
- `app/main.py` – FastAPI servis (audio upload → JSON nəticə)
- `app/asr.py` – inference engine (PyTorch / ONNX / ONNX INT8)
- `locust/locustfile.py` – load testing (Locust)

---

## 2) Quraşdırma (lokal)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```
Nümunə test faylı:

`samples/sample.wav`

## 3) Model training (Fine-tuning) – opsional
Tapşırıq mətninə uyğun olaraq dataset üzərində fine-tuning üçün scripts/train.py əlavə olunub.
Vaxta qənaət üçün datasetin hamısı yox, train split-in kiçik hissəsi (məs: 10–20%) istifadə olunur.

```bash
python scripts/train.py \
  --model_name facebook/wav2vec2-base \
  --dataset_name ysdede/khanacademy-turkish \
  --train_percent 0.1 \
  --eval_percent 0.05 \
  --epochs 1 \
  --output_dir models/checkpoint
```

Tracking
Training zamanı TensorBoard log-ları yazılır (loss və WER izlənə bilər):

```bash
tensorboard --logdir runs
```

Qeyd: Bu tapşırıqda əsas fokus “Engineering Centric” olduğu üçün əsas iş axını export/optimizasiya/API üzərində qurulub.

## 4) Model optimizasiyası (ONNX + Quantization)
Production üçün model:

ONNX formatına export olunur

8-bit quantization (dynamic INT8) tətbiq edilir

Bu repo-da istifadə etdiyim model (Türkcə üçün hazır checkpoint):

cahya/wav2vec2-base-turkish

Export + INT8:

```bash
python scripts/export.py --checkpoint_dir "cahya/wav2vec2-base-turkish" --onnx_dir "models/onnx"
```

## 5) Benchmark Report (PyTorch vs ONNX ölçü və sürət)
Script:

```powershell
python scripts/benchmark.py \
  --checkpoint_dir "cahya/wav2vec2-base-turkish" \
  --onnx_path models/onnx/model.onnx \
  --onnx_int8_path models/onnx/model_int8.onnx \
  --audio_path samples/sample.wav \
  --runs 20
```

Nəticə [artifacts/benchmark_report.md](artifacts/benchmark_report.md) faylına yazılır. Bu run üçün ölçü və latency nəticələri:

### Model ölçüsü (MB)

PyTorch checkpoint folder: 0.0 MB (lokal models/checkpoint yaradılmadığı üçün; model HF cache-dən istifadə olunub)

ONNX (float): 360.32 MB

ONNX (int8): 116.28 MB

Inference time (seconds, average)
PyTorch: 1.5180 s

ONNX (float): 1.5463 s

ONNX (int8): 0.9125 s

## 6) MLOps: FastAPI servis
Lokal run:

### Windows PowerShell
```powershell
$env:ASR_BACKEND="onnx_int8"
$env:ASR_MODEL_DIR="models/onnx"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Swagger:
```
http://localhost:8000/docs
```
API test (curl)
```powershell
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@samples/sample.wav"
```
Output formatı:

```json
{"text": "...", "inference_time": 0.12}
```
## 7) Docker

```powershell
docker compose up --build
```

Sonra:

API: http://localhost:8000

Swagger: http://localhost:8000/docs

Container test:

```powershell
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@samples/sample.wav"
```

## 8) Load testing (Locust)
Locust start:
```powershell
locust -f locust/locustfile.py --host http://localhost:8000
```
Brauzer:
```powershell
http://localhost:8089
```

Nümunə run (10 user, ramp-up 2 user/s) nəticəsində:

Failures: 0%

Median: ~6200 ms

95%: ~11000 ms

Average: ~5846 ms

Current RPS: ~0.6

Qeyd: Bu rəqəmlər CPU inference, audio upload və real model decode prosesinə görə dəyişə bilər.

## Diqqətiniz üçün təşəkkürlər!

![alt text](https://i.pinimg.com/564x/3b/3b/f0/3b3bf06baa1190825dd76846d848ce06.jpg)