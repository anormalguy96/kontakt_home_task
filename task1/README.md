## Task 1 (QC Prototype)

Bu repository müştəri xidməti zənglərinin transkript JSON fayllarını analiz edib *keyfiyyət meyarları* üzrə (KR2.1–KR2.5) skorlayan funksional prototipdir.

### Giriş və çıxış formatı

#### Giriş

Sistem `call_id` və `segments[]` qəbul edir. `segments` daxilində həm `start/end`, həm də `start_time/end_time` formatı qəbul olunur.

Minimal nümunə:

```json
{
  "call_id": "12345",
  "segments": [
    {"speaker": "Operator", "text": "Salam...", "start_time": 0.5, "end_time": 4.2},
    {"speaker": "Customer", "text": "Internet işləmir.", "start_time": 5.0, "end_time": 7.1}
  ]
}
```

Dataset formatında göndərmək də mümkündür (platformadan* asılı olur):

```json
{
  "dataset_id": "dataset_001",
  "input": {
    "call_id": "12345",
    "segments": [...]
  }
}
```

*Request-in göndərildiyi sistem/interfeys.
* **Swagger UI** (FastAPI `/docs`) -> 
```json
{
  "call_id": "12345",
  "segments": [...]
}
```
* **LLM Evaluation Platformaları** -> 
```json
{
  "dataset_id": "dataset_001",
  "input": {
    "call_id": "12345",
    "segments": [...]
  }
}
```
* **IES** (Internal Evaluation Script) -> 
```json
{
  "dataset_id": "dataset_001",
  "input": {
    "call_id": "12345",
    "segments": [...]
  }
}
```
#### Çıxış

Hər meyar üçün:

* `score` (0–3)
* `reasoning` (qısa izah)
* `probability` (`HIGH` / `LOW`)
* `evidence_snippet` (transkriptdən ən uyğun parça)

### Quraşdırma (lokal)

#### 1) Repository

Lokalımda hazır olan folderə keçmək üçün öncə bu komandadan istifadə etdim:

```powershell
cd D:\github_repos\kontakt_home_task\task1
```
Lakin GitHub-dan klonlasaq:

```powershell
git clone https://github.com/anormalguy96/kontakt_home_task.git
cd task
```

#### 2) Virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 3) Testlər

```powershell
pytest -q
```

#### 4) Evaluation dataset ilə yoxlama

```powershell
python evaluate.py
```

Mənim nəticəm: `Overall accuracy: 0.993` (993/1000)

#### 5) API-ni işə salmaq

Repository root-dan:

```powershell
uvicorn qc_service.main:app --reload --app-dir src
```

Sonra brauzerdə:

* Swagger: `http://localhost:8000/docs`
* Health: `http://localhost:8000/health`
* Evaluate: `POST http://localhost:8000/evaluate`

### LLM seçimi: niyə Groq?

Bu tapşırıqda məqsəd ən güclü model deyil, **pipeline məntiqidir**. Odur ki, ən optimal versiyada olan LLM-i deyil, **bizə** ən optimal halı seçməliyik.

Groq platformasını seçməyimin bir neçə praktik və texniki səbəbi var:

* Yüksək sürətli inference – cavab müddətinin qısa olması development prosesini əhəmiyyətli dərəcədə sürətləndirir.

* Free tier – əlavə maliyyə yükü yaratmadan prototip hazırlamaq mümkündür.

* Llama və Mixtral kimi modellərə rahat çıxış – açıq mənbəli modellərlə işləmək həm sürətli, həm də vendor lock-in riskini azaldır.

* Rule-based və LLM əsaslı yanaşma arasında balans qurmaq üçün uyğun seçimdir – Groq sürətli və yüngül olduğu üçün yalnız zəruri hallarda model çağırışı etmək və əsas məntiqi Python alqoritmləri üzərində qurmaq strategiyasını effektiv şəkildə nümayiş etdirməyə imkan verir.

### Groq konfiqurasiyası

`.env.example` faylını `.env` kimi kopyalayın və açarı əlavə edin:

```powershell
copy .env.example .env
```

`.env` içində:

```
GROQ_API_KEY=...
```
Bu hissədə isə öz groq api key-inizi əlavə etməlisiniz ([API key üçün link](https://console.groq.com/keys)).

### Hibrid yanaşma: Rule-based nə vaxt, LLM nə vaxt?

Bu prototipdə əsas prinsip belədir:

* **Rule-based**: stabil, təkrarlana bilən, “qızıl qaydalar” (məs: CVV soruşmaq, etik qaydalar, bəzi açar pattern-lər)
* **LLM-based (opsional)**: mətn çox qeyri-müəyyən olanda və ya rule-ların “qırıldığı” hallarda semantik qərar üçün

Bu yanaşma iki problemi həll edir:

1. Hər şeyi modelə yükləmədən sürətli və ucuz qalır
2. “Hallucination” riskini azaldır (LLM-in cavabı yalnız transkriptdəki real mətnlə uyğun olduqda qəbul edilir, yəni kor-koranə qəbul edilir)

### Robustness / edge-case yanaşmaları

Layihə aşağıdakı ssenariləri nəzərə alır:

* Qarışıq dillər (AZ/RU/EN) - pattern-lər və normalizasiya müxtəlif dillərdə işləyəcək şəkildə seçilib.
* Hallucination riski - LLM istifadə olunsa belə, nəticə yalnız transkriptdəki real mətnlə uyğun olduqda qəbul edilir (və `evidence_snippet` tələb olunur).
* Boş və ya qırıq seqmentlər - `text: ""` və ya `text: "..."` gələndə sistem crash etmir, warning log yazır.
* PII mövzusu - CVV/CVC kimi həssas məlumatların soruşulması “hard violation” kimi qiymətləndirilir.

### DevOps / reproducibility

* Kod idempotent və təkrarlana bilən işləyir (evaluation dataset ilə ölçülür).
* `tests/` altında unit testlər var.
* `.github/workflows/` altında CI pipeline var (push zamanı testlər avtomatik işləyir).
* Docker / docker-compose dəstəyi əlavə olunub (əgər repo-da mövcuddursa).

### Repo strukturu

* `src/qc_service/` — əsas tətbiq
* `src/qc_service/rules/` — rule-engine
* `data/eval/` — evaluation dataset
* `tests/` — unit testlər
* `prompts/` — prompt management (yaml)

### Potensial çətinliklər və həllər

* Input format fərqləri - dataset `start/end` istifadə edir, nümunə isə `start_time/end_time`. Normalizasiya qatında hər ikisi qəbul edildi.
* Real “noise” - boş seqmentlər, non-numeric time dəyərləri — crash əvəzinə warning və default davranış.
* Məntiqi səhvlər - bəzi hallarda `and/or` prioriteti yanlış nəticə verə bilər; bu tip yerlər explicit mötərizə ilə düzəldildi.

### Təkmilləşdirmə ideyaları

* PII detection üçün daha güclü masking ola bilər (məs: kart nömrəsini `**** **** **** 1234`; hansı ki, bununla əlaqədar [proyekt](../task3/) `task3` folderində yerləşdirilmişdir)
* LLM nəticələri üçün daha ciddi “evidence grounding” tətbiq edilə bilər (segment id-ləri ilə)
* Daha geniş evaluation dataset və “regression” testləri mümkündür
* Həmçinin **observability** - structured logging + request id + latency metrik kimi əlavə funksiyalar əlavə edilə bilər.

### Diqqətiniz üçün təşəkkürlər!

![alt text](https://i.pinimg.com/564x/3b/3b/f0/3b3bf06baa1190825dd76846d848ce06.jpg)