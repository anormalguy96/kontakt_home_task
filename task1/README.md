## Task 1 — Zəng Keyfiyyətinin Qiymətləndirilməsi Sistemi (Kontakt)

Bu layihə müştəri xidmətləri zəng transkriptlərini avtomatik qiymətləndirən prototipdir. Fokus “model gücü” deyil, _rule + LLM + guardrails_ məntiqidir.

## 1) Layihənin məqsədi

Sistem zəng keyfiyyətini 5 əsas meyar üzrə ölçür və strukturlaşdırılmış nəticə qaytarır:

- **KR2.1 — sahiblik və proaktiv dəstək**
- **KR2.2 — anlama və effektiv kommunikasiya**
- **KR2.3 — həllin keyfiyyəti**
- **KR2.4 — prosesin idarə olunması və növbəti addımlar**
- **KR2.5 — peşəkarlıq və uyğunluq (PII təhlükəsizliyi daxil olmaqla)**

Əsas məqsədlər:

- Deterministik və izah edilə bilən skorinq (rule-based)
- LLM istifadə olunarsa belə, anti-hallucination qoruması (ki, bu da **evidence verification** vasitəsilə həyata keçirilir)
- PII sızmasının qarşısının alınması
- Robust parsing (boş/broken JSON faylları, çatışmayan məlumatlar, qarışıq timestamp-lar)
- Test + CI + Docker ilə “production mindset”

---

## 2) Repo xəritəsi (Repo Map)

```cmd
task1/
  src/kontakt_qc/
    cli.py
    pipeline.py
    rules.py
    preprocess.py
    llm.py
    prompt_loader.py
    models.py
    hybrid.py
  prompts/
    kr_scoring_system_prompt.yaml
  docs/
    Task-1.docx
    Task_1_Eval_dataset.json
  tests/
    test_*.py
  evaluate.py
  Dockerfile
  docker-compose.yml
  .github/workflows/ci.yml
  requirements.txt
  README.md
  .gitignore
  .env.example
```

## 3) Quraşdırma (Local Setup)

### 3.1 Virtual env

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3.2 Tələb olunan modulların quraşdırılması

```powershell
pip install -r requirements.txt
pip install -e .
```

### 3.3 Testlərin icrası

```powershell
python -m pytest -q
```

## 4) Usage

### 4.1 CLI ilə işlətmə

**Qeyd:** CLI parametrləri cli.py-də necə implement olunubsa ona uyğun işləyir.

Nümunə (concept):

```powershell
python -m kontakt_qc.cli --input sample.json --mode rule --output out.json
```

Mode-lar:

**rule (default):** LLM tələb etmir
**hybrid:** rule + (opsional) LLM, guardrails ilə
**llm:** yalnız LLM (yenə guardrails saxlanılır)

### 4.2 Evaluation (dataset accuracy)

```powershell
python evaluate.py --mode rule
python evaluate.py --mode hybrid
python evaluate.py --mode llm
```

`evaluate.py` nəticəni JSON şəklində stdout-a yazır (automation-friendly).

## 5) Environment variables

### 5.1 Ümumi konfiqurasiya

```python
KONTAKT_QC_MODE=rule|hybrid|llm
LOG_LEVEL=INFO|DEBUG
```

### 5.2 LLM opsional parametrlər (API məcburi deyil)

Default MODE=rule heç bir API açarı tələb etmir.

LLM aktivləşdirmək üçün:

```python
KONTAKT_LLM_PROVIDER=groq|stub|none
KONTAKT_LLM_MODEL=...          # istəyə bağlıdır (defolt: llama-3.1-8b-instant)
KONTAKT_LLM_TIMEOUT_SECONDS=30 # istəyə bağlıdır
GROQ_API_KEY=...               # yalnız provider=groq üçün
```

Windows (PowerShell) nümunə:

```powershell
setx KONTAKT_LLM_PROVIDER groq
setx GROQ_API_KEY "YOUR_KEY"
setx KONTAKT_LLM_MODEL "llama-3.1-8b-instant"
```

## 6) Texniki yanaşma və qərarlar

### 6.1 Niyə rule-based default?

- Deterministik davranış (predictable)
- İzah edilə bilən nəticələr
- API xərci yoxdur
- Hallüsinasiyadan minimum risk

### 6.2 Hybrid/LLM nə zaman işə düşür?

LLM yalnız hybrid/llm rejimində aktiv olur.

Bunun məqsədi uzun/kompleks transkriptlərdə semantik interpretasiya aparmaq, həmçinin qarışıq dil strukturlarını (multilang) düzgün analiz etməkdir.

Əsas fokus model seçimi deyil, rule + LLM + guardrail kombinasiyasıdır.

### 6.3 Anti-hallucination guard (evidence verification)

LLM nəticələri qəbul olunmamışdan əvvəl:

- model “evidence snippet” qaytarır
- həmin snippet transkript daxilində sözbəsöz (normalise olunmuş) tapılmalıdır
- **tapılmırsa** → LLM nəticəsi rədd edilir və fallback tətbiq olunur

### 6.4 PII təhlükəsizliyi (PII Safety)

Transkript LLM-ə göndərilməzdən əvvəl PII maskalanır

Output-da açıq PII saxlanılmır (evidence/reasoning daxil)

## 7) Hard guards & robustluq

### 7.1 <0.1s short-audio guard

Əgər ümumi müddət **< 0.1** saniyə olarsa:

- model **çağırılmır** (LLM skip)
- sistem **crash olmur**
- **score=0** qaytarılır

### 7.2 Broken/empty input

Boş/zədələnmiş JSON payload

Missing fields

Qarışıq timestamp formatları

Segmentlərin olmaması
halları “graceful” şəkildə idarə olunur (warning logs + safe fallback).

## 8) LLM seçimi və səbəb (Groq) — səbəb → nəticə

Bu tapşırıq üçün ödənişli API-lərdən istifadə məcburi deyil. Layihə default olaraq heç bir LLM olmadan işləyir (MODE=rule).

Opsional provider kimi Groq seçilib, çünki:

- **Səbəb 1**: Çox sürətli inference (aşağı latency)
- **Nəticə**: Pipeline iterasiyası, test və debugging daha sürətli olur; real-time ssenariyə daha uyğun görünür.

- **Səbəb 2**: Pulsuz tier mövcuddur / aşağı xərcli işləyir
- **Nəticə**: Tapşırığın “ödənişsiz məcburi deyil” şərti pozulmur; eyni zamanda LLM inteqrasiyası göstərilir.

- **Səbəb 3**: OpenAI-compatible API formatı
- **Nəticə**: Adapter sadə qalır (vendor lock-in az), pipeline hissəsi aydın və audit edilə bilən olur.

Qeyd: LLM istifadə olunsa belə, nəticələr guardrails ilə qorunur:

- JSON parse (best-effort) + fallback

evidence check (anti-hallucination)

PII masking (defense-in-depth)

## 9) Docker

```powershell
docker build -t kontakt-qc .
docker run --rm kontakt-qc python -m pytest -q
```

## 10) Continuous Integration (CI)

GitHub Actions workflow:

pytest testlərinin avtomatik icrası

basic quality gates

## 11) Potensial çətinliklər və həllər

**Inconsistent timestamp formatları** → robust parsing + safe defaults

**Multilang transkript** → normalize + hybrid option

**Hallüsinasiyalar** → evidence verification + fallback

**PII sızması** → mask_pii_in_text (input+output)

## 12) Gələcək inkişaf (Improvements)

Daha geniş PII pattern-lər (telefon/email və s.)

Language detection + per-language keyword lexicon

KR-lərə görə daha zəngin error analysis (confusion cases)

FastAPI service layer + load tests + structured observability
