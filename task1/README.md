## Task 1 — Zəng Keyfiyyətinin Qiymətləndirilməsi Sistemi

### Layihənin məqsədi

Bu layihə müştəri xidmətləri zəng transkriptlərinin avtomatlaşdırılmış şəkildə qiymətləndirilməsi üçün hazırlanmış funksional prototipdir. Sistem zəng keyfiyyətini aşağıdakı 5 əsas meyar üzrə ölçür və strukturlaşdırılmış nəticə təqdim edir:

- **KR2.1 — sahiblik və proaktiv dəstək**
- **KR2.2 — anlama və effektiv kommunikasiya**
- **KR2.3 — həllin keyfiyyəti**
- **KR2.4 — prosesin idarə olunması və növbəti addımlar**
- **KR2.5 — peşəkarlıq və uyğunluq (PII təhlükəsizliyi daxil olmaqla)**

Layihənin əsas məqsədi:

- Deterministik və izah edilə bilən (interpretable) scoring mexanizmi təqdim etmək
- Hallüsinasiyalara qarşı qorunan qiymətləndirmə modeli qurmaq
- Genişlənə bilən (scalable) arxitektura təmin etmək

---

### Sistem arxitekturası

Sistem modul əsaslı arxitekturaya malikdir:

```
src/kontakt_qc/
│
├── types.py        # Data modelləri və parsing
├── preprocess.py   # Normallaşdırma və PII maskalama
├── rules.py        # Qayda əsaslı scoring mühərriki
├── pipeline.py     # Qiymətləndirmə orkestrasiyası
└── cli.py          # Komanda sətri interfeysi
```

Əlavə komponentlər:

- `evaluate.py` — verilmiş dataset üzərində accuracy hesablanması
- `tests/` — avtomatlaşdırılmış testlər (pytest)
- `Dockerfile` və `docker-compose.yml` — konteynerləşdirmə
- `.github/workflows/` — CI inteqrasiyası

---

### Texniki yanaşma

#### Əsas qiymətləndirmə mexanizmi (rule-based)

Sistem ilkin olaraq qayda-əsaslı yanaşmadan istifadə edir. Bunun səbəbləri:

- Deterministik davranış
- API xərclərinin olmaması
- Hallüsinasiyaların qarşısının alınması
- İzah edilə bilən nəticələr

Qiymətləndirmə:

- Pattern matching
- Kontekstual açar söz analizləri
- Prosessual indikatorların aşkarlanması
- PII və daxili məlumat sızmasının müəyyən edilməsi

---

#### Hibrid (LLM) yanaşma

Sistem istəyə bağlı olaraq LLM inteqrasiyasını dəstəkləyir.

Dəstəklənən rejimlər:

- `rule` (default)
- `hybrid`
- `llm`

LLM istifadəsi aşağıdakı hallarda məqsədəuyğundur:

- Uzun və kompleks transkriptlər
- Qarışıq dil strukturları
- Daha zəngin semantik izah tələb olunduqda

#### Anti-hallüsinasiyaya qarşı mexanizm

LLM nəticələri qəbul edilməzdən əvvəl aşağıdakılar yoxlanılır:

1. Model tərəfindən qaytarılmış **evidence snippet**
2. Həmin snippet-in transkript daxilində sözbəsöz mövcudluğu
3. Mövcud deyilsə → nəticə rədd edilir və rule-based mexanizmə geri dönülür

Bu mexanizm sistemin audit edilə bilən və etibarlı qalmasını təmin edir.

---

### Robustluq və səhv hallarının idarə edilməsi

Sistem aşağıdakı halları təhlükəsiz şəkildə idarə edir:

- Boş və ya zədələnmiş JSON payload
- Eksik sahələr
- Qarışıq timestamp formatları
- Ümumi müddət `< 0.1 saniyə`
- Segmentlərin olmaması

Belə hallarda sistem crash olmur və spesifikasiyaya uyğun olaraq `score = 0` qaytarır.

---

### PII Təhlükəsizliyi

Sistem aşağıdakı həssas məlumatları aşkarlayır:

- Kredit kartı tipli rəqəm ardıcıllıqları
- CVV/CVC
- FIN kodları

Təhlükəsizlik prinsipləri:

- Evidence snippet-lər maskalanır
- Daxili sistem məlumatlarının sızması flag edilir
- Çıxışda heç bir açıq PII saxlanılmır

---

### Qiymətləndirmə və ölçmə

Verilmiş `Task_1_Eval_dataset.json` dataset üzərində sistem:

- Hər KR üzrə accuracy
- Ümumi accuracy

hesablayır.

Qiymətləndirmə skripti:

```bash
python evaluate.py
```

---

### Konteynerləşdirmə və CI

#### Docker

```bash
docker build -t kontakt-qc .
docker run --rm kontakt-qc
```

#### CI

GitHub Actions:

- Testlərin avtomatik icrası
- Kod sabitliyinin təmin edilməsi

---

### Performans və genişlənmə potensialı

Gələcək inkişaf istiqamətləri:

- Real call-center datası ilə lüğətlərin zənginləşdirilməsi
- Dil aşkarlama modulu
- LLM modulu üçün daha sərt validasiya mexanizmləri
- FastAPI servis qatının əlavə olunması
- Yük testləri və performans optimizasiyası

---

### Nəticə

Hazırkı prototip:

- Deterministik
- Audit edilə bilən
- Hallüsinasiyaya qarşı qorunan
- Konteynerləşdirilmiş
- Testlərlə dəstəklənmiş
- Genişlənə bilən arxitekturaya malik

bir zəng keyfiyyəti qiymətləndirmə sistemidir.
