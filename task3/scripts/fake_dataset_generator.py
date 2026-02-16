
#INSTALL--> pip install faker # Install the faker library
import random
import json
import string
from faker import Faker

# Konfiqurasiya
fake = Faker('az_AZ')
DATA_COUNT = 5000  # Yaradılacaq data sayı
OUTPUT_FILE = "ner_train_data.json"

# ------------------- 1. DATA GENERATORLAR (PII) -------------------

def get_fin():
    """7 simvollu FİN kod (Böyük hərf və Rəqəm)"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=7))

def get_phone():
    """Azərbaycan nömrələri (Müxtəlif formatlarda)"""
    operators = ["050", "051", "055", "070", "077", "099", "010"]
    prefix = random.choice(operators)
    main_num = "".join([str(random.randint(0, 9)) for _ in range(7)])

    # Modelin formatı əzbərləməməsi üçün müxtəliflik
    formats = [
        f"+994{prefix[1:]}{main_num}",               # +994501234567
        f"({prefix}) {main_num[:3]} {main_num[3:]}", # (050) 123 4567
        f"{prefix}-{main_num[:3]}-{main_num[3:5]}-{main_num[5:]}", # 050-123-45-67
        f"{prefix} {main_num}"                        # 0501234567
    ]
    return random.choice(formats)

def get_card():
    """16 rəqəmli kart (Boşluqlu və ya bitişik)"""
    nums = [str(random.randint(0, 9)) for _ in range(16)]
    if random.random() > 0.5:
        return "".join(nums)  # 4169123456789010
    else:
        return " ".join(["".join(nums[i:i+4]) for i in range(0, 16, 4)]) # 4169 1234 ...

# ------------------- 2. SAFE DATA GENERATOR (HARD NEGATIVES) -------------------

def generate_safe_text():
    """
    İçində rəqəmlər olan, amma PII olmayan cümlələr.
    Məqsəd: Model qiymətləri və ya tarixləri 'Telefon' və ya 'Kart' sanmasın.
    """
    brands = ["Samsung", "iPhone", "Xiaomi", "Honor", "Bosch", "LG", "Sony"]
    products = ["S24", "15 Pro", "Redmi Note 13", "paltaryuyan", "soyuducu", "televizor"]

    scenario = random.choice(['price', 'order', 'general'])

    if scenario == 'price':
        # Nümunə: "iPhone 15 qiyməti 2500 manatdır."
        brand = random.choice(brands)
        prod = random.choice(products)
        price = random.randint(100, 4000)
        return f"{brand} {prod} modeli nağd {price} AZN-ədir."

    elif scenario == 'order':
        # Nümunə: "Sifariş kodum 993322." (6 rəqəm - FİN deyil!)
        code = random.randint(100000, 999999)
        return f"Sifariş nömrəm #{code}, statusu nədir?"

    else:
        # Nümunə: "Mağaza saat 10:00-da açılır."
        hour = random.randint(9, 22)
        return random.choice([
            f"Mağazanız saat {hour}:00-da işləyir?",
            "Kredit faizləri neçə aydan başlayır?",
            "Menecerlə əlaqə saxlamaq istəyirəm.",
            "Çatdırılma pulsuzdur?"
        ])

# ------------------- 3. ŞABLONLAR (TEMPLATES) -------------------

templates_pii = [
    # Tək Entity
    ("Mənim adım {PERSON}-dir.", ["PERSON"]),
    ("Sadəcə {FIN} kodunu yoxlayın.", ["FIN"]),
    ("Əlaqə üçün: {PHONE}", ["PHONE"]),
    ("Kart məlumatım: {CARD}", ["CARD"]),

    # Multi-Entity (Bir cümlədə bir neçə dənə)
    ("Mən {PERSON}, fin kodum {FIN}.", ["PERSON", "FIN"]),
    ("Ad: {PERSON}, Tel: {PHONE}.", ["PERSON", "PHONE"]),
    ("Ödənişi {CARD} ilə etdim, adım {PERSON}.", ["CARD", "PERSON"]),
    ("Fin {FIN}, Ad {PERSON}, Nömrə {PHONE}.", ["FIN", "PERSON", "PHONE"])
]

# ------------------- 4. MAIN GENERATION LOOP -------------------

data = []

print(f"🚀 {DATA_COUNT} ədəd data yaradılır...")

for _ in range(DATA_COUNT):

    # 60% PII Data, 40% Safe Data
    if random.random() < 0.6:
        template, entity_types = random.choice(templates_pii)

        text = template
        entities = []

        # Şablondakı dəyərləri real data ilə əvəzləyirik
        for label in entity_types:
            if label == "PERSON": value = fake.first_name() + " " + fake.last_name()
            elif label == "FIN": value = get_fin()
            elif label == "PHONE": value = get_phone()
            elif label == "CARD": value = get_card()

            placeholder = f"{{{label}}}"

            # Mətndə placeholder varsa, əvəzlə və indeksləri tap
            if placeholder in text:
                text = text.replace(placeholder, value, 1)

                # Yeni mətndə dəyərin yerini tapırıq
                start = text.find(value)
                if start != -1:
                    entities.append({
                        "start": start,
                        "end": start + len(value),
                        "label": label
                    })

        data.append({"text": text, "entities": entities})

    else:
        # Safe Data (Boş entity siyahısı)
        safe_text = generate_safe_text()
        data.append({"text": safe_text, "entities": []})

# Faylı yadda saxlamaq
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Hazırdır! Fayl: {OUTPUT_FILE}")
print("Nümunə PII Data:")
print(json.dumps([d for d in data if len(d['entities']) > 0][:2], ensure_ascii=False, indent=2))