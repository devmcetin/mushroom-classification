# Data Science Project 35 — Mantar: Yenilebilir mi Zehirli mi? (Decision Tree)

**Modül**: ML-04 (Sınıflandırma 2) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir **doğa keşif uygulaması** için data scientist olarak çalışıyorsun. Kullanıcılar ormanda gördükleri bir mantarın fiziksel özelliklerini giriyor (şapka şekli, koku, lamel rengi, sap yapısı...) ve uygulama onlara mantarın **yenilebilir mi yoksa zehirli mi** olduğunu söylüyor.

Bu klasik bir classification problemi gibi görünüyor ama bir farkı var: **insan hayatı söz konusu**. Bu yüzden modelin sadece doğru tahmin etmesi yetmez — verdiği kararı bir uzmanın (botanikçi/doktor) **doğrulayabilmesi** gerekir.

İşte tam burada **Decision Tree** parlıyor: ürettiği kurallar **okunabilir**.

> "Eğer koku küflü ise → ZEHİRLİ"
> "Eğer koku yok ve spor baskısı beyaz değilse → YENİLEBİLİR"

Kara kutu bir model "%99 zehirli" der ama nedenini söyleyemez. Decision Tree ise kararının arkasındaki **kuralı gösterir**. Gıda güvenliği, tıp, kredi gibi kritik alanlarda bu **yorumlanabilirlik** çoğu zaman birkaç puanlık accuracy'den daha değerlidir.

Bu projede ML-04 dersinde öğrendiklerini birleştireceksin:
- ✅ **DecisionTreeClassifier** ve gini/entropi mantığı
- ✅ **Kategorik veri → one-hot encoding** (tüm feature'lar harf kodlu)
- ✅ **plot_tree** ile ağacı görselleştirip **kural okuma**
- ✅ **feature_importances_** — hangi özellik en ayırıcı? (spoiler: koku)
- ✅ **max_depth tuning** — derinlik vs overfit dengesi
- ✅ Decision Tree'de **scaling neden GEREKMEZ** (mesafe tabanlı değil)
- ✅ Logistic Regression ile **kıyas** (accuracy vs yorumlanabilirlik)

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-35

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 715 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📊 Veri Seti

**Kaynak**: UCI Machine Learning Repository — *Mushroom Data Set* (1987, The Audubon Society Field Guide to North American Mushrooms). sklearn'in OpenML köprüsüyle çekiliyor.

**İndirme**: Dosya repo'da DEĞİL — kod `fetch_openml` ile otomatik indirir:

```python
from sklearn.datasets import fetch_openml
data = fetch_openml('mushroom', version=1, as_frame=True)
X = data.data    # 8124 × 22 kategorik feature
y = data.target  # 'e' / 'p'
```

İlk çağrıda internet gerekir; sonra `~/scikit_learn_data` altına **cache'lenir** (sonraki çalıştırmalar offline). Bu yüzden `data/` klasörü gerekmez (yine de `.gitignore`'da).

**Boyut**: 8124 mantar × 22 kategorik feature + 1 hedef
**Hedef**: `e` (edible / yenilebilir) vs `p` (poisonous / zehirli)
**Sınıf dağılımı**: 4208 yenilebilir / 3916 zehirli → **~%48.2 zehirli** (neredeyse dengeli, oversampling gerekmez)

### Tüm Feature'lar Kategorik

22 feature'ın **hepsi kategorik** ve **harf koduyla** saklanır (sayısal hiçbir sütun yok). Örnek feature'lar:

| Feature | Ne ölçer? | Örnek harf kodları |
|---------|-----------|--------------------|
| `cap-shape` | Şapka şekli | `b`=çan, `c`=konik, `x`=dışbükey, `f`=düz, `k`=topuz, `s`=çukur |
| `cap-color` | Şapka rengi | `n`=kahve, `b`=devetüyü, `g`=gri, `r`=yeşil, `e`=kırmızı, `w`=beyaz, `y`=sarı... |
| `odor` | **Koku** | `a`=badem, `l`=anason, `c`=küflü, `f`=balık, `m`=küf, `n`=kokusuz, `p`=keskin, `s`=baharatlı |
| `gill-color` | Lamel (solungaç) rengi | `k`=siyah, `n`=kahve, `g`=gri, `p`=pembe, `w`=beyaz, `h`=çikolata... |
| `spore-print-color` | Spor baskısı rengi | `k`=siyah, `n`=kahve, `w`=beyaz, `h`=çikolata, `r`=yeşil... |

> Tam 22 feature: cap-shape, cap-surface, cap-color, bruises, **odor**, gill-attachment, gill-spacing, gill-size, gill-color, stalk-shape, stalk-root, stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring, stalk-color-below-ring, veil-type, veil-color, ring-number, ring-type, spore-print-color, population, habitat.

Tüm feature'lar harf kodlu olduğu için modele vermeden önce **one-hot encode** etmelisin (`pd.get_dummies`). Bu, her kategoriyi ayrı bir 0/1 sütununa çevirir.

### 🍄 Domain Notu — "Kokla yeter"

Mantar veri setinin meşhur gerçeği: **`odor` (koku) tek başına neredeyse mükemmel ayırıcıdır.**

- Kokusu **badem (`a`)** veya **anason (`l`)** olanlar → büyük çoğunlukla yenilebilir
- Kokusu **küflü/balık/keskin/baharatlı** olanlar → neredeyse tamamı zehirli
- `odor='n'` (kokusuz) grubu hariç, koku tek başına ~%98 ayırma yapar

Bu yüzden eğittiğin ağacın `feature_importances_` çıktısında **odor baskın** çıkacak ve ağacın **kökünde** koku ile dallanma göreceksin. Decision Tree'nin yorumlanabilirliği işte tam bunu görmeni sağlıyor: model "koku küfse zehirli" kuralını kendiliğinden buluyor — bir mantar uzmanının onaylayacağı bir kural.

> ⚠️ Uyarı: Bu model akademik bir benchmark'tır. Gerçek hayatta hiçbir mantarı bir ML modeline güvenerek yemeyin!

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **14 fonksiyonu** sırayla doldur. Her task altta testler pass olana kadar düzenlenmeli.

1. **`load_mushroom_data()`** — `fetch_openml` ile veriyi çek → (X, y)
2. **`explore_data(X, y)`** — örnek/feature sayısı, sınıf dağılımı
3. **`encode_features(X)`** — kategorikleri one-hot encode et
4. **`encode_target(y)`** — `'p'`→1 (zehirli), `'e'`→0
5. **`split_data(X, y)`** — train/test böl (stratify, random_state=42)
6. **`train_tree(X_train, y_train, max_depth)`** — DecisionTree eğit
7. **`evaluate_model(model, X_test, y_test)`** — accuracy/precision/recall/f1/cm
8. **`get_feature_importances(model, feature_names, top_n)`** — en önemli feature'lar
9. **`plot_decision_tree(model, feature_names, class_names, max_depth)`** — `plot_tree` çizimi
10. **`tune_max_depth(...)`** — farklı derinliklerde accuracy
11. **`predict_mushroom(model, X_row)`** — tek mantar tahmini + etiket
12. **`compare_with_lr(...)`** — Decision Tree vs Logistic Regression
13. **`count_perfect_splits(model)`** — gini=0 olan saf yaprak sayısı (bonus)
14. **`run_pipeline()`** — tüm akışı uçtan uca çalıştır

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] `fetch_openml` ile OpenML'den veri çekmeyi öğreneceksin
- [x] Kategorik veriyi **one-hot encode** etmeyi (`pd.get_dummies`) uygulayacaksın
- [x] `DecisionTreeClassifier` eğitmeyi ve değerlendirmeyi yapacaksın
- [x] `plot_tree` ile ağacı çizip **kuralları okumayı** öğreneceksin
- [x] `feature_importances_` ile hangi feature'ın baskın olduğunu (odor!) göreceksin
- [x] `max_depth` ile **underfit/overfit** dengesini gözlemleyeceksin
- [x] Decision Tree'de **scaling neden gereksiz** olduğunu anlayacaksın
- [x] Yorumlanabilir model (DT) ile kara kutu (LR) arasındaki farkı kavrayacaksın

## 🧪 Testler

Test dosyası: `tests/test_question.py` (16+ test)

Tümü pass olmalı:
- Veri yükleme (8124 örnek, 22 feature)
- Sınıf dağılımı (4208 e / 3916 p)
- one-hot encoding feature sayısını arttırıyor mu
- `encode_target` zehirli=1 yapıyor mu
- Stratify çalışıyor (~%48 zehirli)
- Default ağaçta accuracy > 0.95
- Feature importance sıralı ve **odor baskın**
- `plot_decision_tree` hatasız çalışıyor
- `max_depth` arttıkça accuracy artıyor
- `predict_mushroom` etiketi prediction ile tutarlı
- DT ve LR ikisi de > %90
- `run_pipeline` uçtan uca çalışıyor

## 📊 Beklenen Sonuçlar

```
Örnek sayısı       : 8124  (4208 yenilebilir / 3916 zehirli)
One-hot feature    : 22 → 117 sütun
Train/Test         : 5686 / 2438 (stratify)
Test accuracy (max_depth=5) : ~%99-100  (problem kolay)
En önemli feature  : odor_n (kokusuz) — odor baskın çıkar
max_depth tuning   : depth=1 bile ~%88, depth=5 → ~%100
Decision Tree vs LR: ikisi de ~%99-100, ama DT yorumlanabilir
```

## 💡 İpuçları

- **Scaling YOK** — Decision Tree mesafe tabanlı değil, eşik-tabanlı dallanır. Sadece kategorikleri sayısala çevir (one-hot).
- `pd.get_dummies(X)` → tüm kategorik sütunları otomatik one-hot yapar
- `model.feature_importances_` → her sütunun önem skoru (0-1 arası, toplamı 1)
- `plot_tree(..., filled=True, max_depth=3)` → ilk 3 seviyeyi renkli çiz
- `model.tree_.impurity` → her düğümün gini değeri; `==0.0` → saf yaprak
- Ağaç **deterministik** olsun diye `random_state=42` her yerde
- `odor` tek başına çok güçlü → ağacın kökünde göreceksin

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
