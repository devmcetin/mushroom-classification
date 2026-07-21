"""
DS-35 — Mantar: Yenilebilir mi Zehirli mi? (Decision Tree)

Bir doğa uygulaması için ekipte data scientist'sin. Kullanıcılar bir mantarın
fiziksel özelliklerini (şapka şekli, koku, renk...) giriyor; senin modelin bu
özelliklerden mantarın YENİLEBİLİR mi yoksa ZEHİRLİ mi olduğunu tahmin ediyor.

Decision Tree seçiyoruz çünkü ürettiği kurallar OKUNABİLİR:
"Eğer koku küflü ise → ZEHİRLİ". Bir botanikçi/doktor bu kuralları
doğrulayabilir. Gıda güvenliği gibi kritik alanlarda bu yorumlanabilirlik şart.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Veri setini yükle
def load_mushroom_data():
    """
    UCI Mushroom veri setini sklearn'in OpenML köprüsüyle indir.

    Returns:
        tuple: (X, y)
            X: pd.DataFrame — 8124 satır × 22 kategorik feature (harf kodlu)
            y: pd.Series — hedef, 'e' (edible/yenilebilir) veya 'p' (poisonous/zehirli)

    İpucu:
    - from sklearn.datasets import fetch_openml
    - data = fetch_openml('mushroom', version=1, as_frame=True)
    - X = data.data, y = data.target
    - fetch_openml otomatik cache'ler (~/scikit_learn_data) — internet tek sefer gerekir.
    """
    
    data = fetch_openml('mushroom', version=1, as_frame=True)
    X = data.data
    y = data.target
    return X, y


# 2. Veriyi keşfet
def explore_data(X, y):
    """
    Veri seti hakkında özet bilgiler hesapla.

    Args:
        X: feature DataFrame
        y: hedef Series ('e' / 'p')

    Returns:
        dict: {
            'n_samples': int (örnek sayısı, ~8124),
            'n_features': int (feature sayısı, 22),
            'edible_count': int ('e' sayısı),
            'poisonous_count': int ('p' sayısı),
            'feature_names': list (sütun adları),
        }

    İpucu:
    - (y == 'e').sum() → yenilebilir sayısı
    - (y == 'p').sum() → zehirli sayısı
    - list(X.columns) → feature isimleri
    """
    
    params = y.value_counts()
    
    return {
        "n_samples": len(X),
        "n_features": len(X.columns),
        "edible_count": params["e"],
        "poisonous_count": params["p"],
        "feature_names": list(X.columns)
    }


# 3. Kategorik feature'ları one-hot encode et
def encode_features(X):
    """
    Tüm feature'lar kategorik (harf kodlu) — Decision Tree sayısal girdi ister.
    Hepsini one-hot encode et.

    Args:
        X: kategorik feature DataFrame

    Returns:
        pd.DataFrame: one-hot kodlanmış (her kategori için ayrı 0/1 sütun)

    Not: Decision Tree mesafe tabanlı DEĞİL, bu yüzden scaling GEREKMEZ.
    Sadece kategorikleri sayısala çevirmek yeterli.

    İpucu: pd.get_dummies(X) — tüm kategorik sütunları otomatik one-hot yapar.
    """
    
    return pd.get_dummies(X)


# 4. Hedefi encode et
def encode_target(y):
    """
    Hedef sınıfı sayısala çevir. ZEHİRLİ pozitif sınıf (1) olsun — çünkü
    asıl yakalamak istediğimiz tehlikeli durum zehirli mantar.

    - 'p' (poisonous/zehirli) → 1
    - 'e' (edible/yenilebilir) → 0

    Args:
        y: hedef Series ('e' / 'p')

    Returns:
        np.ndarray: 0/1 dizisi

    İpucu: liste comprehension veya y.map({'p': 1, 'e': 0}).values
    """
    
    return y.map({"e": 0, "p": 1}).to_numpy()


# 5. Train / test böl
def split_data(X, y):
    """
    Veriyi train/test olarak böl.

    - test_size = 0.3
    - stratify = y (sınıf oranı korunsun)
    - random_state = 42

    Args:
        X: encode edilmiş feature DataFrame
        y: encode edilmiş hedef dizisi

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    İpucu: from sklearn.model_selection import train_test_split
    """
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# 6. Decision Tree eğit
def train_tree(X_train, y_train, max_depth=5):
    """
    DecisionTreeClassifier kur ve eğit.

    Args:
        X_train, y_train: training verileri
        max_depth: ağacın maksimum derinliği (default 5)

    Returns:
        DecisionTreeClassifier: eğitilmiş model
            (max_depth=max_depth, random_state=42)

    İpucu:
    - from sklearn.tree import DecisionTreeClassifier
    - model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    - model.fit(X_train, y_train)
    """
    
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X_train, y_train)
    
    return model


# 7. Modeli değerlendir
def evaluate_model(model, X_test, y_test):
    """
    Test seti üzerinde tahmin yap ve metrikleri hesapla.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': np.array (shape (2, 2)),
        }

    İpucu:
    - from sklearn.metrics import accuracy_score, precision_score,
      recall_score, f1_score, confusion_matrix
    - y_pred = model.predict(X_test)
    """
    
    y_pred = model.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


# 8. Feature importance
def get_feature_importances(model, feature_names, top_n=10):
    """
    Modelin feature_importances_ değerlerini feature isimleriyle eşleştir,
    önem sırasına göre (büyükten küçüğe) sırala, ilk top_n'i dön.

    Mantar veri setinde 'odor' (koku) feature'larının baskın çıkması beklenir
    — koku tek başına neredeyse mükemmel ayırır.

    Args:
        model: eğitilmiş Decision Tree
        feature_names: feature isimleri (one-hot sonrası sütun adları)
        top_n: kaç tane dönülecek (default 10)

    Returns:
        list of tuple: [(feature_name, importance), ...] sıralı, top_n adet

    İpucu:
    - model.feature_importances_ → her feature'ın önem skoru
    - zip(feature_names, importances) → eşleştir
    - sorted(..., key=lambda p: p[1], reverse=True)[:top_n]
    """
    
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs[:top_n]


# 9. Ağacı çiz
def plot_decision_tree(model, feature_names, class_names, max_depth=3):
    """
    Decision Tree'yi plot_tree ile görselleştir. Ağacın okunabilir kurallar
    ürettiğini göstermek için ilk birkaç seviyeyi çiz.

    Args:
        model: eğitilmiş Decision Tree
        feature_names: feature isimleri
        class_names: sınıf isimleri (örn. ['yenilebilir', 'zehirli'])
        max_depth: çizimde gösterilecek maksimum derinlik (default 3)

    Returns:
        matplotlib Figure (veya None) — çizimi döndür.

    İpucu:
    - from sklearn.tree import plot_tree
    - import matplotlib.pyplot as plt
    - fig, ax = plt.subplots(figsize=(20, 10))
    - plot_tree(model, feature_names=..., class_names=..., max_depth=max_depth,
                filled=True, rounded=True, ax=ax)
    """
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, max_depth=max_depth, feature_names=feature_names, class_names=class_names)
    
    return fig


# 10. max_depth tuning
def tune_max_depth(X_train, X_test, y_train, y_test, depths):
    """
    Verilen her max_depth değeri için ayrı bir ağaç eğit, test accuracy'sini ölç.

    Küçük derinlik → basit, yorumlanabilir ama belki underfit.
    Büyük derinlik → karmaşık, overfit riski. Mantar verisinde sığ ağaç bile
    çok iyi sonuç verir (problem kolay).

    Args:
        X_train, X_test, y_train, y_test: bölünmüş veri
        depths: denenecek derinlik listesi (örn. [1, 2, 3, 5, 10])

    Returns:
        dict: {depth: test_accuracy, ...}

    İpucu: her depth için train_tree çağır, sonra accuracy_score hesapla.
    """
    
    results = {}
    
    for depth in depths:
        model = train_tree(X_train, y_train, depth)
        score = evaluate_model(model, X_test, y_test)["accuracy"]
        
        results[depth] = score
    
    return results


# 11. Tek mantar tahmini
def predict_mushroom(model, X_row):
    """
    Tek bir mantar için tahmin yap.

    Args:
        model: eğitilmiş model
        X_row: tek satırlık feature (DataFrame, shape (1, n_features))

    Returns:
        dict: {
            'prediction': 0 veya 1 (int),
            'label': 'zehirli' (1) veya 'yenilebilir' (0)
        }

    İpucu:
    - pred = int(model.predict(X_row)[0])
    - label = 'zehirli' if pred == 1 else 'yenilebilir'
    """
    
    y_pred = int(model.predict(X_row)[0])
    
    return {
        "prediction": y_pred,
        "label": "zehirli" if y_pred == 1 else "yenilebilir"
    }


# 12. Logistic Regression ile kıyas
def compare_with_lr(X_train, X_test, y_train, y_test):
    """
    Decision Tree ile Logistic Regression'ın test accuracy'sini kıyasla.
    Mantar verisinde ikisi de çok iyi olur ama Decision Tree YORUMLANABİLİR
    (kuralları okunur), LR ise katsayı ağırlıkları verir.

    Args:
        X_train, X_test, y_train, y_test: bölünmüş veri

    Returns:
        dict: {
            'decision_tree': float (DT test accuracy),
            'logistic_regression': float (LR test accuracy),
        }

    İpucu:
    - DT: train_tree(X_train, y_train, max_depth=5)
    - LR: from sklearn.linear_model import LogisticRegression
          LogisticRegression(max_iter=1000).fit(...)
    - ikisinde de accuracy_score(y_test, model.predict(X_test))
    """
    
    lr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    
    return {
        "decision_tree": evaluate_model(train_tree(X_train, y_train, 5), X_test, y_test)["accuracy"],
        "logistic_regression": lr_model.score(X_test, y_test)
    }


# 13. Mükemmel (saf) yaprak sayısı — bonus
def count_perfect_splits(model):
    """
    Ağaçtaki SAF yaprak sayısını say — gini = 0 olan yapraklar. Saf yaprak,
    o dala düşen tüm örneklerin aynı sınıfta olduğu (kesin karar) demektir.
    Mantar verisinde ağaç çok sayıda saf yaprak üretir (net kurallar).

    Args:
        model: eğitilmiş Decision Tree

    Returns:
        int: gini=0 olan yaprak düğüm sayısı

    İpucu:
    - tree = model.tree_
    - bir düğüm yapraktır: tree.children_left[i] == -1 ve children_right[i] == -1
    - saflık: tree.impurity[i] == 0.0
    - tree.node_count → toplam düğüm sayısı
    """
    
    tree = model.tree_
    count = 0

    for i in range(tree.node_count):
        is_leaf = (
            tree.children_left[i] == -1 and
            tree.children_right[i] == -1
        )

        if is_leaf and tree.impurity[i] == 0.0:
            count += 1

    return count


# 14. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    """
    Yukarıdaki fonksiyonları birleştirip tam akışı çalıştır:
    1. Veri yükle (load_mushroom_data)
    2. Feature'ları one-hot encode et (encode_features)
    3. Hedefi encode et (encode_target)
    4. Train/test böl (split_data)
    5. max_depth=5 ile ağaç eğit (train_tree)
    6. Değerlendir (evaluate_model)
    7. Feature importance al (get_feature_importances) → en önemli feature
    8. Bir örnek mantar için tahmin yap (predict_mushroom)

    Returns:
        dict: {
            'n_samples': int,
            'test_accuracy': float (>0.95 beklenir — mantar kolay),
            'top_feature': str (en önemli feature — muhtemelen 'odor' içerir),
            'top_feature_importance': float,
            'sample_prediction': dict (predict_mushroom çıktısı),
        }

    İpucu: sample = X_test.iloc[[0]] (tek satır DataFrame olarak).
    """
    
    # 1. Veri yükle (load_mushroom_data)
    X, y = load_mushroom_data()
    info = explore_data(X, y)
    
    # 2. Feature'ları one-hot encode et (encode_features)
    X = encode_features(X)
    
    # 3. Hedefi encode et (encode_target)
    y = encode_target(y)
    
    # 4. Train/test böl (split_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 5. max_depth=5 ile ağaç eğit (train_tree)
    model = train_tree(X_train, y_train, 5)
    
    # 6. Değerlendir (evaluate_model)
    metrics = evaluate_model(model, X_test, y_test)
    
    # 7. Feature importance al (get_feature_importances) → en önemli feature
    importances = get_feature_importances(model, list(X.columns), 10)
    
    # 8. Bir örnek mantar için tahmin yap (predict_mushroom)
    sample = X_test.iloc[[0]]
    sample_pred = predict_mushroom(model, sample)
    
    return {
        "n_samples": info["n_samples"],
        "test_accuracy": metrics["accuracy"],
        "top_feature": importances[0][0],
        "top_feature_importance": importances[0][1],
        "sample_prediction": sample_pred,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Örnek sayısı       : {result['n_samples']}")
    print(f"  Test accuracy      : {result['test_accuracy']:.2%}")
    print(f"  En önemli feature  : {result['top_feature']}")
    print(f"  Önem skoru         : {result['top_feature_importance']:.3f}")
    print(f"  Örnek tahmin       : {result['sample_prediction']}")
