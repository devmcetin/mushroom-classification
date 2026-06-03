import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_mushroom_data, explore_data, encode_features, encode_target,
    split_data, train_tree, evaluate_model, get_feature_importances,
    plot_decision_tree, tune_max_depth, predict_mushroom, compare_with_lr,
    count_perfect_splits, run_pipeline,
)


# Yardımcı — encode edilmiş veriyi bir kez hazırla (cache)
_CACHE = {}


def _get_encoded():
    if 'data' not in _CACHE:
        X, y = load_mushroom_data()
        X_enc = encode_features(X)
        y_enc = encode_target(y)
        _CACHE['data'] = (X, y, X_enc, y_enc)
    return _CACHE['data']


def _build_trained(max_depth=5):
    _, _, X_enc, y_enc = _get_encoded()
    X_train, X_test, y_train, y_test = split_data(X_enc, y_enc)
    model = train_tree(X_train, y_train, max_depth=max_depth)
    return model, X_train, X_test, y_train, y_test, X_enc


# 1. load_mushroom_data
def test_load_mushroom_data_types():
    X, y = load_mushroom_data()
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 8124


def test_load_mushroom_data_features():
    X, y = load_mushroom_data()
    assert X.shape[1] == 22
    # hedef sadece 'e' ve 'p' içerir
    assert set(pd.Series(y).unique()) == {'e', 'p'}


# 2. explore_data
def test_explore_data_structure():
    X, y = load_mushroom_data()
    info = explore_data(X, y)
    assert info['n_samples'] == 8124
    assert info['n_features'] == 22
    # dengeli — ~%48 zehirli
    assert info['edible_count'] == 4208
    assert info['poisonous_count'] == 3916
    assert len(info['feature_names']) == 22


# 3. encode_features
def test_encode_features_one_hot():
    X, y = load_mushroom_data()
    X_enc = encode_features(X)
    # one-hot sonrası feature sayısı artmalı (her kategori ayrı sütun)
    assert X_enc.shape[1] > 22
    assert X_enc.shape[0] == 8124
    # tüm değerler 0/1 (one-hot)
    assert set(np.unique(X_enc.values.astype(int))).issubset({0, 1})


# 4. encode_target
def test_encode_target_poisonous_is_one():
    X, y = load_mushroom_data()
    y_enc = encode_target(y)
    y_enc = np.asarray(y_enc)
    assert set(np.unique(y_enc)) == {0, 1}
    # 'p' (zehirli) → 1 olmalı
    assert int(y_enc.sum()) == 3916


# 5. split_data
def test_split_data_sizes():
    _, _, X_enc, y_enc = _get_encoded()
    X_train, X_test, y_train, y_test = split_data(X_enc, y_enc)
    # test_size=0.3 → ~2438 test
    assert X_test.shape[0] == 2438
    assert X_train.shape[0] == 5686


def test_split_data_stratified():
    _, _, X_enc, y_enc = _get_encoded()
    _, _, _, y_test = split_data(X_enc, y_enc)
    rate = np.asarray(y_test).mean()
    # ~%48 zehirli oranı korunmalı
    assert 0.46 < rate < 0.50


# 6. train_tree
def test_train_tree_params():
    from sklearn.tree import DecisionTreeClassifier
    model, _, _, _, _, _ = _build_trained(max_depth=4)
    assert isinstance(model, DecisionTreeClassifier)
    assert model.max_depth == 4


# 7. evaluate_model
def test_evaluate_model_high_accuracy():
    model, _, X_test, _, y_test, _ = _build_trained()
    result = evaluate_model(model, X_test, y_test)
    for key in ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']:
        assert key in result
    assert result['confusion_matrix'].shape == (2, 2)
    # mantar kolay — yüksek accuracy beklenir
    assert result['accuracy'] > 0.95


# 8. get_feature_importances
def test_get_feature_importances_structure():
    model, _, _, _, _, X_enc = _build_trained()
    imps = get_feature_importances(model, X_enc.columns, top_n=5)
    assert len(imps) == 5
    # (feature, importance) tuple
    assert all(len(p) == 2 for p in imps)
    # büyükten küçüğe sıralı
    vals = [p[1] for p in imps]
    assert vals == sorted(vals, reverse=True)


def test_get_feature_importances_odor_dominant():
    model, _, _, _, _, X_enc = _build_trained()
    imps = get_feature_importances(model, X_enc.columns, top_n=3)
    top_feature = imps[0][0]
    # ünlü gerçek: koku (odor) baskın ayırıcı
    assert 'odor' in top_feature.lower()


# 9. plot_decision_tree
def test_plot_decision_tree_runs():
    model, _, _, _, _, X_enc = _build_trained()
    # hata fırlatmadan çalışmalı
    fig = plot_decision_tree(
        model, X_enc.columns, ['yenilebilir', 'zehirli'], max_depth=2
    )
    # None veya bir figure dönebilir — fırlatmaması yeterli
    assert fig is None or fig is not None


# 10. tune_max_depth
def test_tune_max_depth():
    _, _, X_enc, y_enc = _get_encoded()
    X_train, X_test, y_train, y_test = split_data(X_enc, y_enc)
    results = tune_max_depth(X_train, X_test, y_train, y_test, depths=[1, 2, 5])
    assert set(results.keys()) == {1, 2, 5}
    # derinlik arttıkça accuracy artar (veya eşit kalır)
    assert results[5] >= results[1]
    assert all(0 < v <= 1 for v in results.values())


# 11. predict_mushroom
def test_predict_mushroom_structure():
    model, _, X_test, _, _, _ = _build_trained()
    row = X_test.iloc[[0]]
    result = predict_mushroom(model, row)
    assert 'prediction' in result
    assert 'label' in result
    assert result['prediction'] in [0, 1]
    assert result['label'] in ['zehirli', 'yenilebilir']


def test_predict_mushroom_label_matches():
    model, _, X_test, _, _, _ = _build_trained()
    row = X_test.iloc[[0]]
    result = predict_mushroom(model, row)
    expected = 'zehirli' if result['prediction'] == 1 else 'yenilebilir'
    assert result['label'] == expected


# 12. compare_with_lr
def test_compare_with_lr():
    _, _, X_enc, y_enc = _get_encoded()
    X_train, X_test, y_train, y_test = split_data(X_enc, y_enc)
    result = compare_with_lr(X_train, X_test, y_train, y_test)
    assert 'decision_tree' in result
    assert 'logistic_regression' in result
    # ikisi de yüksek olmalı
    assert result['decision_tree'] > 0.9
    assert result['logistic_regression'] > 0.9


# 13. count_perfect_splits
def test_count_perfect_splits():
    model, _, _, _, _, _ = _build_trained()
    count = count_perfect_splits(model)
    assert isinstance(count, int)
    # mantar ağacında en az birkaç saf yaprak olur
    assert count >= 1


# 14. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    for key in ['n_samples', 'test_accuracy', 'top_feature',
                'top_feature_importance', 'sample_prediction']:
        assert key in result
    assert result['n_samples'] == 8124
    # mantar kolay
    assert result['test_accuracy'] > 0.95
    # en önemli feature odor olmalı
    assert 'odor' in result['top_feature'].lower()


def test_run_pipeline_sample_prediction():
    result = run_pipeline()
    sp = result['sample_prediction']
    assert sp['prediction'] in [0, 1]
    assert sp['label'] in ['zehirli', 'yenilebilir']


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
