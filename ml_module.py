import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metrics(y_true, y_pred, elapsed_ms):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'time_ms': elapsed_ms,
    }


def run_ml_experiments(X_train, X_test, y_train, y_test):
    print("Uruchamianie modeli Machine Learning...")

    ml_results = {}

    # Skalowanie danych (wymagane szczególnie dla Regresji Logistycznej)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Logistic\nRegression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision\nTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random\nForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train_s, y_train)

        t0 = time.time()
        preds = model.predict(X_test_s)
        elapsed = (time.time() - t0) * 1000

        ml_results[name] = metrics(y_test, preds, elapsed)
        ml_results[name]['y_pred'] = list(preds)

    print("Modele Machine Learning przeliczone!\n")
    return ml_results