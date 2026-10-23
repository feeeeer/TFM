"""
03_modeling.py
Comparativa de 3 arquitecturas (objetivo especifico 2 del anteproyecto):
    1. Regresion Logistica (baseline interpretable)
    2. XGBoost (Gradient Boosting)
    3. MLP (red neuronal, sklearn MLPClassifier)

Version "endurecida" (v2):
    - Seleccion de variables con SelectKBest (f_classif), ajustada solo en train.
    - Balanceo de clases con SMOTE aplicado DENTRO de cada fold de CV (imblearn.Pipeline),
      para evitar fuga de informacion entre folds de validacion.
    - Busqueda de hiperparametros con GridSearchCV (Regresion Logistica) y
      RandomizedSearchCV (XGBoost, MLP), optimizando AUC-ROC con 5-fold CV estratificado.
    - Evaluacion final sobre un hold-out test set que el tuning nunca ve.
    - Metricas tecnicas: AUC-ROC, F1, precision, recall + matriz de confusion +
      curva de calibracion (reliability diagram).
    - Metricas de negocio: Curva de Ganancia (Gain Curve) y Lift Chart.

Autor: Fernando Velasco Alonso
"""
import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_xy(path: str):
    df = pd.read_csv(path)
    y = df["Churn"].values
    X = df.drop(columns=["Churn", "customerID"])
    return X, y, df


def select_features(X_train, y_train, X_test, k):
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_cols = X_train.columns[selector.get_support()]
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    return X_train_sel, X_test_sel, selected_cols, scores


def gain_lift_chart(y_true, y_prob, out_dir, model_name):
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False, duplicates="drop")
    total_positives = df["y"].sum()
    n = len(df)

    gain = df.groupby("decile")["y"].sum().cumsum() / total_positives * 100
    pct_customers = (df.groupby("decile").size().cumsum() / n * 100)
    lift = gain / pct_customers

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(pct_customers, gain, marker="o", label=model_name)
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="gray", label="Aleatorio")
    axes[0].set_xlabel("% clientes contactados (ordenados por riesgo)")
    axes[0].set_ylabel("% churners capturados")
    axes[0].set_title(f"Curva de Ganancia - {model_name}")
    axes[0].legend()

    axes[1].bar(range(1, len(lift) + 1), lift.values, color="steelblue")
    axes[1].axhline(1, linestyle="--", color="gray")
    axes[1].set_xlabel("Decil (1=mayor riesgo)")
    axes[1].set_ylabel("Lift")
    axes[1].set_title(f"Lift Chart - {model_name}")

    fig.tight_layout()
    fname = os.path.join(out_dir, f"fig_gain_lift_{model_name.replace(' ', '_')}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def evaluate(model, X_test, y_test, name):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "modelo": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics, y_prob, y_pred


PARAM_GRIDS = {
    "Regresion_Logistica": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(32,), (32, 16)],
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01],
    },
}


def build_base_models():
    return {
        "Regresion_Logistica": LogisticRegression(max_iter=2000, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "MLP": MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
    }


def tune_model(name, base_model, X_train_sel, y_train, cv, n_iter_random=12):
    """SMOTE dentro del pipeline -> se aplica solo en el fold de entrenamiento de cada CV split.
    n_jobs=1 deliberadamente: en algunos entornos en contenedor el backend de multiproceso
    de joblib (loky) muere en silencio con n_jobs=-1. Mas lento pero fiable."""
    pipe = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("clf", base_model),
    ])
    grid = PARAM_GRIDS[name]

    if name == "Regresion_Logistica":
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1)
    else:
        search = RandomizedSearchCV(
            pipe, grid, n_iter=n_iter_random, scoring="roc_auc", cv=cv,
            random_state=42, n_jobs=1
        )
    search.fit(X_train_sel, y_train)
    return search


def plot_confusion_matrices(models_pred, y_test, out_dir):
    fig, axes = plt.subplots(1, len(models_pred), figsize=(5 * len(models_pred), 4.5))
    if len(models_pred) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, models_pred.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.suptitle("Matrices de Confusion (umbral = 0.5)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig12_matrices_confusion.png"), dpi=150)
    plt.close(fig)


def plot_calibration(models_prob, y_test, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibracion perfecta")
    for name, y_prob in models_prob.items():
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Probabilidad predicha media (por bin)")
    ax.set_ylabel("Frecuencia observada de churn")
    ax.set_title("Curva de Calibracion (Reliability Diagram)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig13_calibracion.png"), dpi=150)
    plt.close(fig)


def prep(args):
    """Preparacion determinista y barata (split + scale + SelectKBest). Se recalcula en
    cada invocacion (--only) porque con random_state fijo siempre da el mismo resultado,
    lo que permite lanzar el tuning de cada modelo en llamadas de shell separadas sin
    perder consistencia entre ellas (el sandbox no mantiene procesos en segundo plano
    entre llamadas)."""
    X, y, df_full = load_xy(args.input)

    if len(df_full) < 1000:
        print("=" * 70)
        print(f"AVISO: dataset con solo {len(df_full)} filas (muestra parcial de validacion).")
        print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_train_sel, X_test_sel, selected_cols, scores = select_features(
        X_train_scaled, y_train, X_test_scaled, k=args.k_features
    )
    return X_train_sel, X_test_sel, y_train, y_test, selected_cols, scores, scaler


def run_one(name, args):
    X_train_sel, X_test_sel, y_train, y_test, selected_cols, scores, scaler = prep(args)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    base_model = build_base_models()[name]

    print(f">>> Tuning {name} con {args.cv_folds}-fold CV (SMOTE dentro de cada fold)...")
    search = tune_model(name, base_model, X_train_sel, y_train, cv)
    best_model = search.best_estimator_
    print(f"Mejor AUC-ROC (CV, {args.cv_folds}-fold): {search.best_score_:.4f}")
    print(f"Mejores hiperparametros: {search.best_params_}")

    metrics, y_prob, y_pred = evaluate(best_model, X_test_sel, y_test, name)
    print(f"Metricas hold-out test: {metrics}")

    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(args.models_dir, f"{name}.joblib"))
    np.save(os.path.join(args.models_dir, f"y_prob_{name}.npy"), y_prob)
    np.save(os.path.join(args.models_dir, f"y_pred_{name}.npy"), y_pred)
    np.save(os.path.join(args.models_dir, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(args.models_dir, "scaler.joblib"))
    with open(os.path.join(args.models_dir, "selected_features.json"), "w") as f:
        json.dump(list(selected_cols), f, indent=2, ensure_ascii=False)

    os.makedirs(args.output, exist_ok=True)
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    gain_lift_chart(y_test, y_prob, fig_dir, name)

    # Acumular metricas/tuning en CSV (append si ya existe de una ejecucion previa de otro modelo)
    metrics_path = os.path.join(args.output, "model_comparison.csv")
    tuning_path = os.path.join(args.output, "cv_hyperparameter_tuning.csv")

    metrics_df = pd.DataFrame([metrics])
    if os.path.exists(metrics_path):
        prev = pd.read_csv(metrics_path)
        prev = prev[prev["modelo"] != name]
        metrics_df = pd.concat([prev, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)

    tuning_row = pd.DataFrame([{
        "modelo": name,
        "mejor_auc_cv_mean": round(search.best_score_, 4),
        "mejores_hiperparametros": json.dumps(search.best_params_, ensure_ascii=False),
    }])
    if os.path.exists(tuning_path):
        prev = pd.read_csv(tuning_path)
        prev = prev[prev["modelo"] != name]
        tuning_row = pd.concat([prev, tuning_row], ignore_index=True)
    tuning_row.to_csv(tuning_path, index=False)

    print("\nTop variables segun SelectKBest (f_classif):")
    print(scores.head(args.k_features).to_string())


def combine(args):
    """Combina los artefactos guardados por cada run_one() en las graficas comparativas
    (ROC, matrices de confusion, calibracion). Requiere haber ejecutado --only para los
    3 modelos primero."""
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    names = list(build_base_models().keys())

    y_test = np.load(os.path.join(args.models_dir, "y_test.npy"))
    models_pred, models_prob, fitted_models = {}, {}, {}
    for name in names:
        models_prob[name] = np.load(os.path.join(args.models_dir, f"y_prob_{name}.npy"))
        models_pred[name] = np.load(os.path.join(args.models_dir, f"y_pred_{name}.npy"))
        fitted_models[name] = joblib.load(os.path.join(args.models_dir, f"{name}.joblib"))

    plot_confusion_matrices(models_pred, y_test, fig_dir)
    plot_calibration(models_prob, y_test, fig_dir)

    # Curva ROC comparativa (usando las probabilidades ya guardadas por cada run_one)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in models_prob.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curvas ROC - Comparativa de modelos (con tuning)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig05_roc_comparativa.png"), dpi=150)
    plt.close(fig)

    results_df = pd.read_csv(os.path.join(args.output, "model_comparison.csv"))
    cv_summary_df = pd.read_csv(os.path.join(args.output, "cv_hyperparameter_tuning.csv"))
    print("\n=== Resumen CV (tuning) ===")
    print(cv_summary_df.to_string(index=False))
    print("\n=== Resumen hold-out test ===")
    print(results_df.to_string(index=False))
    print(f"\nModelos guardados en: {args.models_dir}")
    print(f"Metricas guardadas en: {os.path.join(args.output, 'model_comparison.csv')}")
    print(f"Tuning guardado en: {os.path.join(args.output, 'cv_hyperparameter_tuning.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/processed/telco_processed.csv")
    parser.add_argument("--output", default="../reports")
    parser.add_argument("--models_dir", default="../models")
    parser.add_argument("--k_features", type=int, default=15)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--only", choices=["Regresion_Logistica", "XGBoost", "MLP"],
                         help="Ajusta y evalua solo este modelo (para trocear la ejecucion)")
    parser.add_argument("--combine", action="store_true",
                         help="Genera las graficas comparativas a partir de los artefactos ya guardados")
    args = parser.parse_args()

    if args.combine:
        combine(args)
    elif args.only:
        run_one(args.only, args)
    else:
        for name in build_base_models().keys():
            run_one(name, args)
        combine(args)


if __name__ == "__main__":
    main()
