"""
Genera notebooks/TFM_Churn_CLV_Pipeline.ipynb a partir de bloques de codigo,
uniendo los scripts de src/ en un unico notebook ejecutable celda a celda.
No forma parte del pipeline de analisis: es solo la herramienta de construccion
del notebook. Ejecutar una vez desde la raiz del proyecto: python3 build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md("""# TFM — Predicción de Churn y CLV mediante Modelos de Supervivencia y Deep Learning en Telecomunicaciones

**Autor:** Fernando Velasco Alonso · Máster en Análisis de Datos Masivos (Big Data) · Universidad Europea de Madrid
**Director:** Jesús Gil Ruiz

Este notebook une en un único flujo ejecutable los bloques del pipeline:

1. Análisis Exploratorio de Datos (EDA)
2. Preprocesado y Feature Engineering
3. Modelos de clasificación (Regresión Logística, XGBoost, MLP) con SelectKBest + SMOTE
4. Modelo de supervivencia (Cox Proportional Hazards) + estimación de CLV dinámico
5. Explicabilidad (SHAP)
6. Supervivencia ampliada: Cox completo, diagnóstico de riesgos proporcionales y **DeepSurv**
7. Diagnósticos ampliados: EDA, curvas precisión-exhaustividad, umbral óptimo, equidad por subgrupos y SHAP dependence
8. Calibración de probabilidades, métricas train/validación/test y simulación económica sobre el conjunto de test

**Antes de ejecutar:** coloca el CSV completo de Kaggle (`WA_Fn-UseC_-Telco-Customer-Churn.csv`,
7.043 filas) en `../data/raw/` y ajusta `INPUT_CSV` en la primera celda de código si el nombre
del archivo es distinto.
""")

# ---------------------------------------------------------------------------
code("""# --- Configuración general ---
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

INPUT_CSV = "../data/raw/Telco-Customer-Churn_FULL.csv"  # <-- cambia esto si tu archivo se llama distinto
REPORTS_DIR = "../reports"
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = "../models"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

df_raw = pd.read_csv(INPUT_CSV)
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
print(df_raw.shape)
df_raw.head()
""")

# ---------------------------------------------------------------------------
md("## 1. Análisis Exploratorio de Datos (EDA)\n\nObjetivo específico 1 del anteproyecto: identificar patrones de comportamiento vinculados al abandono.")

code("""print(f"Filas: {df_raw.shape[0]}  |  Columnas: {df_raw.shape[1]}")
print("\\n--- Nulos ---")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print("\\n--- Duplicados customerID ---", df_raw["customerID"].duplicated().sum())
print("\\n--- Balance de clases ---")
print(df_raw["Churn"].value_counts())
print(df_raw["Churn"].value_counts(normalize=True) * 100)
df_raw.describe()
""")

code("""fig, ax = plt.subplots(figsize=(5, 4))
order = df_raw["Churn"].value_counts().index
sns.countplot(data=df_raw, x="Churn", order=order, ax=ax, palette="Set2")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
ax.set_title("Distribución de Churn (desbalanceo de clases)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_churn_balance.png"), dpi=150)
plt.show()
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.boxplot(data=df_raw, x="Churn", y=col, ax=ax, palette="Set3")
    ax.set_title(f"{col} vs Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig02_numeric_vs_churn.png"), dpi=150)
plt.show()
""")

code("""cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
            "SeniorCitizen", "Partner", "Dependents", "TechSupport", "OnlineSecurity"]
df_plot = df_raw.copy()
df_plot["ChurnFlag"] = (df_plot["Churn"] == "Yes").astype(int)

ncols = 3
nrows = int(np.ceil(len(cat_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)
for i, col in enumerate(cat_cols):
    rate = df_plot.groupby(col)["ChurnFlag"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, ax=axes[i], palette="rocket")
    axes[i].set_title(f"Tasa de churn por {col}")
    axes[i].tick_params(axis="x", rotation=30)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_churn_rate_categoricas.png"), dpi=150)
plt.show()
""")

code("""df_num = df_raw.copy()
df_num["ChurnFlag"] = (df_num["Churn"] == "Yes").astype(int)
df_num["SeniorCitizen"] = df_num["SeniorCitizen"].astype(int)
corr = df_num[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "ChurnFlag"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlación entre variables numéricas y Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig04_correlacion.png"), dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("""## 2. Preprocesado y Feature Engineering

- `TotalCharges` vacío (tenure=0) → se imputa a 0.
- Categorías "No internet/phone service" → colapsadas a "No" (evita multicolinealidad).
- Nuevas variables: `AvgMonthlySpend`, `NumServices`, `TenureGroup`, `HighRiskContract`.
- Codificación: Label Encoding (binarias) + One-Hot (Contract, InternetService, PaymentMethod, TenureGroup).
""")

code("""from sklearn.preprocessing import LabelEncoder

def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    return df

def feature_engineering(df):
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    yes_services = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = (df[yes_services] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int) \\
        + (df["PhoneService"] == "Yes").astype(int)
    bins = [0, 12, 24, 48, 73]
    labels = ["0-12m", "12-24m", "24-48m", "48-72m"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    df["HighRiskContract"] = (df["Contract"] == "Month-to-month").astype(int)
    return df

def encode(df):
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod", "TenureGroup"], drop_first=True)
    return df

df_proc = encode(feature_engineering(clean(df_raw)))
df_proc.to_csv(os.path.join("../data/processed", "telco_processed.csv"), index=False)
print(df_proc.shape)
df_proc.head()
""")

# ---------------------------------------------------------------------------
md("""## 3. Modelos de clasificación: Regresión Logística, XGBoost, MLP

Objetivo específico 2 del anteproyecto. Incluye selección de variables (SelectKBest),
balanceo de clases (SMOTE) **dentro de cada fold de CV** (evita fuga de información),
y ajuste de hiperparámetros con `GridSearchCV`/`RandomizedSearchCV` sobre 5-fold CV
estratificado, optimizando AUC-ROC. La evaluación final se hace sobre un hold-out test
que el tuning nunca ve.

Nota metodológica: `StandardScaler` y `SelectKBest` se ajustan **una sola vez** sobre `X_train`, fuera de los folds de CV
(solo SMOTE y el clasificador están dentro del `Pipeline`). Por eso las métricas de CV pueden ser algo optimistas;
las de test no se ven afectadas.
""")

code("""import joblib
import json
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

y = df_proc["Churn"].values
X = df_proc.drop(columns=["Churn", "customerID"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

K_FEATURES = 15
selector = SelectKBest(score_func=f_classif, k=min(K_FEATURES, X_train.shape[1]))
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)
selected_cols = X_train.columns[selector.get_support()]
scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
print("Top variables (SelectKBest):")
print(scores.head(K_FEATURES))
""")

code("""CV_FOLDS = 5
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

base_models = {
    "Regresion_Logistica": LogisticRegression(max_iter=2000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "MLP": MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
}

param_grids = {
    "Regresion_Logistica": {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]},
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.8, 1.0], "clf__colsample_bytree": [0.8, 1.0],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(32,), (32, 16)], "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01],
    },
}

# n_jobs=1 deliberado: evita crashes silenciosos del backend de multiproceso de joblib
# en algunos entornos en contenedor. Es mas lento pero fiable.
all_metrics, cv_summary_rows = [], []
fitted_models, models_pred, models_prob = {}, {}, {}

for name, base_model in base_models.items():
    print(f">>> Tuning {name}...")
    pipe = ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", base_model)])
    if name == "Regresion_Logistica":
        search = GridSearchCV(pipe, param_grids[name], scoring="roc_auc", cv=cv, n_jobs=1)
    else:
        search = RandomizedSearchCV(pipe, param_grids[name], n_iter=12, scoring="roc_auc",
                                     cv=cv, random_state=42, n_jobs=1)
    search.fit(X_train_sel, y_train)
    best_model = search.best_estimator_
    fitted_models[name] = best_model
    cv_summary_rows.append({"modelo": name, "mejor_auc_cv_mean": round(search.best_score_, 4),
                             "mejores_hiperparametros": json.dumps(search.best_params_, ensure_ascii=False)})
    print(f"  Mejor AUC-ROC (CV {CV_FOLDS}-fold): {search.best_score_:.4f} | params: {search.best_params_}")

    y_prob = best_model.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "modelo": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
    }
    all_metrics.append(metrics)
    models_pred[name], models_prob[name] = y_pred, y_prob
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    print(f"  Metricas hold-out test: {metrics}")

results_df = pd.DataFrame(all_metrics)
results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
pd.DataFrame(cv_summary_rows).to_csv(os.path.join(REPORTS_DIR, "cv_hyperparameter_tuning.csv"), index=False)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
with open(os.path.join(MODELS_DIR, "selected_features.json"), "w") as f:
    json.dump(list(selected_cols), f, indent=2, ensure_ascii=False)
results_df
""")

code("""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, y_pred) in zip(axes, models_pred.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
fig.suptitle("Matrices de Confusión (umbral = 0.5)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig12_matrices_confusion.png"), dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
for name, y_prob in models_prob.items():
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=name)
ax.set_xlabel("Probabilidad predicha media (por bin)"); ax.set_ylabel("Frecuencia observada de churn")
ax.set_title("Curva de Calibración (Reliability Diagram)"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig13_calibracion.png"), dpi=150)
plt.show()
""")

code("""# Curva ROC comparativa + curva de ganancia / lift para XGBoost
fig, ax = plt.subplots(figsize=(6, 5))
for name, y_prob in models_prob.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("Tasa de Falsos Positivos"); ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Curvas ROC - Comparativa de modelos (con tuning)"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig05_roc_comparativa.png"), dpi=150)
plt.show()

def gain_lift_chart(y_true, y_prob, model_name):
    d = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, 10, labels=False, duplicates="drop")
    total_pos = d["y"].sum(); n = len(d)
    gain = d.groupby("decile")["y"].sum().cumsum() / total_pos * 100
    pct_customers = d.groupby("decile").size().cumsum() / n * 100
    lift = gain / pct_customers
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(pct_customers, gain, marker="o", label=model_name)
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="gray", label="Aleatorio")
    axes[0].set_xlabel("% clientes contactados"); axes[0].set_ylabel("% churners capturados")
    axes[0].set_title(f"Curva de Ganancia - {model_name}"); axes[0].legend()
    axes[1].bar(range(1, len(lift) + 1), lift.values, color="steelblue")
    axes[1].axhline(1, linestyle="--", color="gray")
    axes[1].set_xlabel("Decil (1=mayor riesgo)"); axes[1].set_ylabel("Lift")
    axes[1].set_title(f"Lift Chart - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_gain_lift_{model_name}.png"), dpi=150)
    plt.show()

for name, y_prob in models_prob.items():
    gain_lift_chart(y_test, y_prob, name)
""")

# ---------------------------------------------------------------------------
md("""## 4. Modelo de supervivencia (Cox PH) + CLV dinámico

Este es el bloque que justifica el título del TFM. `duration = tenure`, `event = Churn`
(1 = abandono observado, 0 = censurado / sigue activo). Es la forma estándar de tratar
este dataset como snapshot de supervivencia. Limitación: no hay fecha de alta real ni seguimiento longitudinal.
""")

code("""from lifelines import CoxPHFitter, KaplanMeierFitter

df_surv = df_raw.copy()
df_surv["TotalCharges"] = pd.to_numeric(df_surv["TotalCharges"], errors="coerce").fillna(0.0)
df_surv["event"] = (df_surv["Churn"] == "Yes").astype(int)
df_surv["duration"] = df_surv["tenure"].clip(lower=1)

df_cox = df_surv[["duration", "event", "MonthlyCharges", "SeniorCitizen"]].copy()
df_cox["Contract_OneYear"] = (df_surv["Contract"] == "One year").astype(int)
df_cox["Contract_TwoYear"] = (df_surv["Contract"] == "Two year").astype(int)
df_cox["Fiber"] = (df_surv["InternetService"] == "Fiber optic").astype(int)
df_cox["Paperless"] = (df_surv["PaperlessBilling"] == "Yes").astype(int)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(df_cox, duration_col="duration", event_col="event")
summary = cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
summary.to_csv(os.path.join(REPORTS_DIR, "cox_hazard_ratios.csv"))
print(summary)

fig, ax = plt.subplots(figsize=(6, 4))
cph.plot(ax=ax)
ax.set_title("Coeficientes Cox (log hazard ratio) con IC 95%")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig06_cox_coeficientes.png"), dpi=150)
plt.show()
""")

code("""fig, ax = plt.subplots(figsize=(7, 5))
kmf = KaplanMeierFitter()
for contract_type in df_surv["Contract"].unique():
    mask = df_surv["Contract"] == contract_type
    kmf.fit(df_surv.loc[mask, "duration"], df_surv.loc[mask, "event"], label=contract_type)
    kmf.plot_survival_function(ax=ax)
ax.set_title("Curvas de Supervivencia Kaplan-Meier por tipo de contrato")
ax.set_xlabel("Meses (tenure)"); ax.set_ylabel("Probabilidad de retención S(t)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig07_kaplan_meier_contrato.png"), dpi=150)
plt.show()
""")

code("""HORIZON = 24  # meses de horizonte de negocio para el CLV
surv_funcs = cph.predict_survival_function(df_cox, times=np.arange(1, HORIZON + 1))
rmst = surv_funcs.sum(axis=0)

clv_df = pd.DataFrame({
    "customerID": df_surv["customerID"].values,
    "MonthlyCharges": df_surv["MonthlyCharges"].values,
    "tenure_actual": df_surv["tenure"].values,
    "Churn_actual": df_surv["Churn"].values,
    f"RMST_{HORIZON}m": rmst.values,
})
clv_df[f"CLV_{HORIZON}m"] = clv_df["MonthlyCharges"] * clv_df[f"RMST_{HORIZON}m"]
clv_df = clv_df.sort_values(f"CLV_{HORIZON}m", ascending=False)
clv_df.to_csv(os.path.join(REPORTS_DIR, "clv_estimado.csv"), index=False)

print(f"CLV medio de la cartera: {clv_df[f'CLV_{HORIZON}m'].mean():.2f} EUR")
print(f"CLV total de la cartera: {clv_df[f'CLV_{HORIZON}m'].sum():.2f} EUR")
clv_df.head(10)
""")

code("""risk_proxy = 1 - rmst.values / HORIZON
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(risk_proxy, clv_df[f"CLV_{HORIZON}m"], c=(clv_df["Churn_actual"] == "Yes"), cmap="coolwarm", alpha=0.5)
ax.set_xlabel("Riesgo relativo de churn (proxy 1 - RMST/horizonte)")
ax.set_ylabel(f"CLV estimado a {HORIZON} meses (EUR)")
ax.set_title("Matriz Riesgo x Valor para priorizar retención")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig08_matriz_riesgo_valor.png"), dpi=150)
plt.show()
""")

# ---------------------------------------------------------------------------
md("## 5. Explicabilidad (SHAP)\n\nObjetivo específico 6 del anteproyecto: explicabilidad global y local con SHAP sobre el modelo XGBoost.")

code("""import shap

X_sel_full = X.copy()
X_scaled_full = pd.DataFrame(scaler.transform(X_sel_full), columns=X_sel_full.columns)[selected_cols]
X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(
    X_scaled_full, y, test_size=0.25, random_state=42, stratify=y
)

# fitted_models["XGBoost"] es un imblearn.Pipeline (SMOTE + XGBClassifier); SHAP
# TreeExplainer necesita el estimador de arbol puro, no el pipeline completo.
xgb_model = fitted_models["XGBoost"].named_steps["clf"]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test_shap)

shap.summary_plot(shap_values, X_test_shap, show=False)
plt.title("SHAP Summary Plot - Importancia global de variables (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig09_shap_summary.png"), dpi=150, bbox_inches="tight")
plt.show()

mean_abs_shap = pd.DataFrame({
    "feature": X_test_shap.columns,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
mean_abs_shap.to_csv(os.path.join(REPORTS_DIR, "shap_importancia_global.csv"), index=False)
mean_abs_shap.head(10)
""")

code("""probs = fitted_models["XGBoost"].predict_proba(X_test_shap)[:, 1]
idx_riesgo = probs.argmax()
shap.plots.waterfall(shap_values[idx_riesgo], show=False)
plt.title(f"Explicación local SHAP - cliente de mayor riesgo (p={probs[idx_riesgo]:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_shap_waterfall_cliente_riesgo.png"), dpi=150, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
md("""## 6. Supervivencia ampliada: Cox completo, riesgos proporcionales y DeepSurv

Objetivo de la ampliación: comparar sobre la **misma partición train/test** tres modelos de supervivencia
(Cox pequeño de 6 covariables, Cox completo de 21 covariables y DeepSurv), contrastar el supuesto de riesgos proporcionales (test de Schoenfeld),
ajustar un Cox estratificado por contrato, evaluar la calibración de S(24 meses) y recalcular el CLV con cada modelo.

DeepSurv (Katzman et al., 2018) está implementado en NumPy en `src/deepsurv_numpy.py`: un perceptrón multicapa que predice el log-riesgo η(x)
y se entrena minimizando la verosimilitud parcial de Cox (Breslow) con penalización L2. La función `_self_check()` verifica el gradiente
analítico frente a diferencias finitas.
""")

code("""import sys, inspect, importlib
sys.path.append("../src")
import deepsurv_numpy
deepsurv_numpy._self_check()          # gradiente analítico vs diferencias finitas + aprende señal
print(inspect.getsource(deepsurv_numpy.DeepSurv))
""")

code("""import matplotlib
sv = importlib.import_module("07_supervivencia_ampliada")
sv.main()   # Cox pequeño/completo/estratificado, test PH, CV de DeepSurv, calibración y CLV (≈ 30 s)
matplotlib.use("module://matplotlib_inline.backend_inline", force=True)  # el script usa Agg: se restaura el backend del notebook
""")

code("""import json
from IPython.display import Image, display
res7 = json.load(open(os.path.join(REPORTS_DIR, "survival_ampliada_resultados.json")))
print("C-index en test:", {k: round(v, 4) for k, v in res7["cindex_test"].items()})
print("IC 95% bootstrap:", {k: [round(x, 4) for x in v] for k, v in res7["cindex_test_ic95"].items()})
print("Diferencia DeepSurv - Cox completo (IC 95%):", [round(x, 4) for x in res7["diff_DeepSurv_menos_CoxCompleto_ic95"]])
print("Error abs. medio de calibración S(24):", {k: round(v, 4) for k, v in res7["calibracion_error_abs_medio_S24"].items()})
print("Violaciones de riesgos proporcionales: pequeño =", res7["ph_violaciones_cox_pequeno"], "| completo =", res7["ph_violaciones_cox_completo"])
display(pd.read_csv(os.path.join(REPORTS_DIR, "survival_deepsurv_cv.csv")).round(4))
for f in ["fig20_cox_completo_forest", "fig21_schoenfeld", "fig22_km_loglog", "fig23_deepsurv_loss", "fig24_calibracion_supervivencia", "fig25_comparacion_clv"]:
    display(Image(os.path.join(FIG_DIR, f + ".png"), width=600))
""")

md("""## 7. Diagnósticos ampliados

EDA ampliado (antigüedad, servicios, contrato, pago), curvas de precisión-exhaustividad,
umbral de decisión óptimo según el coste de contacto (con las etiquetas reales del test), equidad por subgrupos (género, edad, pareja, personas a cargo y contrato)
y gráficos de dependencia SHAP.
""")

code("""dg = importlib.import_module("08_diagnosticos_ampliados")
dg.main()
matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
""")

code("""res8 = json.load(open(os.path.join(REPORTS_DIR, "diagnosticos_resultados.json")))
print("Average precision (test):", res8["average_precision"], "| prevalencia:", round(res8["prevalencia_test"], 4))
print("Cambio de signo del efecto SHAP de tenure (meses):", res8["shap_tenure_cambio_signo_meses"])
display(pd.read_csv(os.path.join(REPORTS_DIR, "diagnosticos_subgrupos.csv")))
for f in ["fig15_eda_tenure_charges", "fig16_churn_tenure_servicios", "fig17_boxplot_cargos_contrato", "fig18_heatmap_internet_contrato",
          "fig19_heatmap_pago_papel", "fig26_pr_curves", "fig27_umbral_optimo", "fig28_subgrupos", "fig29_shap_dependence_tenure", "fig30_shap_dependence_charges"]:
    display(Image(os.path.join(FIG_DIR, f + ".png"), width=600))
""")

# ---------------------------------------------------------------------------
md("""## 8. Calibración, métricas train/validación/test y simulación económica en test

Ejecuta `src/09_calibracion_y_valor_esperado.py`, que (i) calcula AUC, precisión, recall, F1 y Brier en entrenamiento, validación cruzada y test,
(ii) comprueba la calibración de las probabilidades (SMOTE las sobrestima) y las recalibra con regresión isotónica y sigmoide,
(iii) estima el CLV de los clientes de test con un Cox ajustado **solo con el entrenamiento**, y (iv) simula el ahorro neto de la campaña de retención
**solo sobre el conjunto de test y con las etiquetas reales**, frente a un contacto aleatorio (esperanza exacta), con IC bootstrap y análisis de sensibilidad al coste.
""")

code("""m9 = importlib.import_module("09_calibracion_y_valor_esperado")
m9.main()
matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
""")

code("""res9 = json.load(open(os.path.join(REPORTS_DIR, "calibracion_valor_esperado.json")))
display(pd.read_csv(os.path.join(REPORTS_DIR, "metricas_train_val_test.csv")))
display(pd.read_csv(os.path.join(REPORTS_DIR, "calibracion_probabilidades.csv")))
display(pd.read_csv(os.path.join(REPORTS_DIR, "economia_test_por_presupuesto.csv")))
display(pd.read_csv(os.path.join(REPORTS_DIR, "economia_test_sensibilidad_coste.csv")))
display(pd.read_csv(os.path.join(REPORTS_DIR, "segmentos_riesgo_valor.csv")))
e20 = res9["economia_20pct"]
print(f"Presupuesto 20% ({e20['contactados']} clientes): modelo {e20['modelo']:,.0f} EUR vs aleatorio {e20['aleatorio']:,.0f} EUR")
for f in ["fig31_calibracion_antes_despues", "fig32_ahorro_test_presupuesto", "fig33_distribucion_clv", "fig34_roc_pr_conjuntas", "fig35_ganancia_lift_conjuntas"]:
    display(Image(os.path.join(FIG_DIR, f + ".png"), width=650))
""")

md("""## Líneas futuras

- Modelos de supervivencia que relajen el supuesto de riesgos proporcionales (tiempo discreto, DeepHit) y búsqueda de hiperparámetros más amplia para DeepSurv.
- CLV condicionado a la antigüedad actual, calculado sobre margen y con tasa de descuento.
- Un **Causal Forest** (`econml`) o un modelo de uplift sobre un experimento A/B real o simulado para estimar la tasa de éxito de retención.
""")

nb["cells"] = cells

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/TFM_Churn_CLV_Pipeline.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook generado en notebooks/TFM_Churn_CLV_Pipeline.ipynb")
