"""
05_explainability.py
Explicabilidad (XAI) del modelo XGBoost con SHAP: importancia global (summary plot)
y explicacion local para un cliente individual (waterfall plot), tal y como pide
el objetivo especifico 6 del anteproyecto.

Autor: Fernando Velasco Alonso
"""
import argparse
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/processed/telco_processed.csv")
    parser.add_argument("--models_dir", default="../models")
    parser.add_argument("--output", default="../reports")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    y = df["Churn"].values
    X = df.drop(columns=["Churn", "customerID"])

    import json
    with open(os.path.join(args.models_dir, "selected_features.json")) as f:
        selected_cols = json.load(f)

    scaler = joblib.load(os.path.join(args.models_dir, "scaler.joblib"))
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_sel = X_scaled[selected_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.25, random_state=42, stratify=y
    )

    model = joblib.load(os.path.join(args.models_dir, "XGBoost.joblib"))
    # Desde que 03_modeling.py mete SMOTE en un imblearn.Pipeline junto al clasificador
    # (para evitar fuga de datos en la CV), el objeto guardado ya no es un XGBClassifier
    # "pelado" sino el Pipeline completo. SHAP TreeExplainer necesita el estimador de
    # arbol puro, asi que se extrae el ultimo paso del pipeline.
    xgb_model = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    # --- Explicabilidad GLOBAL ---
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot - Importancia global de variables (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig09_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap.to_csv(os.path.join(args.output, "shap_importancia_global.csv"), index=False)
    print("--- Importancia global SHAP (top 10) ---")
    print(mean_abs_shap.head(10).to_string(index=False))

    # --- Explicabilidad LOCAL: cliente de mayor riesgo predicho en test ---
    probs = model.predict_proba(X_test)[:, 1]
    idx_riesgo = probs.argmax()
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[idx_riesgo], show=False)
    plt.title(f"Explicacion local SHAP - cliente de mayor riesgo (p={probs[idx_riesgo]:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig10_shap_waterfall_cliente_riesgo.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nCliente de mayor riesgo en test (idx={idx_riesgo}): "
          f"probabilidad de churn = {probs[idx_riesgo]:.3f}")
    print("Graficas SHAP guardadas en:", fig_dir)


if __name__ == "__main__":
    main()
