"""
01_eda.py
Analisis Exploratorio de Datos (EDA) - Telco Customer Churn
TFM: Prediccion de Churn y CLV mediante Modelos de Supervivencia y Deep Learning
Autor: Fernando Velasco Alonso

Uso:
    python 01_eda.py --input ../data/raw/Telco-Customer-Churn.csv --output ../reports

Si no se pasa --input, usa por defecto la muestra parcial incluida en el repo
(Telco-Customer-Churn_SAMPLE.csv), pensada solo para validar que el pipeline
funciona de extremo a extremo. Para el analisis real del TFM hay que sustituirla
por el CSV completo de Kaggle (7,043 filas).
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges viene como string en el dataset original de IBM (hay celdas vacias
    # para clientes con tenure=0). Se convierte a numerico forzando NaN en esos casos.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def basic_report(df: pd.DataFrame, out_dir: str):
    lines = []
    lines.append("=" * 70)
    lines.append("REPORTE EDA - TELCO CUSTOMER CHURN")
    lines.append("=" * 70)
    lines.append(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
    lines.append("")
    lines.append("--- Tipos de dato ---")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("--- Valores nulos por columna ---")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    lines.append(nulls.to_string() if len(nulls) else "Sin valores nulos directos "
                  "(ojo: TotalCharges puede tener nulos tras la conversion numerica)")
    lines.append("")
    lines.append("--- Duplicados (customerID) ---")
    lines.append(f"customerID duplicados: {df['customerID'].duplicated().sum()}")
    lines.append("")
    lines.append("--- Balance de clases (Churn) ---")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    for k in churn_counts.index:
        lines.append(f"  {k}: {churn_counts[k]} ({churn_pct[k]:.1f}%)")
    lines.append("")
    lines.append("--- Estadisticos descriptivos (numericas) ---")
    lines.append(df.describe().to_string())
    lines.append("")

    report_txt = "\n".join(lines)
    with open(os.path.join(out_dir, "eda_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
    print(report_txt)
    return churn_counts


def plot_churn_balance(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = df["Churn"].value_counts().index
    sns.countplot(data=df, x="Churn", order=order, ax=ax, palette="Set2")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Distribucion de Churn (desbalanceo de clases)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig01_churn_balance.png"), dpi=150)
    plt.close(fig)


def plot_numeric_by_churn(df: pd.DataFrame, out_dir: str):
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(data=df, x="Churn", y=col, ax=ax, palette="Set3")
        ax.set_title(f"{col} vs Churn")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig02_numeric_vs_churn.png"), dpi=150)
    plt.close(fig)


def plot_categorical_churn_rate(df: pd.DataFrame, out_dir: str):
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
                "SeniorCitizen", "Partner", "Dependents", "TechSupport", "OnlineSecurity"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    n = len(cat_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    df_plot = df.copy()
    df_plot["ChurnFlag"] = (df_plot["Churn"] == "Yes").astype(int)
    for i, col in enumerate(cat_cols):
        rate = df_plot.groupby(col)["ChurnFlag"].mean().sort_values(ascending=False)
        sns.barplot(x=rate.index, y=rate.values, ax=axes[i], palette="rocket")
        axes[i].set_title(f"Tasa de churn por {col}")
        axes[i].set_ylabel("Tasa de churn")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig03_churn_rate_categoricas.png"), dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out_dir: str):
    df_num = df.copy()
    df_num["ChurnFlag"] = (df_num["Churn"] == "Yes").astype(int)
    df_num["SeniorCitizen"] = df_num["SeniorCitizen"].astype(int)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "ChurnFlag"]
    corr = df_num[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlacion entre variables numericas y Churn")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig04_correlacion.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/raw/Telco-Customer-Churn_SAMPLE.csv")
    parser.add_argument("--output", default="../reports")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = load_data(args.input)
    basic_report(df, args.output)
    plot_churn_balance(df, fig_dir)
    plot_numeric_by_churn(df, fig_dir)
    plot_categorical_churn_rate(df, fig_dir)
    plot_correlation(df, fig_dir)

    print(f"\nFiguras guardadas en: {fig_dir}")
    print(f"Reporte de texto en: {os.path.join(args.output, 'eda_report.txt')}")


if __name__ == "__main__":
    main()
