"""
04_survival_clv.py
Modelo de supervivencia (Cox Proportional Hazards) + estimacion de CLV dinamico.
Este script es el que justifica el titulo del TFM ("Modelos de Supervivencia...")
y conecta con la Propuesta 1 (framework supervivencia + CLV + causal).

Planteamiento:
    - duration = tenure (meses que el cliente lleva activo/estuvo activo)
    - event = Churn (1 = abandono observado, 0 = cliente censurado, sigue activo)
    Esto es la forma estandar de tratar este dataset como snapshot de supervivencia
    (ver Larivière & Van den Poel, 2005). OJO: es una aproximacion razonable pero
    imperfecta porque no tenemos fecha de alta real ni seguimiento longitudinal;
    conviene documentar esta limitacion en la memoria (capitulo de limitaciones).

Salidas:
    - Modelo CoxPH ajustado (resumen de hazard ratios)
    - Curvas de supervivencia por segmento de contrato
    - CLV estimado = Ingreso mensual esperado x Esperanza de vida restante (RMST)
      truncada a un horizonte de negocio (p.ej. 24 meses)

Autor: Fernando Velasco Alonso
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

warnings.filterwarnings("ignore")


def prepare_survival_df(path_raw: str) -> pd.DataFrame:
    df = pd.read_csv(path_raw)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["event"] = (df["Churn"] == "Yes").astype(int)
    df["duration"] = df["tenure"].clip(lower=1)  # evitar duracion 0 (rompe Cox)
    return df


def fit_cox(df: pd.DataFrame, out_dir: str):
    cols = ["duration", "event", "MonthlyCharges", "SeniorCitizen"]
    df_cox = df[cols].copy()

    df_cox["Contract_OneYear"] = (df["Contract"] == "One year").astype(int)
    df_cox["Contract_TwoYear"] = (df["Contract"] == "Two year").astype(int)
    df_cox["Fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df_cox["Paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_cox, duration_col="duration", event_col="event")

    summary = cph.summary[["coef", "exp(coef)", "p"]].sort_values("exp(coef)", ascending=False)
    summary.to_csv(os.path.join(out_dir, "cox_hazard_ratios.csv"))
    print("\n--- Cox Proportional Hazards: Hazard Ratios (exp(coef)) ---")
    print(summary.to_string())
    print("\nInterpretacion: exp(coef) > 1 aumenta el riesgo instantaneo de churn,")
    print("exp(coef) < 1 lo reduce (efecto protector), respecto a la referencia.")

    fig, ax = plt.subplots(figsize=(6, 4))
    cph.plot(ax=ax)
    ax.set_title("Coeficientes Cox (log hazard ratio) con IC 95%")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig06_cox_coeficientes.png"), dpi=150)
    plt.close(fig)

    return cph, df_cox


def plot_km_by_contract(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for contract_type in df["Contract"].unique():
        mask = df["Contract"] == contract_type
        kmf.fit(df.loc[mask, "duration"], df.loc[mask, "event"], label=contract_type)
        kmf.plot_survival_function(ax=ax)
    ax.set_title("Curvas de Supervivencia Kaplan-Meier por tipo de contrato")
    ax.set_xlabel("Meses (tenure)")
    ax.set_ylabel("Probabilidad de retencion S(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig07_kaplan_meier_contrato.png"), dpi=150)
    plt.close(fig)


def estimate_clv(df: pd.DataFrame, cph: CoxPHFitter, df_cox: pd.DataFrame,
                  out_dir: str, horizon_months: int = 24):
    """
    CLV_i = MonthlyCharges_i * RMST_i(horizon)
    donde RMST (Restricted Mean Survival Time) es la esperanza de vida restante
    del cliente truncada al horizonte de negocio, obtenida de la funcion de
    supervivencia individual predicha por el modelo Cox.
    """
    surv_funcs = cph.predict_survival_function(df_cox, times=np.arange(1, horizon_months + 1))
    rmst = surv_funcs.sum(axis=0)  # aproximacion discreta de la integral de S(t)

    clv_df = pd.DataFrame({
        "customerID": df["customerID"].values,
        "MonthlyCharges": df["MonthlyCharges"].values,
        "tenure_actual": df["tenure"].values,
        "Churn_actual": df["Churn"].values,
        f"RMST_{horizon_months}m": rmst.values,
    })
    clv_df[f"CLV_{horizon_months}m"] = clv_df["MonthlyCharges"] * clv_df[f"RMST_{horizon_months}m"]
    clv_df = clv_df.sort_values(f"CLV_{horizon_months}m", ascending=False)

    clv_df.to_csv(os.path.join(out_dir, "clv_estimado.csv"), index=False)

    print(f"\n--- CLV estimado (horizonte {horizon_months} meses) ---")
    print(clv_df.head(10).to_string(index=False))
    print(f"\nCLV medio de la cartera: {clv_df[f'CLV_{horizon_months}m'].mean():.2f} EUR")
    print(f"CLV total de la cartera (muestra): {clv_df[f'CLV_{horizon_months}m'].sum():.2f} EUR")

    # Prioriza clientes de alto riesgo Y alto valor (cuadrante superior derecho)
    fig, ax = plt.subplots(figsize=(6, 5))
    risk_proxy = 1 - rmst.values / horizon_months  # a mas bajo RMST relativo, mas riesgo
    scatter = ax.scatter(risk_proxy, clv_df[f"CLV_{horizon_months}m"],
                          c=(clv_df["Churn_actual"] == "Yes"), cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("Riesgo relativo de churn (proxy 1 - RMST/horizonte)")
    ax.set_ylabel(f"CLV estimado a {horizon_months} meses (EUR)")
    ax.set_title("Matriz Riesgo x Valor para priorizar retencion")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fig08_matriz_riesgo_valor.png"), dpi=150)
    plt.close(fig)

    return clv_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/raw/Telco-Customer-Churn_SAMPLE.csv")
    parser.add_argument("--output", default="../reports")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)

    df = prepare_survival_df(args.input)
    cph, df_cox = fit_cox(df, args.output)
    plot_km_by_contract(df, args.output)
    estimate_clv(df, cph, df_cox, args.output, horizon_months=args.horizon)


if __name__ == "__main__":
    main()
