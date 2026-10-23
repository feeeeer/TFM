"""
02_preprocessing.py
Limpieza, codificacion y feature engineering para el dataset Telco Customer Churn.
TFM: Prediccion de Churn y CLV mediante Modelos de Supervivencia y Deep Learning
Autor: Fernando Velasco Alonso

Genera:
    ../data/processed/telco_processed.csv

Decisiones de preprocesado (documentar en la memoria, capitulo de Metodologia):
    1. TotalCharges: viene como string; los ~0.15% de casos vacios corresponden a
       clientes con tenure=0 (clientes nuevos, aun no facturados). Se imputan a 0
       en vez de eliminarlos, porque customerID y el resto de variables son validos.
    2. Variables "No internet service" / "No phone service": se colapsan a "No"
       porque son redundantes con InternetService/PhoneService (evita multicolinealidad
       y columnas dummy inútiles tras el One-Hot Encoding).
    3. Codificacion: Label Encoding para binarias (Yes/No, gender), One-Hot para
       categoricas con >2 niveles (Contract, InternetService, PaymentMethod).
    4. Feature engineering:
         - AvgMonthlySpend = TotalCharges / max(tenure, 1)
         - NumServices = conteo de servicios contratados (proxy de "engagement")
         - TenureGroup = bucketizacion (0-12, 12-24, 24-48, 48-72 meses)
         - HighRiskContract = 1 si Contract == Month-to-month (el driver de churn
           mas fuerte segun la literatura, Ahmad et al. 2019)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


SERVICE_COLS = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                 "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                 "StreamingMovies"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Colapsar "No internet/phone service" -> "No" (redundante con la variable madre)
    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    yes_services = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = (df[yes_services] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int) \
        + (df["PhoneService"] == "Yes").astype(int)

    bins = [0, 12, 24, 48, 73]
    labels = ["0-12m", "12-24m", "24-48m", "48-72m"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

    df["HighRiskContract"] = (df["Contract"] == "Month-to-month").astype(int)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    onehot_cols = ["InternetService", "Contract", "PaymentMethod", "TenureGroup"]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/raw/Telco-Customer-Churn_SAMPLE.csv")
    parser.add_argument("--output", default="../data/processed/telco_processed.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n0 = len(df)
    df = clean(df)
    df = feature_engineering(df)
    df = encode(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Filas entrada: {n0} | Filas salida: {len(df)} | Columnas salida: {df.shape[1]}")
    print(f"Guardado en: {args.output}")
    print("\nColumnas finales:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
