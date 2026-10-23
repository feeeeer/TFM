"""
08_diagnosticos_ampliados.py
Analisis adicionales sobre los modelos de clasificacion YA entrenados (sin reentrenar):
  - EDA ampliado (fig15-fig19)
  - Curvas Precision-Recall (fig26)
  - Umbral de decision optimo por coste de negocio, con etiquetas reales (fig27)
  - Equidad por subgrupos: AUC, tasa de contacto y recall por genero/edad/pareja/contrato (fig28)
  - Dependencia SHAP de tenure y MonthlyCharges (fig29, fig30)
Salidas: reports/diagnosticos_*.csv|json

Autor: Fernando Velasco Alonso
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RAW, PROC, MODELS, OUT = "../data/raw/Telco-Customer-Churn_FULL.csv", "../data/processed/telco_processed.csv", "../models", "../reports"
FIG = os.path.join(OUT, "figures")
COSTE, EXITO = 15.0, 0.30


def eda(raw):
    raw = raw.copy()
    raw["Churn_bin"] = (raw["Churn"] == "Yes").astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, c in [("No", "#4c9"), ("Yes", "#e74")]:
        sns.kdeplot(raw.loc[raw.Churn == lab, "tenure"], ax=ax[0], label=f"Churn={lab}", color=c, fill=True, alpha=0.3)
        sns.kdeplot(raw.loc[raw.Churn == lab, "MonthlyCharges"], ax=ax[1], label=f"Churn={lab}", color=c, fill=True, alpha=0.3)
    ax[0].set_title("Densidad de tenure según Churn"); ax[1].set_title("Densidad de MonthlyCharges según Churn")
    ax[0].legend(); ax[1].legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig15_eda_tenure_charges.png", dpi=150); plt.close(fig)

    raw["TenureBin"] = pd.cut(raw.tenure, [-1, 6, 12, 24, 36, 48, 60, 72], labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
    ns = raw[["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]].eq("Yes").sum(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = raw.groupby("TenureBin").Churn_bin.agg(["mean", "size"]); ax[0].bar(t.index.astype(str), t["mean"], color="#357")
    for i, (m, n) in enumerate(zip(t["mean"], t["size"])): ax[0].text(i, m + 0.01, f"n={n}", ha="center", fontsize=8)
    ax[0].set_title("Tasa de churn por antigüedad (meses)"); ax[0].set_ylabel("Tasa de churn")
    s = raw.groupby(ns).Churn_bin.mean(); ax[1].bar(s.index.astype(str), s.values, color="#a53")
    ax[1].set_title("Tasa de churn por nº de servicios adicionales"); ax[1].set_xlabel("Servicios adicionales contratados")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig16_churn_tenure_servicios.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=raw, x="Contract", y="MonthlyCharges", hue="Churn", ax=ax, palette=["#4c9", "#e74"])
    ax.set_title("MonthlyCharges por tipo de contrato y Churn"); fig.tight_layout(); fig.savefig(f"{FIG}/fig17_boxplot_cargos_contrato.png", dpi=150); plt.close(fig)

    h = raw.pivot_table(index="InternetService", columns="Contract", values="Churn_bin", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.5, 4)); sns.heatmap(h, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    ax.set_title("Tasa de churn: internet × contrato"); fig.tight_layout(); fig.savefig(f"{FIG}/fig18_heatmap_internet_contrato.png", dpi=150); plt.close(fig)

    h2 = raw.pivot_table(index="PaymentMethod", columns="PaperlessBilling", values="Churn_bin", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.5, 4)); sns.heatmap(h2, annot=True, fmt=".2f", cmap="Oranges", ax=ax)
    ax.set_title("Tasa de churn: forma de pago × factura sin papel"); fig.tight_layout(); fig.savefig(f"{FIG}/fig19_heatmap_pago_papel.png", dpi=150); plt.close(fig)
    return {"churn_por_tenure": t["mean"].round(4).to_dict(), "churn_por_nservicios": s.round(4).to_dict(),
            "churn_internet_contrato": h.round(4).to_dict(), "churn_pago_papel": h2.round(4).to_dict(),
            "mediana_cargos_contrato_churn": raw.groupby(["Contract", "Churn"]).MonthlyCharges.median().round(2).reset_index().to_dict("records")}


def main():
    res = {}
    raw = pd.read_csv(RAW)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
    res["eda"] = eda(raw)

    df = pd.read_csv(PROC)
    y = df["Churn"].values
    X = df.drop(columns=["Churn", "customerID"])
    sel = json.load(open(f"{MODELS}/selected_features.json"))
    scaler = joblib.load(f"{MODELS}/scaler.joblib")
    Xs = pd.DataFrame(scaler.transform(X), columns=X.columns)[sel]
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
    assert (yte == np.load(f"{MODELS}/y_test.npy")).all(), "el split no coincide con el de 03_modeling"
    ids = df.loc[Xte.index, "customerID"].values
    clv = pd.read_csv(f"{OUT}/clv_estimado.csv").set_index("customerID").loc[ids, "CLV_24m"].values
    probs = {n: np.load(f"{MODELS}/y_prob_{n}.npy") for n in ["Regresion_Logistica", "XGBoost", "MLP"]}

    # ---- Precision-Recall ----
    fig, ax = plt.subplots(figsize=(6.5, 5)); pr = {}
    for n, p in probs.items():
        pre, rec, _ = precision_recall_curve(yte, p); ap = average_precision_score(yte, p); pr[n] = round(float(ap), 4)
        ax.plot(rec, pre, label=f"{n} (AP={ap:.3f})")
    ax.axhline(yte.mean(), color="gray", ls="--", label=f"Azar ({yte.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precisión"); ax.set_title("Curvas Precisión-Recall (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig26_pr_curves.png", dpi=150); plt.close(fig)
    res["average_precision"] = pr; res["prevalencia_test"] = float(yte.mean())

    # ---- Umbral optimo por coste (XGBoost), etiquetas reales ----
    p = probs["XGBoost"]; ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
    def neto(th):
        m = p >= th
        return float(np.sum(EXITO * clv[m] * yte[m] - COSTE)), int(m.sum())
    nets = np.array([neto(t)[0] for t in ths]); ncon = np.array([neto(t)[1] for t in ths])
    tb = float(ths[nets.argmax()])
    f1 = []
    for t in ths:
        m = p >= t; tp = (m & (yte == 1)).sum(); f1.append(2 * tp / max(m.sum() + (yte == 1).sum(), 1))
    tf1 = float(ths[int(np.argmax(f1))])
    res["umbral"] = {"coste_contacto": COSTE, "tasa_exito": EXITO, "umbral_max_beneficio": tb, "beneficio_max": float(nets.max()),
                     "clientes_contactados_umbral_opt": int(ncon[nets.argmax()]), "beneficio_umbral_0.5": neto(0.5)[0], "contactados_0.5": neto(0.5)[1],
                     "umbral_max_F1": tf1, "beneficio_umbral_maxF1": neto(tf1)[0], "beneficio_contactar_a_todos": neto(0.0)[0], "n_test": int(len(yte))}
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); por_coste = []
    for c in [15, 50, 100, 200]:
        nn = np.array([float(np.sum(EXITO * clv[p >= t] * yte[p >= t] - c)) for t in ths])
        k = int(nn.argmax()); por_coste.append(dict(coste_contacto=c, umbral_optimo=float(ths[k]), beneficio_max=float(nn.max()),
                                                    pct_contactados=round(100 * float((p >= ths[k]).mean()), 1),
                                                    beneficio_umbral_0_5=float(np.sum(EXITO * clv[p >= 0.5] * yte[p >= 0.5] - c)),
                                                    beneficio_contactar_a_todos=float(np.sum(EXITO * clv * yte - c))))
        ax.plot(ths, nn / 1000, label=f"Coste de contacto {c} €"); ax.plot(ths[k], nn[k] / 1000, "o", color="black")
    ax.axvline(0.5, color="gray", ls="--", label="Umbral 0,5"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Umbral de decisión sobre P(churn)"); ax.set_ylabel("Beneficio neto en test (miles de €)")
    ax.set_title("Beneficio neto según el umbral y el coste de contacto (XGBoost)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig27_umbral_optimo.png", dpi=150); plt.close(fig)
    res["umbral_por_coste"] = por_coste

    # ---- Subgrupos / equidad ----
    g = raw.loc[Xte.index].copy(); g["p"] = p; g["y"] = yte; g["pred"] = (p >= 0.5).astype(int)
    g["Edad"] = np.where(g.SeniorCitizen == 1, "Mayor (65+)", "No mayor")
    rows = []
    for col in ["gender", "Edad", "Partner", "Dependents", "Contract"]:
        for v, s in g.groupby(col):
            auc = roc_auc_score(s.y, s.p) if s.y.nunique() > 1 else np.nan
            rows.append(dict(variable=col, grupo=v, n=len(s), churn_real=round(s.y.mean(), 3), AUC=round(auc, 3),
                             tasa_contacto=round(s.pred.mean(), 3), recall=round(((s.pred == 1) & (s.y == 1)).sum() / max((s.y == 1).sum(), 1), 3),
                             precision=round(((s.pred == 1) & (s.y == 1)).sum() / max((s.pred == 1).sum(), 1), 3)))
    sg = pd.DataFrame(rows); sg.to_csv(f"{OUT}/diagnosticos_subgrupos.csv", index=False)
    names = {"gender": "Género", "Edad": "Edad", "Partner": "Pareja", "Dependents": "A cargo", "Contract": "Contrato"}
    lab = [f"{names[v]}: {gr}" for v, gr in zip(sg.variable, sg.grupo)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    for a, c, t in zip(ax, ["AUC", "tasa_contacto", "recall"], ["AUC-ROC", "Tasa de contacto (P ≥ 0,5)", "Recall"]):
        a.barh(lab, sg[c], color="#2F4D74"); a.set_title(t, fontsize=13); a.invert_yaxis(); a.tick_params(labelsize=11); a.set_xlim(0, 1.05)
        for y, v in enumerate(sg[c]): a.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig28_subgrupos.png", dpi=150); plt.close(fig)

    # ---- SHAP dependencia ----
    m = joblib.load(f"{MODELS}/XGBoost.joblib"); xgb = m.named_steps["clf"] if hasattr(m, "named_steps") else m
    sv = shap.TreeExplainer(xgb)(Xte)
    for feat, nm in [("tenure", "fig29_shap_dependence_tenure.png"), ("MonthlyCharges", "fig30_shap_dependence_charges.png")]:
        plt.figure(figsize=(6.5, 4.5)); shap.plots.scatter(sv[:, feat], color=sv[:, "HighRiskContract"], show=False)
        plt.title(f"Dependencia SHAP de {feat} (color: HighRiskContract)"); plt.tight_layout()
        plt.savefig(f"{FIG}/{nm}", dpi=150, bbox_inches="tight"); plt.close()
    # umbrales aproximados donde el efecto de tenure cambia de signo
    tv = scaler.mean_[list(X.columns).index("tenure")] + scaler.scale_[list(X.columns).index("tenure")] * Xte["tenure"].values
    sh = sv[:, "tenure"].values; o = np.argsort(tv); sm = pd.Series(sh[o]).rolling(60, center=True, min_periods=20).mean().values
    cross = tv[o][np.where(np.diff(np.sign(sm[~np.isnan(sm)])) != 0)[0]] if (~np.isnan(sm)).any() else []
    res["shap_tenure_cambio_signo_meses"] = [float(round(c, 1)) for c in cross[:3]]
    with open(f"{OUT}/diagnosticos_resultados.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "eda"}, indent=2, ensure_ascii=False, default=float))
    print(sg.to_string(index=False))


if __name__ == "__main__":
    main()
