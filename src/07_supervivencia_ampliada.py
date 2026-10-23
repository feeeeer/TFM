"""
07_supervivencia_ampliada.py
Ampliacion del analisis de supervivencia (pedida en la Propuesta 1: DeepSurv):
  1. Cox "pequeno" (6 covariables, el del script 04) vs Cox "completo" vs DeepSurv.
  2. Diagnostico del supuesto de riesgos proporcionales (Schoenfeld) y Cox estratificado.
  3. Seleccion de hiperparametros de DeepSurv por CV 5-fold (solo con el 75% de train)
     y evaluacion final en el 25% de test (misma semilla que la clasificacion).
  4. Calibracion de S(24 meses) y CLV dinamico con DeepSurv vs CLV con Cox.
Salidas: reports/survival_*.csv|json y figuras fig20-fig25.

Autor: Fernando Velasco Alonso
"""
import itertools
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split

from deepsurv_numpy import DeepSurv

warnings.filterwarnings("ignore")
RAW = "../data/raw/Telco-Customer-Churn_FULL.csv"
OUT = "../reports"
FIG = os.path.join(OUT, "figures")
H = 24  # horizonte CLV (meses)


def build_data():
    df = pd.read_csv(RAW)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    X = pd.DataFrame(index=df.index)
    X["duration"] = df["tenure"].clip(lower=1).astype(float)
    X["event"] = (df["Churn"] == "Yes").astype(int)
    X["MonthlyCharges"] = df["MonthlyCharges"]
    X["SeniorCitizen"] = df["SeniorCitizen"]
    for c in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        X[c] = (df[c].isin(["Yes", "Male"])).astype(int)
    for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]:
        X[c] = (df[c] == "Yes").astype(int)
    X["Fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    X["NoInternet"] = (df["InternetService"] == "No").astype(int)
    X["Contract_OneYear"] = (df["Contract"] == "One year").astype(int)
    X["Contract_TwoYear"] = (df["Contract"] == "Two year").astype(int)
    for p in ["Credit card (automatic)", "Electronic check", "Mailed check"]:
        X["Pay_" + p.split(" (")[0].replace(" ", "")] = (df["PaymentMethod"] == p).astype(int)
    X["Contract"] = df["Contract"]
    return df, X


SMALL = ["MonthlyCharges", "SeniorCitizen", "Contract_OneYear", "Contract_TwoYear", "Fiber", "PaperlessBilling"]


def cindex(dur, ev, risk):
    return concordance_index(dur, -risk, ev)


def cox_fit(Xtr, cols, penalizer=0.1, strata=None):
    cph = CoxPHFitter(penalizer=penalizer)
    c = cols + ["duration", "event"] + ([strata] if strata else [])
    cph.fit(Xtr[c], duration_col="duration", event_col="event", strata=[strata] if strata else None)
    return cph


def main():
    os.makedirs(FIG, exist_ok=True)
    df, X = build_data()
    FULL = [c for c in X.columns if c not in ("duration", "event", "Contract")]
    tr_idx, te_idx = train_test_split(X.index, test_size=0.25, random_state=42, stratify=X["event"])
    Xtr, Xte = X.loc[tr_idx].reset_index(drop=True), X.loc[te_idx].reset_index(drop=True)
    res = {"n_train": len(Xtr), "n_test": len(Xte), "n_cov_full": len(FULL)}

    # ---------------- 1. Cox pequeno / completo / estratificado: C-index test ----------------
    cox_s = cox_fit(Xtr, SMALL)
    cox_f = cox_fit(Xtr, FULL)
    cox_st = cox_fit(Xtr, [c for c in FULL if not c.startswith("Contract_")], strata="Contract")
    res["cindex_test"] = {
        "Cox_pequeno": cindex(Xte.duration, Xte.event, cox_s.predict_log_partial_hazard(Xte[SMALL]).values),
        "Cox_completo": cindex(Xte.duration, Xte.event, cox_f.predict_log_partial_hazard(Xte[FULL]).values),
    }
    cox_f.summary.to_csv(os.path.join(OUT, "survival_cox_completo.csv"))
    cox_st.summary.to_csv(os.path.join(OUT, "survival_cox_estratificado.csv"))

    # forest plot Cox completo
    fig, ax = plt.subplots(figsize=(7, 7))
    cox_f.plot(ax=ax)
    ax.set_title("Cox completo: coeficientes (log HR) con IC 95%")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig20_cox_completo_forest.png"), dpi=150); plt.close(fig)

    # ---------------- 2. Supuesto de riesgos proporcionales ----------------
    ph = proportional_hazard_test(cox_s, Xtr[SMALL + ["duration", "event"]], time_transform="rank")
    ph_df = ph.summary[["test_statistic", "p"]]
    ph_df.to_csv(os.path.join(OUT, "survival_test_PH_cox_pequeno.csv"))
    ph2 = proportional_hazard_test(cox_f, Xtr[FULL + ["duration", "event"]], time_transform="rank")
    ph2.summary[["test_statistic", "p"]].to_csv(os.path.join(OUT, "survival_test_PH_cox_completo.csv"))
    res["ph_violaciones_cox_pequeno"] = int((ph_df["p"] < 0.05).sum())
    res["ph_violaciones_cox_completo"] = int((ph2.summary["p"] < 0.05).sum())

    sch = cox_s.compute_residuals(Xtr[SMALL + ["duration", "event"]], kind="scaled_schoenfeld")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["Contract_TwoYear", "Fiber"]):
        tt = Xtr.loc[sch.index, "duration"] if len(sch) == len(Xtr) else Xtr.duration.loc[sch.index]
        ax.scatter(tt, sch[col], s=6, alpha=0.4)
        z = np.polyfit(tt, sch[col], 2); xs = np.linspace(tt.min(), tt.max(), 100)
        ax.plot(xs, np.polyval(z, xs), color="red")
        ax.axhline(0, color="gray", ls="--"); ax.set_title(f"Residuos de Schoenfeld escalados: {col}")
        ax.set_xlabel("Tiempo (meses)"); ax.set_ylabel("Residuo")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig21_schoenfeld.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ctr in ["Month-to-month", "One year", "Two year"]:
        m = X.Contract == ctr
        kmf = KaplanMeierFitter().fit(X.duration[m], X.event[m])
        sf = kmf.survival_function_.iloc[1:]
        ax.plot(np.log(sf.index), np.log(-np.log(sf.iloc[:, 0].clip(1e-6, 0.999999))), label=ctr)
    ax.set_xlabel("log(tiempo)"); ax.set_ylabel("log(-log S(t))"); ax.legend()
    ax.set_title("Grafico log-log por tipo de contrato (paralelismo = PH)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig22_km_loglog.png"), dpi=150); plt.close(fig)

    # ---------------- 3. DeepSurv: seleccion por CV en train ----------------
    grid = [dict(hidden=h, l2=l2) for h, l2 in itertools.product([(16,), (32, 16), (64, 32)], [1e-2, 1e-3])]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    Xtr_f = Xtr[FULL].values
    for g in grid:
        sc = []
        for a, b in skf.split(Xtr_f, Xtr.event):
            m = DeepSurv(hidden=g["hidden"], l2=g["l2"], lr=5e-3, epochs=200, seed=0).fit(Xtr_f[a], Xtr.duration.values[a], Xtr.event.values[a])
            sc.append(cindex(Xtr.duration.values[b], Xtr.event.values[b], m.predict_risk(Xtr_f[b])))
        rows.append(dict(hidden=str(g["hidden"]), l2=g["l2"], cindex_cv_mean=np.mean(sc), cindex_cv_std=np.std(sc)))
        print(rows[-1], flush=True)
    # referencia Cox en los mismos folds
    cs, cf = [], []
    for a, b in skf.split(Xtr_f, Xtr.event):
        A, B = Xtr.iloc[a], Xtr.iloc[b]
        cs.append(cindex(B.duration, B.event, cox_fit(A, SMALL).predict_log_partial_hazard(B[SMALL]).values))
        cf.append(cindex(B.duration, B.event, cox_fit(A, FULL).predict_log_partial_hazard(B[FULL]).values))
    gdf = pd.DataFrame(rows)
    gdf.to_csv(os.path.join(OUT, "survival_deepsurv_cv.csv"), index=False)
    res["cindex_cv"] = {"Cox_pequeno": [float(np.mean(cs)), float(np.std(cs))], "Cox_completo": [float(np.mean(cf)), float(np.std(cf))]}
    best = gdf.sort_values("cindex_cv_mean", ascending=False).iloc[0]
    bh = eval(best.hidden); bl2 = float(best.l2)
    res["deepsurv_mejor"] = {"hidden": list(bh), "l2": bl2, "cindex_cv": float(best.cindex_cv_mean)}

    ds = DeepSurv(hidden=bh, l2=bl2, lr=5e-3, epochs=200, seed=0).fit(Xtr_f, Xtr.duration.values, Xtr.event.values)
    res["cindex_test"]["DeepSurv"] = cindex(Xte.duration, Xte.event, ds.predict_risk(Xte[FULL].values))
    # IC bootstrap del C-index en test
    rng = np.random.default_rng(42)
    risk = {"Cox_pequeno": cox_s.predict_log_partial_hazard(Xte[SMALL]).values,
            "Cox_completo": cox_f.predict_log_partial_hazard(Xte[FULL]).values,
            "DeepSurv": ds.predict_risk(Xte[FULL].values)}
    boots = {k: [] for k in risk}
    for _ in range(500):
        ix = rng.integers(0, len(Xte), len(Xte))
        for k, r in risk.items():
            boots[k].append(cindex(Xte.duration.values[ix], Xte.event.values[ix], r[ix]))
    res["cindex_test_ic95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    res["diff_DeepSurv_menos_CoxCompleto_ic95"] = [float(x) for x in np.percentile(np.array(boots["DeepSurv"]) - np.array(boots["Cox_completo"]), [2.5, 97.5])]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ds.loss_history); ax.set_xlabel("Epoca"); ax.set_ylabel("Perdida (-log verosimilitud parcial + L2)")
    ax.set_title("DeepSurv: curva de entrenamiento")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig23_deepsurv_loss.png"), dpi=150); plt.close(fig)

    # ---------------- 4. Calibracion S(24) en test ----------------
    t24 = np.array([H])
    preds = {"Cox pequeño": cox_s.predict_survival_function(Xte[SMALL], times=[H]).values.ravel(),
             "Cox completo": cox_f.predict_survival_function(Xte[FULL], times=[H]).values.ravel(),
             "DeepSurv": ds.predict_survival(Xte[FULL].values, t24).ravel()}
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); cal = {}
    for name, s in preds.items():
        q = pd.qcut(s, 5, labels=False, duplicates="drop"); pp, oo = [], []
        for k in np.unique(q):
            mk = q == k
            kmf = KaplanMeierFitter().fit(Xte.duration[mk], Xte.event[mk])
            pp.append(s[mk].mean()); oo.append(float(kmf.predict(H)))
        ax.plot(pp, oo, marker="o", label=name); cal[name] = float(np.mean(np.abs(np.array(pp) - np.array(oo))))
    ax.plot([0.5, 1], [0.5, 1], "k--"); ax.set_xlabel(f"S({H}m) predicha (media por quintil)"); ax.set_ylabel(f"S({H}m) observada (Kaplan-Meier)")
    ax.legend(); ax.set_title("Calibracion de la supervivencia a 24 meses (test)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig24_calibracion_supervivencia.png"), dpi=150); plt.close(fig)
    res["calibracion_error_abs_medio_S24"] = cal

    # ---------------- 5. CLV con DeepSurv vs Cox (modelos ajustados con todos los datos) ----------------
    ds_all = DeepSurv(hidden=bh, l2=bl2, lr=5e-3, epochs=200, seed=0).fit(X[FULL].values, X.duration.values, X.event.values)
    S_ds = ds_all.predict_survival(X[FULL].values, np.arange(1, H + 1))
    clv_ds = df["MonthlyCharges"].values * S_ds.sum(1)
    clv_cox = pd.read_csv(os.path.join(OUT, "clv_estimado.csv")).set_index("customerID").loc[df["customerID"], f"CLV_{H}m"].values
    cox_all = cox_fit(X, FULL)
    clv_cf = df["MonthlyCharges"].values * cox_all.predict_survival_function(X[FULL], times=np.arange(1, H + 1)).values.sum(0)
    res["clv"] = {"media_cox_pequeno": float(clv_cox.mean()), "media_cox_completo": float(clv_cf.mean()), "media_deepsurv": float(clv_ds.mean()),
                  "total_cox_pequeno": float(clv_cox.sum()), "total_cox_completo": float(clv_cf.sum()), "total_deepsurv": float(clv_ds.sum()),
                  "spearman_cox_deepsurv": float(spearmanr(clv_cox, clv_ds)[0]), "spearman_coxcompleto_deepsurv": float(spearmanr(clv_cf, clv_ds)[0])}
    pd.DataFrame({"customerID": df["customerID"], "CLV_cox_pequeno": clv_cox, "CLV_cox_completo": clv_cf, "CLV_deepsurv": clv_ds}).to_csv(os.path.join(OUT, "clv_comparado.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(clv_cox, clv_ds, s=4, alpha=0.3); lim = [0, max(clv_cox.max(), clv_ds.max())]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("CLV 24m con Cox pequeño (€)"); ax.set_ylabel("CLV 24m con DeepSurv (€)")
    ax.set_title("Comparación del CLV individual: Cox vs DeepSurv")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig25_comparacion_clv.png"), dpi=150); plt.close(fig)

    with open(os.path.join(OUT, "survival_ampliada_resultados.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
