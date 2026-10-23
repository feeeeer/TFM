"""
09_calibracion_y_valor_esperado.py
Revision metodologica final (observaciones del tutor):
  1. Metricas de los 3 clasificadores en TRAIN, VALIDACION (5-fold CV, predicciones out-of-fold) y TEST.
  2. Calibracion de las probabilidades: SMOTE las sobreestima. Se mide (Brier, ECE, media predicha vs prevalencia)
     y se recalibra con regresion isotonica (CalibratedClassifierCV, ajustada solo con train).
  3. Simulacion economica SOLO sobre test y con etiquetas reales (sin circularidad): el CLV de los clientes de test
     se calcula con un Cox ajustado unicamente con train. Ahorro = tasa_exito * CLV_i * y_i - coste, de los contactados.
  4. Distribucion del CLV, curvas ROC/PR conjuntas y curvas de ganancia/lift conjuntas.
Salidas: reports/calibracion_valor_esperado.json, reports/*.csv, figuras fig31-fig38.

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
from lifelines import CoxPHFitter
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RAW = "../data/raw/Telco-Customer-Churn_FULL.csv"
PROC = "../data/processed/telco_processed.csv"
MODELS = "../models"
OUT = "../reports"
FIG = os.path.join(OUT, "figures")
H, COSTE, EXITO = 24, 15.0, 0.30
NAMES = ["Regresion_Logistica", "XGBoost", "MLP"]
LAB = {"Regresion_Logistica": "Regresión Logística", "XGBoost": "XGBoost", "MLP": "MLP"}
COL = {"Regresion_Logistica": "#1f77b4", "XGBoost": "#ff7f0e", "MLP": "#2ca02c"}


def ece(y, p, bins=10):
    """Error de calibracion esperado (bins de igual anchura)."""
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(auc=roc_auc_score(y, p), precision=precision_score(y, yh), recall=recall_score(y, yh),
                f1=f1_score(y, yh), brier=brier_score_loss(y, p))


def main():
    os.makedirs(FIG, exist_ok=True)
    raw = pd.read_csv(RAW)
    dfp = pd.read_csv(PROC)
    y = dfp["Churn"].values
    X = dfp.drop(columns=["Churn", "customerID"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    assert (yte == np.load(f"{MODELS}/y_test.npy")).all(), "la particion no coincide con la del modelado"
    sc = StandardScaler().fit(Xtr)
    sel = SelectKBest(f_classif, k=15).fit(sc.transform(Xtr), ytr)
    cols = list(X.columns[sel.get_support()])
    assert cols == json.load(open(f"{MODELS}/selected_features.json")), "variables seleccionadas distintas"
    Ztr, Zte = sel.transform(sc.transform(Xtr)), sel.transform(sc.transform(Xte))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    res = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "prevalencia_train": float(ytr.mean()), "prevalencia_test": float(yte.mean())}

    # ---------- 1. train / validacion (OOF) / test ----------
    rows, P_te, P_oof = [], {}, {}
    models = {n: joblib.load(f"{MODELS}/{n}.joblib") for n in NAMES}
    for n, m in models.items():
        ptr = m.predict_proba(Ztr)[:, 1]
        poof = cross_val_predict(clone(m), Ztr, ytr, cv=skf, method="predict_proba")[:, 1]
        pte = m.predict_proba(Zte)[:, 1]
        P_te[n], P_oof[n] = pte, poof
        for split, yy, pp in [("Train", ytr, ptr), ("Validación (CV 5-fold)", ytr, poof), ("Test", yte, pte)]:
            rows.append(dict(modelo=LAB[n], conjunto=split, **{k: round(v, 4) for k, v in metrics(yy, pp).items()}))
    tvt = pd.DataFrame(rows); tvt.to_csv(f"{OUT}/metricas_train_val_test.csv", index=False)
    print(tvt.to_string(index=False))

    # ---------- 2. calibracion ----------
    cal_rows, P_cal = [], {}
    for n, m in models.items():
        for met in ("isotonic", "sigmoid"):
            cc = CalibratedClassifierCV(clone(m), method=met, cv=5).fit(Ztr, ytr)
            pc = cc.predict_proba(Zte)[:, 1]
            if met == "isotonic": P_cal[n] = pc
            cal_rows.append(dict(modelo=LAB[n], version=f"calibrado ({met})", prob_media=pc.mean(), brier=brier_score_loss(yte, pc), ece=ece(yte, pc), auc=roc_auc_score(yte, pc)))
        p = P_te[n]
        cal_rows.append(dict(modelo=LAB[n], version="sin calibrar (SMOTE)", prob_media=p.mean(), brier=brier_score_loss(yte, p), ece=ece(yte, p), auc=roc_auc_score(yte, p)))
    cal = pd.DataFrame(cal_rows).round(4).sort_values(["modelo", "version"]); cal.to_csv(f"{OUT}/calibracion_probabilidades.csv", index=False)
    print(cal.to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, d, t in [(ax[0], P_te, "Sin calibrar (modelos entrenados con SMOTE)"), (ax[1], P_cal, "Calibradas (regresión isotónica)")]:
        for n in NAMES:
            fr, mp = calibration_curve(yte, d[n], n_bins=10, strategy="quantile")
            a.plot(mp, fr, "o-", color=COL[n], label=LAB[n])
        a.plot([0, 1], [0, 1], "k--", label="Calibración perfecta"); a.set_xlabel("Probabilidad predicha (media por decil)")
        a.set_ylabel("Frecuencia observada de churn"); a.set_title(t, fontsize=11); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig31_calibracion_antes_despues.png", dpi=150); plt.close(fig)

    # ---------- 3. CLV en test (Cox ajustado solo con train) y simulacion economica ----------
    s = raw.copy(); s["TotalCharges"] = pd.to_numeric(s["TotalCharges"], errors="coerce").fillna(0.0)
    c = s[["MonthlyCharges", "SeniorCitizen"]].copy()
    c["duration"] = s["tenure"].clip(lower=1); c["event"] = (s["Churn"] == "Yes").astype(int)
    c["Contract_OneYear"] = (s["Contract"] == "One year").astype(int); c["Contract_TwoYear"] = (s["Contract"] == "Two year").astype(int)
    c["Fiber"] = (s["InternetService"] == "Fiber optic").astype(int); c["Paperless"] = (s["PaperlessBilling"] == "Yes").astype(int)
    cph = CoxPHFitter(penalizer=0.1).fit(c.loc[Xtr.index], duration_col="duration", event_col="event")
    rmst_te = cph.predict_survival_function(c.loc[Xte.index], times=np.arange(1, H + 1)).sum(axis=0).values
    clv = s.loc[Xte.index, "MonthlyCharges"].values * rmst_te
    gain = EXITO * clv * yte                      # valor esperado de retener si el cliente iba a abandonar (y=1)
    N = len(yte)

    def net(idx, coste=COSTE): return float(gain[idx].sum() - coste * len(idx))
    def rand_net(n, coste=COSTE): return float(gain.sum() * n / N - coste * n)   # esperanza exacta de n contactos aleatorios
    score = {"Riesgo × CLV (modelo calibrado)": P_cal["XGBoost"] * clv, "Solo riesgo (modelo calibrado)": P_cal["XGBoost"],
             "Solo CLV": clv, "Riesgo × CLV (sin calibrar)": P_te["XGBoost"] * clv}
    budgets = [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    rows = []
    for b in budgets:
        n = max(1, int(round(N * b)))
        r = dict(presupuesto=f"{int(b*100)}%", contactados=n, aleatorio=rand_net(n))
        for k, sc_ in score.items(): r[k] = net(np.argsort(-sc_)[:n])
        # oraculo: contactar solo a los que van a abandonar, por CLV (cota superior)
        orc = np.argsort(-(gain * (yte == 1)))[:n]; r["Oráculo (cota superior)"] = net(orc)
        rows.append(r)
    eco = pd.DataFrame(rows); eco.round(0).to_csv(f"{OUT}/economia_test_por_presupuesto.csv", index=False)
    print(eco.round(0).to_string(index=False))
    # segmentos riesgo x valor (cortes en la mediana del test): tasa real de abandono y valor en juego
    pc = P_cal["XGBoost"]; hi_r = pc >= np.median(pc); hi_v = clv >= np.median(clv)
    seg = []
    for nm, m_ in [("Alto riesgo / alto valor", hi_r & hi_v), ("Alto riesgo / bajo valor", hi_r & ~hi_v),
                   ("Bajo riesgo / alto valor", ~hi_r & hi_v), ("Bajo riesgo / bajo valor", ~hi_r & ~hi_v)]:
        seg.append(dict(segmento=nm, clientes=int(m_.sum()), abandono_real=float(yte[m_].mean()), clv_medio=float(clv[m_].mean()),
                        ahorro_neto=float(gain[m_].sum() - COSTE * m_.sum())))
    pd.DataFrame(seg).round(3).to_csv(f"{OUT}/segmentos_riesgo_valor.csv", index=False); print(pd.DataFrame(seg).round(3))
    # IC bootstrap de la diferencia modelo - aleatorio al 20 %
    rng = np.random.default_rng(42); diffs, mods, rands = [], [], []
    n20 = int(round(N * 0.20)); sm = score["Riesgo × CLV (modelo calibrado)"]
    for _ in range(2000):
        ix = rng.integers(0, N, N); g = gain[ix]; o = np.argsort(-sm[ix])[:n20]
        m_ = g[o].sum() - COSTE * n20; r_ = g.sum() * n20 / N - COSTE * n20
        mods.append(m_); rands.append(r_); diffs.append(m_ - r_)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res["economia_20pct"] = dict(contactados=n20, modelo=float(eco.loc[eco.presupuesto == "20%", "Riesgo × CLV (modelo calibrado)"].iloc[0]),
                                aleatorio=float(rand_net(n20)), ic95_modelo=ci(mods), ic95_aleatorio=ci(rands), ic95_diferencia=ci(diffs),
                                valor_total_en_riesgo=float(gain.sum()), clientes_que_abandonan_test=int(yte.sum()),
                                churners_capturados_modelo=int(yte[np.argsort(-sm)[:n20]].sum()), churners_esperados_aleatorio=float(yte.sum() * n20 / N))
    # sensibilidad al coste de contacto (20 % de presupuesto)
    sens = []
    for cc_ in (5, 15, 50, 100):
        o = np.argsort(-sm)[:n20]; sens.append(dict(coste=cc_, modelo=net(o, cc_), aleatorio=rand_net(n20, cc_), contactar_a_todos=float(gain.sum() - cc_ * N)))
    pd.DataFrame(sens).round(0).to_csv(f"{OUT}/economia_test_sensibilidad_coste.csv", index=False); print(pd.DataFrame(sens).round(0))
    res["economia_sensibilidad_coste"] = sens
    # figura: ahorro frente a presupuesto
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = [int(round(N * b)) / N * 100 for b in budgets]
    ax.plot(xs, eco["Riesgo × CLV (modelo calibrado)"], "o-", color="seagreen", label="Riesgo × CLV (modelo calibrado)")
    ax.plot(xs, eco["Solo riesgo (modelo calibrado)"], "s--", color="#1f77b4", label="Solo riesgo")
    ax.plot(xs, eco["Solo CLV"], "^--", color="#9467bd", label="Solo CLV")
    ax.plot(xs, eco["aleatorio"], "k:", label="Aleatorio (esperanza)")
    ax.set_xlabel("Presupuesto (% de clientes de test contactados)"); ax.set_ylabel("Ahorro neto (€)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Ahorro neto sobre el conjunto de test (etiquetas reales, éxito 30 %, coste 15 €)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig32_ahorro_test_presupuesto.png", dpi=150); plt.close(fig)

    # ---------- 4. distribucion del CLV ----------
    sA = s.copy(); sA["CLV"] = np.nan
    cph_all = CoxPHFitter(penalizer=0.1).fit(c, duration_col="duration", event_col="event")
    sA["CLV"] = s["MonthlyCharges"].values * cph_all.predict_survival_function(c, times=np.arange(1, H + 1)).sum(axis=0).values
    q = sA["CLV"].quantile([.1, .25, .5, .75, .9]).round(1)
    desc = pd.DataFrame({"Contrato": ["Todos"] + sorted(sA.Contract.unique())})
    stat = lambda d: [round(d.mean(), 1), round(d.median(), 1), round(d.std(), 1), round(d.min(), 1), round(d.max(), 1)]
    desc[["media", "mediana", "desv", "min", "max"]] = [stat(sA.CLV)] + [stat(sA.loc[sA.Contract == k, "CLV"]) for k in sorted(sA.Contract.unique())]
    desc["n"] = [len(sA)] + [int((sA.Contract == k).sum()) for k in sorted(sA.Contract.unique())]
    desc.to_csv(f"{OUT}/clv_distribucion.csv", index=False); print(desc)
    res["clv"] = dict(media=float(sA.CLV.mean()), mediana=float(sA.CLV.median()), p10=float(q[.1]), p90=float(q[.9]),
                      asimetria=float(sA.CLV.skew()), top20_pct_del_valor=float(sA.CLV.sort_values(ascending=False).head(int(.2 * len(sA))).sum() / sA.CLV.sum()))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(sA.CLV, bins=40, color="#2F4D74", edgecolor="white"); ax[0].axvline(sA.CLV.mean(), color="r", ls="--", label=f"Media = {sA.CLV.mean():,.0f} €")
    ax[0].axvline(sA.CLV.median(), color="orange", ls="--", label=f"Mediana = {sA.CLV.median():,.0f} €"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("CLV a 24 meses (€)"); ax[0].set_ylabel("N.º de clientes"); ax[0].set_title("Distribución del CLV", fontsize=11)
    for k, cl in zip(sorted(sA.Contract.unique()), ["#d62728", "#ff7f0e", "#2ca02c"]):
        ax[1].hist(sA.loc[sA.Contract == k, "CLV"], bins=30, alpha=.55, color=cl, label=k)
    ax[1].legend(fontsize=8); ax[1].set_xlabel("CLV a 24 meses (€)"); ax[1].set_title("CLV según tipo de contrato", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig33_distribucion_clv.png", dpi=150); plt.close(fig)

    # ---------- 5. ROC + PR conjuntas, ganancia + lift conjuntos ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for n in NAMES:
        fpr, tpr, _ = roc_curve(yte, P_te[n]); ax[0].plot(fpr, tpr, color=COL[n], label=f"{LAB[n]} (AUC = {roc_auc_score(yte, P_te[n]):.3f})")
        pr, rc, _ = precision_recall_curve(yte, P_te[n]); ax[1].plot(rc, pr, color=COL[n], label=f"{LAB[n]} (AP = {average_precision_score(yte, P_te[n]):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", label="Azar"); ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Tasa de verdaderos positivos"); ax[0].set_title("Curvas ROC (test)", fontsize=11)
    ax[1].axhline(yte.mean(), color="k", ls="--", label=f"Azar ({yte.mean():.3f})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precisión"); ax[1].set_title("Curvas precisión-recall (test)", fontsize=11)
    for a in ax: a.legend(fontsize=8, loc="lower right" if a is ax[0] else "upper right"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig34_roc_pr_conjuntas.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6)); w = 0.27; lift_rows = []
    for k, n in enumerate(NAMES):
        o = np.argsort(-P_te[n]); yy = yte[o]; cg = np.cumsum(yy) / yy.sum(); x = np.arange(1, N + 1) / N
        ax[0].plot(x * 100, cg * 100, color=COL[n], label=LAB[n])
        dec = np.array_split(yy, 10); lift = [d.mean() / yte.mean() for d in dec]
        ax[1].bar(np.arange(1, 11) + (k - 1) * w, lift, w, color=COL[n], label=LAB[n])
        lift_rows.append(dict(modelo=LAB[n], lift_decil1=round(lift[0], 2), ganancia_10=round(cg[int(N * .1) - 1] * 100, 1), ganancia_30=round(cg[int(N * .3) - 1] * 100, 1)))
    ax[0].plot([0, 100], [0, 100], "k--", label="Aleatorio"); ax[0].set_xlabel("% de clientes contactados (por riesgo)"); ax[0].set_ylabel("% de abandonos capturados"); ax[0].set_title("Curva de ganancia acumulada (test)", fontsize=11)
    ax[1].axhline(1, color="k", ls="--"); ax[1].set_xlabel("Decil de riesgo (1 = mayor riesgo)"); ax[1].set_ylabel("Lift"); ax[1].set_title("Lift por decil (test)", fontsize=11); ax[1].set_xticks(range(1, 11))
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig35_ganancia_lift_conjuntas.png", dpi=150); plt.close(fig)
    pd.DataFrame(lift_rows).to_csv(f"{OUT}/ganancia_lift_resumen.csv", index=False); print(pd.DataFrame(lift_rows))

    res["train_val_test"] = tvt.to_dict("records"); res["calibracion"] = cal.to_dict("records")
    with open(f"{OUT}/calibracion_valor_esperado.json", "w") as f: json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("train_val_test", "calibracion")}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
