# TFM — Predicción de Churn y Valor de Vida del Cliente (CLV) mediante Modelos de Supervivencia y Deep Learning en Telecomunicaciones

Fernando Velasco Alonso — Máster en Análisis de Datos Masivos (Big Data), Universidad Europea de Madrid. Director: Jesús Gil Ruiz.

Repositorio con el código del TFM: pipeline completo (EDA, preprocesado, clasificación con SMOTE, supervivencia Cox y DeepSurv, CLV, SHAP, calibración y simulación económica) sobre el dataset público **IBM Telco Customer Churn** (7.043 clientes).

## Estructura

```
TFM_Churn_CLV/
├── data/raw/Telco-Customer-Churn_FULL.csv      # dataset completo (7.043 filas)
├── notebooks/TFM_Churn_CLV_Pipeline.ipynb      # pipeline completo en un notebook, ejecutado de principio a fin
├── src/
│   ├── 01_eda.py                       # EDA
│   ├── 02_preprocessing.py             # limpieza, codificación, feature engineering
│   ├── 03_modeling.py                  # Regresión Logística / XGBoost / MLP + SelectKBest + SMOTE + búsqueda de hiperparámetros
│   ├── 04_survival_clv.py              # Cox PH + Kaplan-Meier + CLV dinámico a 24 meses
│   ├── 05_explainability.py            # SHAP global y local
│   ├── 07_supervivencia_ampliada.py    # Cox completo, test de riesgos proporcionales, Cox estratificado, DeepSurv, calibración, CLV comparado
│   ├── 08_diagnosticos_ampliados.py    # EDA ampliado, curvas precisión-exhaustividad, umbral óptimo, equidad por subgrupos, SHAP dependence
│   ├── 09_calibracion_y_valor_esperado.py  # métricas train/val/test, calibración, simulación económica EN TEST, segmentos riesgo × valor
│   └── deepsurv_numpy.py               # DeepSurv implementado en NumPy (con comprobación del gradiente)
├── models/                             # modelos entrenados (.joblib), scaler y variables seleccionadas
├── reports/                            # tablas (.csv/.json) y figuras (reports/figures)
├── build_notebook.py                   # solo regenera el notebook a partir de los scripts
└── requirements.txt
```

## Cómo ejecutar

1. Crear un entorno con Python 3.10 e instalar las dependencias **con las versiones fijadas**:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install nbformat nbclient ipykernel      # solo si se va a ejecutar el notebook desde la línea de comandos
   ```

2. **Opción A — Notebook** (recomendado): abrir `notebooks/TFM_Churn_CLV_Pipeline.ipynb` y ejecutar todas las celdas en orden (o desde la terminal:
   `cd notebooks && jupyter nbconvert --to notebook --execute --inplace TFM_Churn_CLV_Pipeline.ipynb`). La primera celda de código define la ruta del CSV (`INPUT_CSV`).

3. **Opción B — Scripts** (desde la carpeta `src`, en este orden; cada script lee lo que generan los anteriores):

   ```bash
   cd src
   python3 01_eda.py --input ../data/raw/Telco-Customer-Churn_FULL.csv --output ../reports
   python3 02_preprocessing.py --input ../data/raw/Telco-Customer-Churn_FULL.csv --output ../data/processed/telco_processed.csv
   python3 03_modeling.py
   python3 04_survival_clv.py --input ../data/raw/Telco-Customer-Churn_FULL.csv --output ../reports
   python3 05_explainability.py
   python3 07_supervivencia_ampliada.py        # ≈ 30 s
   python3 08_diagnosticos_ampliados.py
   python3 09_calibracion_y_valor_esperado.py  # ≈ 10 s
   ```

Todas las semillas están fijadas (`random_state = 42`), por lo que los resultados son reproducibles.

## Resultados principales (conjunto de test: 1.761 clientes, 25% estratificado)

- **Churn real**: 26,5 % (1.869 de 7.043).
- **Clasificación (test)**: AUC-ROC 0,841 (Regresión Logística), 0,842 (XGBoost), 0,844 (MLP); en validación cruzada el mejor es XGBoost (0,844). Las diferencias no son relevantes.
- **Calibración**: SMOTE sobrestima las probabilidades (media predicha 0,38–0,41 frente a una tasa real de 0,27). La recalibración isotónica baja el ECE de 0,12–0,15 a 0,01–0,03 sin cambiar el AUC.
- **Cox PH**: frente al contrato mensual, el contrato a dos años reduce el riesgo un 81 % (HR = 0,19) y el anual un 67 % (HR = 0,33). Penalización L2 `penalizer = 0.1`.
- **DeepSurv (NumPy)**: C-index en test 0,861 frente a 0,850 del Cox completo y 0,796 del Cox sencillo.
- **CLV medio a 24 meses**: ≈ 1.295 €.
- **Simulación económica (solo test, etiquetas reales, éxito 30 %, coste 15 €)**: contactando al 20 % de los clientes priorizados por riesgo × CLV, ahorro neto de 92.247 € frente a 32.509 € con contacto aleatorio (IC 95 % de la diferencia: 52.484–67.033 €).

## Decisiones metodológicas y limitaciones (resumen)

- `StandardScaler` y `SelectKBest` (k = 15) se ajustan **una vez** sobre el conjunto de entrenamiento, fuera de los folds de validación cruzada; SMOTE y el clasificador sí están dentro del `Pipeline` de cada fold. Las métricas de test no se ven afectadas.
- `TotalCharges` se imputa a 0 en los 11 casos con `tenure = 0` y **sí** entra en los modelos de clasificación.
- Los espacios de búsqueda reales son: Regresión Logística, 5 combinaciones (`GridSearchCV`); XGBoost, 12 de 108 combinaciones (`RandomizedSearchCV`); MLP, 12 combinaciones (espacio completo).
- `CLV = MonthlyCharges × Σ_{t=1}^{24} Ŝ(t | x)`: parte del instante cero (no condiciona a la antigüedad actual), es un ingreso (no un margen) y no se descuenta.
- La tasa de éxito de retención (30 %) y el coste de contacto (15 €) son supuestos; la simulación no es una predicción del beneficio de una campaña real.

## Nota técnica: pin de versiones

`xgboost==2.0.3` es necesario para que `shap.TreeExplainer` funcione (las versiones 3.x rompen el parseo de `base_score` en SHAP 0.49) y es incompatible con `scikit-learn>=1.6`, que cambia el sistema de *tags* de los estimadores y hace que cualquier `Pipeline`/`cross_val_score` con XGBoost devuelva NaN. Combinación validada:

```
scikit-learn==1.5.2
xgboost==2.0.3
shap==0.49.1
imbalanced-learn==0.14.2
```
