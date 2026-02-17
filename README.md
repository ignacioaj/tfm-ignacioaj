# 🧠 P300 Detection – EEG Checkerboard BCI

Detección de potenciales evocados **P300** en EEG (dataset BigP3BCI, PhysioNet).  
Clasificación de *epochs*: **target (1)** vs **non-target (0)**.

---

# 📁 Estructura del proyecto
```
├── main.py
├── scripts/
│ ├── signal_preprocessing.py
│ └── feature_extraction.py
├── notebooks/
│ ├── epoch_plot.ipynb
│ ├── data-analysis.ipynb
│ └── data-analysis-after-cleansing.ipynb
│ └── data-cleansing.ipynb
├── data/ # (NO incluido – disponible en Drive)
└── outputs/ # (generado automáticamente – no incluido)
```
⚠️ Las carpetas `data/` y `outputs/` no están en GitHub por tamaño.
- ```data```: https://drive.google.com/file/d/1r3ySy0zoSP30JyJ8kDfu-P3ML6YXO_XG/view?usp=drive_link
- ```outputs```: https://drive.google.com/file/d/1HK4WLzuq77-bhY_N7VOxwfYyNKQOf6q1/view?usp=drive_link

---

# 🚀 Ejecución
En primer lugar se deberá ejecutar main.py

```bash
python main.py
```

## Preprocesado EEG (scripts/signal_preprocessing.py)
- Band-pass 0.1–15 Hz, Notch 50 Hz, referencia promedio
- Segmentación por StimulusBegin y StimulusType
- Genera: outputs/epochs.parquet
- 
## Extracción de features (scripts/feature_extraction.py): 
- Temporales, Espaciales, Frecuenciales 

Genera:

- ```outputs/features_temporal.parquet```
- ```outputs/features_spatial.parquet```
- ```outputs/features_frequency.parquet```

# 📊 Análisis de datos
- ```data-analysis.ipynb```: EDA inicial (desbalance, outliers, multicolinealidad)
- ```cleansing.ipynb```: limpieza de outliers e imputación
- ```data-analysis-after-cleansing.ipynb```: análisis post-limpieza
- ```epoch_plot.ipynb```: visualización de epochs crudas

# 🤖 Modelado

## Modelos clásicos
- Logistic Regression, LDA (shrinkage), SVM lineal, XGBoost, LightGBM
- Validación: GroupKFold 
- SMOTE

## Red neuronal convolucional
- Ventana 200–550 ms, xDAWN (4 componentes)
- Conv1D + FC + Dropout
- BCEWithLogitsLoss, Adam
- Validación: GroupKFold

# 📈 Resultados
- AUC ≈ 0.5, F1 bajo
- SMOTE y ensemble no mejoran significativamente
- CNN limitada en generalización inter-sujeto
- El problema es altamente complejo y desbalanceado.

# 📦 Requisitos
```pip install -r requirements.txt```

Principales librerías:
mne, scikit-learn, imbalanced-learn, xgboost, lightgbm, torch, pandas, pyarrow

# 📥 Dataset
BigP3BCI – PhysioNet: https://physionet.org/content/bigp3bci/1.0.0/
