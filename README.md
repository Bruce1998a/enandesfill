# enandesfill (Python) — v0.2.0

Librería Python para **relleno/imputación de series climáticas** (precipitación y temperatura).  
Incluye métodos clásicos y de *machine learning*, con reglas específicas para precipitación.

## Métodos soportados

- `kalman` → Filtro de Kalman (pykalman)
- `stl_kalman` → STL (descomposición estacional) + Kalman en el residuo
- `knn` → `KNNImputer` (scikit-learn)
- `mice` → `IterativeImputer` (MICE-like) (scikit-learn)
- `rf` → `RandomForestRegressor` con *lags* y *rolling stats*
- `missforest` → `missingpy.MissForest` (**instalar**: `pip install missingpy`)
- `xgboost` → `xgboost.XGBRegressor` (**instalar**: `pip install xgboost`)
- `lightgbm` → `lightgbm.LGBMRegressor` (**instalar**: `pip install lightgbm`)
- `autoencoder` → Autoencoder Keras/TensorFlow (**instalar**: `pip install tensorflow`)

> Las dependencias pesadas (XGBoost/LightGBM/TensorFlow/MissingPy) **son opcionales**. El paquete funciona sin ellas; si usas esos métodos y no están instaladas, verás un mensaje indicando cómo instalarlas.

---

## Instalación

Desde carpeta local (recomendado para pruebas):

```bash
pip install .
```

Después de subirlo a GitHub:
```bash
pip install git+https://github.com/Bruce1998a/enandesfill.git
```

### Extras opcionales (si usarás esos métodos)

```bash
pip install missingpy
pip install xgboost
pip install lightgbm
pip install tensorflow
```

---

## Uso rápido (Python)

```python
import pandas as pd
from enandesfill import load_example, verify_and_regularize, fill_series

# 1) Datos de ejemplo con NA
df = load_example()  # columnas: date (dd/mm/yyyy), value (float)

# 2) Regularizar fechas
dreg = verify_and_regularize(df, start_date="01/01/2020", end_date="31/12/2020")

# 3) Imputar (variable=1 precipitación; 2 tmedia; 3 tmax; 4 tmin)
filled = fill_series(dreg, variable=1, method="stl_kalman", seasonal_period=365)

# 4) Guardar
filled.to_csv("filled_example.csv", index=False)
```

### CLI

```bash
python -m enandesfill   --input src/enandesfill/ejemplo_fill_py.csv   --output out.csv   --variable 1   --method stl_kalman
```

---

## Notas por variable

- **1 = Precipitación (mm)**: se aplica `log1p` → imputación → `expm1` y se recorta a `[0, ∞)`.
- **2/3/4 = Temperatura (°C)**: continuo; recorte preventivo `[-80, 60]`.

---

## Buenas prácticas

- Establece `seasonal_period=365` para series diarias (ajústalo si usas otra frecuencia).
- Para `xgboost` y `lightgbm`, necesitas suficientes observaciones con ventana de *lags* para aprender.
- El Autoencoder es experimental; utilízalo cuando tengas bastantes datos y una estacionalidad marcada.
