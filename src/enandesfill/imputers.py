import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter

def _post_clip(x, variable):
    if variable == 1:
        x = np.maximum(0.0, x)
    else:
        x = np.clip(x, -80.0, 60.0)
    return x

def _apply_kalman(values):
    x = values.copy().astype(float)
    mask = np.isnan(x)
    x_init = pd.Series(x).interpolate('linear', limit_direction='both').values
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=x_init[0],
                      observation_covariance=1,
                      transition_covariance=0.01)
    (state_means, _) = kf.smooth(x_init)
    x[mask] = state_means.reshape(-1)[mask]
    return x

def _stl_kalman(values, period):
    s = pd.Series(values)
    s0 = s.interpolate('linear', limit_direction='both')
    res = STL(s0, period=period, robust=True).fit()
    resid = s - res.seasonal
    resid_filled = _apply_kalman(resid.values)
    return (resid_filled + res.seasonal.values)

def _build_features(y, nlags=14, windows=(7, 30)):
    s = pd.Series(y)
    feats = {}
    for l in range(1, nlags+1):
        feats[f"lag{l}"] = s.shift(l)
    for w in windows:
        feats[f"roll_mean_{w}"] = s.rolling(w).mean()
        feats[f"roll_std_{w}"]  = s.rolling(w).std()
    X = pd.DataFrame(feats)
    return X

def _rf_impute(values):
    y = pd.Series(values).copy()
    X = _build_features(y, nlags=14, windows=(7,30))
    df = pd.concat([y.rename('target'), X], axis=1)
    train = df.dropna()
    if len(train) < 50:
        return y.interpolate().ffill().bfill().values
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(train.drop(columns=['target']), train['target'])
    miss_idx = df[df['target'].isna()].index
    if len(miss_idx) > 0:
        preds = model.predict(df.loc[miss_idx].drop(columns=['target']).fillna(method='ffill').fillna(method='bfill'))
        y.loc[miss_idx] = preds
    return y.values

def _missforest_impute(values):
    try:
        from missingpy import MissForest
    except Exception:
        raise ImportError("Necesita instalar missingpy: pip install missingpy")
    arr = values.reshape(-1,1).astype(float)
    imputer = MissForest(random_state=42)
    filled = imputer.fit_transform(arr)[:,0]
    return filled

def _xgboost_impute(values):
    try:
        from xgboost import XGBRegressor
    except Exception:
        raise ImportError("Necesita instalar xgboost: pip install xgboost")
    y = pd.Series(values).copy()
    X = _build_features(y, nlags=21, windows=(7,30,60))
    df = pd.concat([y.rename('target'), X], axis=1)
    train = df.dropna()
    if len(train) < 80:
        return y.interpolate().ffill().bfill().values
    model = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4)
    model.fit(train.drop(columns=['target']), train['target'])
    miss_idx = df[df['target'].isna()].index
    if len(miss_idx) > 0:
        preds = model.predict(df.loc[miss_idx].drop(columns=['target']).fillna(method='ffill').fillna(method='bfill'))
        y.loc[miss_idx] = preds
    return y.values

def _lightgbm_impute(values):
    try:
        import lightgbm as lgb
    except Exception:
        raise ImportError("Necesita instalar lightgbm: pip install lightgbm")
    y = pd.Series(values).copy()
    X = _build_features(y, nlags=21, windows=(7,30,60))
    df = pd.concat([y.rename('target'), X], axis=1)
    train = df.dropna()
    if len(train) < 80:
        return y.interpolate().ffill().bfill().values
    model = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(train.drop(columns=['target']), train['target'])
    miss_idx = df[df['target'].isna()].index
    if len(miss_idx) > 0:
        preds = model.predict(df.loc[miss_idx].drop(columns=['target']).fillna(method='ffill').fillna(method='bfill'))
        y.loc[miss_idx] = preds
    return y.values

def _autoencoder_impute(values, epochs=40, batch_size=32):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception:
        raise ImportError("Necesita instalar tensorflow: pip install tensorflow")
    # Preparar secuencias ventana para autoencoder  (simple sliding-window)
    y = pd.Series(values).copy()
    win = 14
    # Relleno ligero para entrenar
    y_fill = y.interpolate().ffill().bfill().values.astype('float32')
    # construir matriz con ventanas
    X = np.array([y_fill[i:i+win] for i in range(0, len(y_fill)-win)])
    # modelo autoencoder denso sencillo
    inp = layers.Input(shape=(win,))
    h = layers.Dense(16, activation='relu')(inp)
    z = layers.Dense(4, activation='relu')(h)
    h2 = layers.Dense(16, activation='relu')(z)
    out = layers.Dense(win)(h2)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
    # reconstrucción
    X_hat = model.predict(X, verbose=0)
    # reconstruir serie a partir de ventanas (promedio superpuesto)
    rec = np.zeros_like(y_fill)
    counts = np.zeros_like(y_fill)
    for i in range(len(X_hat)):
        rec[i:i+win] += X_hat[i]
        counts[i:i+win] += 1
    rec = rec / np.maximum(1, counts)
    # usar reconstrucción solo donde había NaN originalmente
    miss = y.isna().values
    y_fill[miss] = rec[miss]
    return y_fill

def fill_series(df, variable: int, method: str = 'stl_kalman', seasonal_period: int = 365):
    values = pd.to_numeric(df['value'], errors='coerce').values.astype(float)
    # Preprocesado para precipitación (sesgo y ceros) en métodos ML
    needs_log = (variable == 1 and method in ('knn','mice','rf','missforest','xgboost','lightgbm','autoencoder'))
    tvalues = np.log1p(np.maximum(0.0, values)) if needs_log else values.copy()

    if method == 'kalman':
        filled = _apply_kalman(tvalues)
    elif method == 'stl_kalman':
        filled = _stl_kalman(tvalues, seasonal_period)
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        filled = imputer.fit_transform(tvalues.reshape(-1,1)).ravel()
    elif method == 'mice':
        imputer = IterativeImputer(random_state=42, sample_posterior=False, max_iter=15, initial_strategy='median')
        filled = imputer.fit_transform(tvalues.reshape(-1,1)).ravel()
    elif method == 'rf':
        filled = _rf_impute(tvalues)
    elif method == 'missforest':
        filled = _missforest_impute(tvalues)
    elif method == 'xgboost':
        filled = _xgboost_impute(tvalues)
    elif method == 'lightgbm':
        filled = _lightgbm_impute(tvalues)
    elif method == 'autoencoder':
        filled = _autoencoder_impute(tvalues)
    else:
        raise ValueError('Método no soportado.')

    if needs_log:
        filled = np.expm1(filled)

    filled = _post_clip(filled, variable)
    out = df.copy()
    out['value_filled'] = filled
    out['imputed'] = df['value'].isna()
    return out
