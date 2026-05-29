"""
Trener ensemble av tre modeller for kvm-pris: XGBoost + LightGBM + CatBoost.

Hver modell trenes med tidsserie-CV og egen hyperparameter-søk. Slutt-
prediksjonen er gjennomsnittet av de tre modellene. Sluttmodellen lagres
i models/saved/ sammen med metadata.

Bruk:
  python models/train_price.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sørg for at modelpakken er på sys.path
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features_price import build_price_features, categorical_feature_names  # noqa: E402
from features_shared import (  # noqa: E402
    choose_ensemble_weights,
    filter_quality,
    load_dataset,
)
from splits import TRAIN_END, make_time_split, time_series_cv_indices  # noqa: E402

# Demp støy-varsler fra ML-bibliotekene
warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

SAVED_DIR = _HERE / "saved"
SAVED_DIR.mkdir(exist_ok=True)
TARGET = "pris_kvm_alle_kommune"

# Hyperparameter-grids — bevisst små per modell siden vi har tre å søke gjennom
XGB_PARAMS = ["max_depth", "learning_rate", "min_child_weight", "subsample"]
XGB_GRID = list(itertools.product([4, 6, 8], [0.05, 0.1], [1, 3], [0.8, 1.0]))

LGB_PARAMS = ["num_leaves", "learning_rate", "min_child_samples"]
LGB_GRID = list(itertools.product([15, 31, 63], [0.05, 0.1], [5, 20]))

CB_PARAMS = ["depth", "learning_rate", "l2_leaf_reg"]
CB_GRID = list(itertools.product([4, 6, 8], [0.05, 0.1], [1, 3]))


# --- Modell-fabrikker ---

def make_xgb(params: dict, n_estimators: int = 2000) -> xgb.XGBRegressor:
    """Lag en ny XGBoost-modell satt opp for GPU-trening."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators, device="cuda", tree_method="hist",
        eval_metric="mae", early_stopping_rounds=30,
        random_state=42, verbosity=0, **params,
    )


def make_lgbm(params: dict, n_estimators: int = 2000) -> lgb.LGBMRegressor:
    """Lag en ny LightGBM-modell satt opp for GPU-trening."""
    # objective='regression_l1' = MAE-loss (matcher våre metrikker)
    return lgb.LGBMRegressor(
        n_estimators=n_estimators, device="gpu",
        objective="regression_l1",
        random_state=42, verbose=-1, **params,
    )


def make_catboost(params: dict, n_estimators: int = 2000) -> cb.CatBoostRegressor:
    """Lag en ny CatBoost-modell satt opp for GPU-trening."""
    return cb.CatBoostRegressor(
        iterations=n_estimators, task_type="GPU",
        loss_function="MAE",
        random_state=42, verbose=False,
        early_stopping_rounds=30, **params,
    )


# --- Trening med early stopping (egen for hver modell pga. API-forskjeller) ---

def fit_xgb(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_lgbm(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def fit_catboost(model, X_tr, y_tr, X_val, y_val, cat_features=None):
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features)
    return model


# --- Hyperparameter-søk ---

def cv_search(
    name: str,
    grid: list,
    param_names: list[str],
    factory,
    fit_fn,
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    cat_features: list[str] | None = None,
) -> tuple[dict, float]:
    """Prøv alle hyperparameter-kombinasjoner med tidsserie-CV på trenings-perioden."""
    # Begrens til trenings-perioden — val/test holdes urørt
    train_mask = (years <= TRAIN_END).to_numpy()
    Xt = X[train_mask].reset_index(drop=True)
    yt = y[train_mask].reset_index(drop=True)
    yt_years = years[train_mask].reset_index(drop=True)

    folds = time_series_cv_indices(yt_years, n_splits=3, min_train_years=8)
    print(f"\n[{name}] søk: {len(grid)} kombinasjoner x {len(folds)} folds")

    best_params: dict | None = None
    best_score = float("inf")
    t0 = time.time()

    # Iterer alle kombinasjoner, beregn snitt-CV-MAE
    for idx, combo in enumerate(grid, start=1):
        params = dict(zip(param_names, combo))
        scores: list[float] = []
        for tr_idx, vl_idx in folds:
            X_tr, y_tr = Xt.iloc[tr_idx], yt.iloc[tr_idx]
            X_vl, y_vl = Xt.iloc[vl_idx], yt.iloc[vl_idx]
            m = factory(params, n_estimators=1000)
            # Send cat_features til CatBoost, ellers vanlig fit
            if cat_features is not None:
                fit_fn(m, X_tr, y_tr, X_vl, y_vl, cat_features=cat_features)
            else:
                fit_fn(m, X_tr, y_tr, X_vl, y_vl)
            scores.append(mean_absolute_error(y_vl, m.predict(X_vl)))
        mean_score = float(np.mean(scores))
        marker = ""
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            marker = " *"
        # Progress: hver 4. + første + siste
        if idx % 4 == 0 or idx == 1 or idx == len(grid):
            elapsed = time.time() - t0
            print(f"  [{idx:2d}/{len(grid)}] CV-MAE={mean_score:.4f}{marker}  "
                  f"({elapsed:.0f}s)")

    assert best_params is not None
    print(f"  -> Beste: {best_params}, CV-MAE={best_score:.4f}")
    return best_params, best_score


def evaluate_predictions(name: str, y_true: pd.Series, pred: np.ndarray) -> dict:
    """Beregn og print MAE, RMSE, R² og MAPE for en prediksjon."""
    mae = mean_absolute_error(y_true, pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    r2 = r2_score(y_true, pred)
    mape = float((np.abs(y_true - pred) / y_true).mean() * 100)
    print(f"  {name:12s}  MAE = {mae:>8,.0f}  RMSE = {rmse:>8,.0f}  "
          f"MAPE = {mape:5.1f} %  R^2 = {r2:6.3f}")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "mape_pct": mape}


def main() -> None:
    print("=== Pris-modell (ensemble): XGBoost + LightGBM + CatBoost ===\n")
    print(f"XGBoost: {xgb.__version__}, LightGBM: {lgb.__version__}, "
          f"CatBoost: {cb.__version__}")

    # Last datasettet og preprocess
    df = load_dataset()
    print(f"Datasett: {df.shape[0]:,} rader x {df.shape[1]} kolonner")
    df = filter_quality(df)
    X_oh, X_cat, y = build_price_features(df, target=TARGET)
    # Reset-indekser for posisjonell indeksering med iloc
    X_oh = X_oh.reset_index(drop=True)
    X_cat = X_cat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    print(f"Etter preprocess: {len(X_oh):,} rader")
    print(f"  Features (one-hot): {len(X_oh.columns)}")
    print(f"  Features (rå-kat for CatBoost): {len(X_cat.columns)}")
    print(f"Target {TARGET}: median={y.median():,.0f}, "
          f"min={y.min():,.0f}, max={y.max():,.0f}")

    # Log-transform mål så MAE blir relativ og modellen lærer prosent-endringer
    y_log = pd.Series(np.log1p(y.values), index=y.index)

    # Tidsbasert split
    years = X_oh["aar"]
    split = make_time_split(years)
    X_tr_oh, X_val_oh, X_te_oh = X_oh.iloc[split.train], X_oh.iloc[split.val], X_oh.iloc[split.test]
    X_tr_cat, X_val_cat, X_te_cat = X_cat.iloc[split.train], X_cat.iloc[split.val], X_cat.iloc[split.test]
    y_tr_log = y_log.iloc[split.train]
    y_val_log = y_log.iloc[split.val]
    y_te_raw = y.iloc[split.test]
    print(f"\nSplitt: train={len(X_tr_oh):,}, val={len(X_val_oh):,}, test={len(X_te_oh):,}")

    # Sjekk om noen nye engineerte features faktisk havnet i feature-listen
    engineered = [c for c in X_oh.columns if c in (
        "pris_lag1_kommune", "pris_lag2_kommune", "inntekt_yoy_pct",
        "pris_til_inntekt_ratio_lag1", "pris_relativ_norge_lag1_pct",
    )]
    print(f"Engineerte features med: {engineered}")

    # --- Søk hyperparametere for hver modell ---
    cat_feats = categorical_feature_names()
    xgb_params, _ = cv_search("XGBoost", XGB_GRID, XGB_PARAMS,
                              make_xgb, fit_xgb, X_oh, y_log, years)
    lgbm_params, _ = cv_search("LightGBM", LGB_GRID, LGB_PARAMS,
                               make_lgbm, fit_lgbm, X_oh, y_log, years)
    cb_params, _ = cv_search("CatBoost", CB_GRID, CB_PARAMS,
                             make_catboost, fit_catboost, X_cat, y_log, years,
                             cat_features=cat_feats)

    # --- Tren endelige modeller med best params + early stopping på val ---
    print("\nTrener endelige modeller med beste hyperparameter...")

    t0 = time.time()
    xgb_model = make_xgb(xgb_params, n_estimators=3000)
    fit_xgb(xgb_model, X_tr_oh, y_tr_log, X_val_oh, y_val_log)
    t_xgb = time.time() - t0
    print(f"  XGBoost:  {t_xgb:.1f} s, beste iter = {xgb_model.best_iteration}")

    t0 = time.time()
    lgbm_model = make_lgbm(lgbm_params, n_estimators=3000)
    fit_lgbm(lgbm_model, X_tr_oh, y_tr_log, X_val_oh, y_val_log)
    t_lgbm = time.time() - t0
    print(f"  LightGBM: {t_lgbm:.1f} s, beste iter = {lgbm_model.best_iteration_}")

    t0 = time.time()
    cb_model = make_catboost(cb_params, n_estimators=3000)
    fit_catboost(cb_model, X_tr_cat, y_tr_log, X_val_cat, y_val_log, cat_features=cat_feats)
    t_cb = time.time() - t0
    print(f"  CatBoost: {t_cb:.1f} s, beste iter = {cb_model.best_iteration_}")

    # --- Velg ensemble-vekter på val-settet ---
    # choose_ensemble_weights prøver flere vekt-skjemaer og velger det med
    # lavest val-MAE. Faller tilbake til beste enkeltmodell hvis svake
    # modeller (typisk CatBoost her) ellers ville dratt ned ensemblet.
    val_pred_xgb_log = xgb_model.predict(X_val_oh)
    val_pred_lgbm_log = lgbm_model.predict(X_val_oh)
    val_pred_cb_log = cb_model.predict(X_val_cat)
    weights, scheme = choose_ensemble_weights(
        [val_pred_xgb_log, val_pred_lgbm_log, val_pred_cb_log],
        y_val_log.values,
    )
    print(f"\nEnsemble-vekter (skjema: {scheme}):")
    print(f"  XGBoost:  {weights[0]:.3f}")
    print(f"  LightGBM: {weights[1]:.3f}")
    print(f"  CatBoost: {weights[2]:.3f}")

    # --- Predikér på test og invertér log ---
    pred_xgb_log = xgb_model.predict(X_te_oh)
    pred_lgbm_log = lgbm_model.predict(X_te_oh)
    pred_cb_log = cb_model.predict(X_te_cat)
    # Vektet snitt i log-skala, så invers-transform til NOK/kvm
    pred_ens_log = (
        weights[0] * pred_xgb_log
        + weights[1] * pred_lgbm_log
        + weights[2] * pred_cb_log
    )

    pred_xgb = np.expm1(pred_xgb_log)
    pred_lgbm = np.expm1(pred_lgbm_log)
    pred_cb = np.expm1(pred_cb_log)
    pred_ens = np.expm1(pred_ens_log)

    # --- Evaluer hver modell + ensemble + naiv baseline ---
    print(f"\n=== Resultater på test-sett ({int(years[split.test].min())}-{int(years[split.test].max())}) ===")
    naive_pred = np.full_like(y_te_raw.values, np.median(y.iloc[split.train]))
    naive_m = evaluate_predictions("Naiv", y_te_raw, naive_pred)
    xgb_m = evaluate_predictions("XGBoost", y_te_raw, pred_xgb)
    lgbm_m = evaluate_predictions("LightGBM", y_te_raw, pred_lgbm)
    cb_m = evaluate_predictions("CatBoost", y_te_raw, pred_cb)
    ens_m = evaluate_predictions("Ensemble", y_te_raw, pred_ens)

    forbedring_naiv = (1 - ens_m["mae"] / naive_m["mae"]) * 100
    best_single_mae = min(xgb_m["mae"], lgbm_m["mae"], cb_m["mae"])
    forbedring_single = (1 - ens_m["mae"] / best_single_mae) * 100
    print(f"\nEnsemble vs naiv: {forbedring_naiv:.0f} % lavere MAE")
    print(f"Ensemble vs beste single: {forbedring_single:+.1f} % MAE (negativ = ensemble bedre)")

    # Per-år MAE for ensemble — avslører om modellen er skjev mot enkelte år
    years_te = years.iloc[split.test].to_numpy()
    ens_mae_per_year = {}
    for yr in sorted(np.unique(years_te)):
        m = years_te == yr
        ens_mae_per_year[int(yr)] = round(
            float(mean_absolute_error(y_te_raw.values[m], pred_ens[m])), 0
        )
    print(f"Ensemble MAE per år (NOK/kvm): {ens_mae_per_year}")

    # Top features fra XGBoost (representativt for ensemble-en)
    print("\nTop 15 features (XGBoost importance):")
    importance = pd.Series(xgb_model.feature_importances_, index=X_oh.columns)
    importance = importance.sort_values(ascending=False).head(15)
    for n, s in importance.items():
        bar = "#" * int(s * 30 / importance.max())
        print(f"  {s:.4f}  {n:38s} {bar}")

    # --- Lagre modeller + metadata ---
    ensemble = {
        "xgb": xgb_model,
        "lgbm": lgbm_model,
        "catboost": cb_model,
        "weights": {
            "xgb": float(weights[0]),
            "lgbm": float(weights[1]),
            "catboost": float(weights[2]),
        },
    }
    model_path = SAVED_DIR / "price_ensemble.joblib"
    joblib.dump(ensemble, model_path)

    metadata = {
        "type": "price_ensemble",
        "target": TARGET,
        "target_transform": "log1p",
        "trained_at": datetime.now().isoformat(),
        "feature_cols_oh": list(X_oh.columns),
        "feature_cols_cat": list(X_cat.columns),
        "categorical_features": cat_feats,
        "n_train": int(len(X_tr_oh)),
        "n_val": int(len(X_val_oh)),
        "n_test": int(len(X_te_oh)),
        "best_params": {
            "xgb": xgb_params,
            "lgbm": lgbm_params,
            "catboost": cb_params,
        },
        "ensemble_weights": {
            "xgb": float(weights[0]),
            "lgbm": float(weights[1]),
            "catboost": float(weights[2]),
        },
        "ensemble_scheme": scheme,
        "training_time_sec": {
            "xgb": round(t_xgb, 1),
            "lgbm": round(t_lgbm, 1),
            "catboost": round(t_cb, 1),
        },
        "test_metrics": {
            "xgb": xgb_m,
            "lgbm": lgbm_m,
            "catboost": cb_m,
            "ensemble": ens_m,
            "naive_baseline": naive_m,
        },
        "ensemble_mae_per_year": ens_mae_per_year,
        "ensemble_vs_naive_mae_pct": round(forbedring_naiv, 1),
        "ensemble_vs_best_single_mae_pct": round(forbedring_single, 2),
        "library_versions": {
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "catboost": cb.__version__,
        },
        "device": "cuda",
    }
    meta_path = SAVED_DIR / "price_ensemble_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nLagret: {model_path.name}")
    print(f"Lagret: {meta_path.name}")


if __name__ == "__main__":
    main()
