"""
Trener ensemble for investerings-modellen: XGBoost + LightGBM + CatBoost.

Mål er percentile rank (0-1) av kommunens fremtidige real prisstigning
innen samme år. Reframet fra absolutt avkastning for å eliminere regime-
skift-problemet — se features_investment.py for begrunnelsen.

Evaluering på testsettet:
  - Per-år Spearman: snitt over testårene
  - Top-10 precision: av modellens topp-10 hver år, hvor mange var i topp 25 %?
  - MAE på rank-skala (0-1) — sekundær metrikk

Lagrer modellene + metadata i models/saved/investment_ensemble.*.

Bruk:
  python models/train_investment.py
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
from sklearn.metrics import mean_absolute_error

# Sørg for at modelpakken er på sys.path
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features_investment import (  # noqa: E402
    build_investment_features,
    categorical_feature_names,
)
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
HORIZON = 1  # 1 år frem; lengre testet og forkastet (negativ Spearman)

# Hyperparameter-grids (samme som pris)
XGB_PARAMS = ["max_depth", "learning_rate", "min_child_weight", "subsample"]
XGB_GRID = list(itertools.product([4, 6, 8], [0.05, 0.1], [1, 3], [0.8, 1.0]))

LGB_PARAMS = ["num_leaves", "learning_rate", "min_child_samples"]
LGB_GRID = list(itertools.product([15, 31, 63], [0.05, 0.1], [5, 20]))

CB_PARAMS = ["depth", "learning_rate", "l2_leaf_reg"]
CB_GRID = list(itertools.product([4, 6, 8], [0.05, 0.1], [1, 3]))


# --- Modell-fabrikker (loss: MAE på rank-skala) ---

def make_xgb(params: dict, n_estimators: int = 2000) -> xgb.XGBRegressor:
    """Lag XGBoost-modell for ranking-regresjon."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators, device="cuda", tree_method="hist",
        eval_metric="mae", early_stopping_rounds=30,
        random_state=42, verbosity=0, **params,
    )


def make_lgbm(params: dict, n_estimators: int = 2000) -> lgb.LGBMRegressor:
    """Lag LightGBM-modell for ranking-regresjon."""
    return lgb.LGBMRegressor(
        n_estimators=n_estimators, device="gpu",
        objective="regression_l1",
        random_state=42, verbose=-1, **params,
    )


def make_catboost(params: dict, n_estimators: int = 2000) -> cb.CatBoostRegressor:
    """Lag CatBoost-modell for ranking-regresjon."""
    return cb.CatBoostRegressor(
        iterations=n_estimators, task_type="GPU",
        loss_function="MAE",
        random_state=42, verbose=False,
        early_stopping_rounds=30, **params,
    )


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


# --- Hyperparameter-søk (samme mønster som pris-modellen) ---

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
    """TimeSeriesSplit-CV over hele param-griden, returner beste."""
    train_mask = (years <= TRAIN_END).to_numpy()
    Xt = X[train_mask].reset_index(drop=True)
    yt = y[train_mask].reset_index(drop=True)
    yt_years = years[train_mask].reset_index(drop=True)

    folds = time_series_cv_indices(yt_years, n_splits=3, min_train_years=8)
    print(f"\n[{name}] søk: {len(grid)} kombinasjoner x {len(folds)} folds")

    best_params: dict | None = None
    best_score = float("inf")
    t0 = time.time()

    for idx, combo in enumerate(grid, start=1):
        params = dict(zip(param_names, combo))
        scores: list[float] = []
        for tr_idx, vl_idx in folds:
            X_tr, y_tr = Xt.iloc[tr_idx], yt.iloc[tr_idx]
            X_vl, y_vl = Xt.iloc[vl_idx], yt.iloc[vl_idx]
            m = factory(params, n_estimators=1000)
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
        if idx % 4 == 0 or idx == 1 or idx == len(grid):
            elapsed = time.time() - t0
            print(f"  [{idx:2d}/{len(grid)}] CV-MAE={mean_score:.4f}{marker}  "
                  f"({elapsed:.0f}s)")

    assert best_params is not None
    print(f"  -> Beste: {best_params}, CV-MAE={best_score:.4f}")
    return best_params, best_score


# --- Evaluerings-helpere for ranking ---

def aggregate_to_kommune_year(
    pred: np.ndarray, y_true: np.ndarray, y_raw: np.ndarray,
    kommune_nr: pd.Series, years: pd.Series,
) -> pd.DataFrame:
    """Snitt postnummer-prediksjoner per (kommune, år) for ranking-evaluering."""
    # Investerings-mål er kommune-nivå men data er postnummer-nivå —
    # vi midler prediksjoner per kommune for å få meningsfull rank-evaluering.
    df = pd.DataFrame({
        "kommune_nr": kommune_nr.values,
        "aar": years.values,
        "pred": pred,
        "true_rank": y_true,
        "true_raw": y_raw,
    })
    agg = df.groupby(["kommune_nr", "aar"]).agg(
        pred=("pred", "mean"),
        true_rank=("true_rank", "first"),  # samme for alle postnummer
        true_raw=("true_raw", "first"),
    ).reset_index()
    return agg


def per_year_spearman(agg: pd.DataFrame) -> tuple[float, dict]:
    """Snitt Spearman-korrelasjon per år (predikert vs faktisk rank)."""
    # Korrelasjon per år, så snitt over alle testårene
    per_year = {}
    for year, grp in agg.groupby("aar"):
        if len(grp) < 5:
            continue
        corr = grp["pred"].corr(grp["true_rank"], method="spearman")
        per_year[int(year)] = float(corr) if not pd.isna(corr) else 0.0
    if not per_year:
        return 0.0, {}
    mean_corr = float(np.mean(list(per_year.values())))
    return mean_corr, per_year


def top_n_precision(
    agg: pd.DataFrame, n: int = 10, threshold: float = 0.75,
) -> tuple[float, dict]:
    """Av modellens topp-N prediksjoner per år, hvor stor andel var faktisk
    i topp (1-threshold) %?"""
    # Hvis n=10 og threshold=0.75, sjekker vi om de 10 høyest-rangerte
    # var i faktisk topp-25 %.
    per_year = {}
    for year, grp in agg.groupby("aar"):
        if len(grp) < n:
            continue
        # Sorter etter prediksjon, ta topp N
        top = grp.nlargest(n, "pred")
        # Hvor mange av disse hadde faktisk true_rank >= threshold?
        hits = (top["true_rank"] >= threshold).sum()
        per_year[int(year)] = float(hits / n)
    if not per_year:
        return 0.0, {}
    mean_prec = float(np.mean(list(per_year.values())))
    return mean_prec, per_year


def per_year_mae(agg: pd.DataFrame) -> dict:
    """MAE på rank-skala per år (predikert vs faktisk rank)."""
    # Avslører om et samlet snitt skjuler store år-til-år-svingninger
    out = {}
    for year, grp in agg.groupby("aar"):
        out[int(year)] = float(mean_absolute_error(grp["true_rank"], grp["pred"]))
    return out


def evaluate(name: str, agg: pd.DataFrame) -> dict:
    """Skriv ut Spearman + top-10 precision + MAE for én modells aggregerte prediksjoner."""
    sp_mean, sp_per_year = per_year_spearman(agg)
    p10, p10_per_year = top_n_precision(agg, n=10, threshold=0.75)
    mae = mean_absolute_error(agg["true_rank"], agg["pred"])
    mae_per_year = per_year_mae(agg)
    print(f"  {name:12s}  Spearman = {sp_mean:+.3f}   "
          f"Top-10 precision = {p10:.2f}   MAE_rank = {mae:.3f}")
    return {
        "spearman_mean": sp_mean,
        "spearman_per_year": sp_per_year,
        "top10_precision_mean": p10,
        "top10_precision_per_year": p10_per_year,
        "mae_rank": float(mae),
        "mae_rank_per_year": mae_per_year,
    }


def main() -> None:
    print(f"=== Investerings-modell (ensemble): forutsi rank av {HORIZON}-ars fremtidig real avkastning ===\n")
    print(f"XGBoost: {xgb.__version__}, LightGBM: {lgb.__version__}, "
          f"CatBoost: {cb.__version__}")

    # Last og preprocess
    df = load_dataset()
    print(f"Datasett: {df.shape[0]:,} rader x {df.shape[1]} kolonner")
    df = filter_quality(df)

    # build_investment_features returnerer X_oh, X_cat, y_rank, y_raw, kommune
    X_oh, X_cat, y, y_raw, kommune_series = build_investment_features(df, horizon=HORIZON)
    X_oh = X_oh.reset_index(drop=True)
    X_cat = X_cat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_raw = y_raw.reset_index(drop=True)
    kommune_series = kommune_series.reset_index(drop=True)
    print(f"Etter preprocess: {len(X_oh):,} rader")
    print(f"  Features (one-hot): {len(X_oh.columns)}")
    print(f"  Features (rå-kat): {len(X_cat.columns)}")
    print(f"Target rank: min={y.min():.3f}, median={y.median():.3f}, max={y.max():.3f}")
    print(f"Raw avkastning: median={y_raw.median():.2f} %, "
          f"range=[{y_raw.min():.1f}, {y_raw.max():.1f}] %")

    # Tidsbasert split
    years = X_oh["aar"]
    split = make_time_split(years)
    X_tr_oh = X_oh.iloc[split.train]
    X_val_oh = X_oh.iloc[split.val]
    X_te_oh = X_oh.iloc[split.test]
    X_tr_cat = X_cat.iloc[split.train]
    X_val_cat = X_cat.iloc[split.val]
    X_te_cat = X_cat.iloc[split.test]
    y_tr = y.iloc[split.train]
    y_val = y.iloc[split.val]
    y_te = y.iloc[split.test]
    y_te_raw = y_raw.iloc[split.test]
    kommune_te = kommune_series.iloc[split.test]
    years_te = years.iloc[split.test]
    print(f"\nSplitt: train={len(X_tr_oh):,} ({int(years[split.train].min())}-{int(years[split.train].max())}), "
          f"val={len(X_val_oh):,}, test={len(X_te_oh):,} "
          f"({int(years_te.min())}-{int(years_te.max())})")

    # Sjekk hvilke engineerte features faktisk havnet i feature-listen
    engineered_keys = [
        "pris_lag1_kommune", "pris_lag2_kommune", "inntekt_yoy_pct",
        "pris_til_inntekt_ratio", "pris_relativ_norge_pct",
        "real_rente_pct", "momentum_3y_real_pct", "pris_relativ_norge_yoy",
    ]
    engineered = [c for c in X_oh.columns if c in engineered_keys]
    print(f"Engineerte features med: {len(engineered)}/{len(engineered_keys)}")

    # --- Søk hyperparametere ---
    cat_feats = categorical_feature_names()
    xgb_params, _ = cv_search("XGBoost", XGB_GRID, XGB_PARAMS,
                              make_xgb, fit_xgb, X_oh, y, years)
    lgbm_params, _ = cv_search("LightGBM", LGB_GRID, LGB_PARAMS,
                               make_lgbm, fit_lgbm, X_oh, y, years)
    cb_params, _ = cv_search("CatBoost", CB_GRID, CB_PARAMS,
                             make_catboost, fit_catboost, X_cat, y, years,
                             cat_features=cat_feats)

    # --- Tren endelige modeller ---
    print("\nTrener endelige modeller med beste hyperparameter...")
    t0 = time.time()
    xgb_model = make_xgb(xgb_params, n_estimators=3000)
    fit_xgb(xgb_model, X_tr_oh, y_tr, X_val_oh, y_val)
    t_xgb = time.time() - t0
    print(f"  XGBoost:  {t_xgb:.1f} s, beste iter = {xgb_model.best_iteration}")

    t0 = time.time()
    lgbm_model = make_lgbm(lgbm_params, n_estimators=3000)
    fit_lgbm(lgbm_model, X_tr_oh, y_tr, X_val_oh, y_val)
    t_lgbm = time.time() - t0
    print(f"  LightGBM: {t_lgbm:.1f} s, beste iter = {lgbm_model.best_iteration_}")

    t0 = time.time()
    cb_model = make_catboost(cb_params, n_estimators=3000)
    fit_catboost(cb_model, X_tr_cat, y_tr, X_val_cat, y_val, cat_features=cat_feats)
    t_cb = time.time() - t0
    print(f"  CatBoost: {t_cb:.1f} s, beste iter = {cb_model.best_iteration_}")

    # --- Velg ensemble-vekter på val-settet ---
    # Samme robuste valg som pris-modellen: prøv flere vekt-skjemaer, velg
    # det med lavest val-MAE, fall tilbake til beste enkeltmodell ved behov.
    val_pred_xgb = xgb_model.predict(X_val_oh)
    val_pred_lgbm = lgbm_model.predict(X_val_oh)
    val_pred_cb = cb_model.predict(X_val_cat)
    weights, scheme = choose_ensemble_weights(
        [val_pred_xgb, val_pred_lgbm, val_pred_cb],
        y_val.values,
    )
    print(f"\nEnsemble-vekter (skjema: {scheme}):")
    print(f"  XGBoost:  {weights[0]:.3f}")
    print(f"  LightGBM: {weights[1]:.3f}")
    print(f"  CatBoost: {weights[2]:.3f}")

    # --- Predikér på test ---
    pred_xgb = xgb_model.predict(X_te_oh)
    pred_lgbm = lgbm_model.predict(X_te_oh)
    pred_cb = cb_model.predict(X_te_cat)
    pred_ens = (
        weights[0] * pred_xgb
        + weights[1] * pred_lgbm
        + weights[2] * pred_cb
    )

    # --- Aggreger til kommune-år-nivå og evaluer ---
    print(f"\n=== Resultater på test-sett ({int(years_te.min())}-{int(years_te.max())}) ===")
    print("Aggregert til kommune-år-nivå (snitt over postnummer):\n")

    # Naiv baseline: predikér 0.5 (midten av rank-skalaen) for alle
    naive_pred = np.full_like(pred_ens, 0.5)
    naive_agg = aggregate_to_kommune_year(
        naive_pred, y_te.values, y_te_raw.values,
        kommune_te, years_te,
    )
    naive_m = evaluate("Naiv", naive_agg)

    xgb_agg = aggregate_to_kommune_year(
        pred_xgb, y_te.values, y_te_raw.values, kommune_te, years_te)
    xgb_m = evaluate("XGBoost", xgb_agg)

    lgbm_agg = aggregate_to_kommune_year(
        pred_lgbm, y_te.values, y_te_raw.values, kommune_te, years_te)
    lgbm_m = evaluate("LightGBM", lgbm_agg)

    cb_agg = aggregate_to_kommune_year(
        pred_cb, y_te.values, y_te_raw.values, kommune_te, years_te)
    cb_m = evaluate("CatBoost", cb_agg)

    ens_agg = aggregate_to_kommune_year(
        pred_ens, y_te.values, y_te_raw.values, kommune_te, years_te)
    ens_m = evaluate("Ensemble", ens_agg)

    print(f"\nEnsemble Spearman per år: {ens_m['spearman_per_year']}")
    print(f"Ensemble top-10 precision per år: {ens_m['top10_precision_per_year']}")
    print(f"Ensemble MAE_rank per år: "
          f"{ {y: round(v, 3) for y, v in ens_m['mae_rank_per_year'].items()} }")

    # Tolk Spearman-resultatet for leseren
    sp = ens_m["spearman_mean"]
    if sp > 0.20:
        print(f"\n  -> Sterkt signal (Spearman {sp:+.2f}). Modellen rangerer "
              "kommuner meningsfullt vs. faktisk avkastning.")
    elif sp > 0.10:
        print(f"\n  -> Meningsfullt signal (Spearman {sp:+.2f}). Realistisk "
              "for boligprediksjon — markedet har stor random-walk-komponent.")
    elif sp > 0.0:
        print(f"\n  -> Svakt positivt signal (Spearman {sp:+.2f}). Slår naiv, "
              "men bruk med varsomhet og kombiner med risiko-features i UI.")
    else:
        print(f"\n  -> Negativ Spearman ({sp:+.2f}). Modellen rangerer "
              "kommuner motsatt av virkeligheten. Sjekk features eller mål.")

    # Top features
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
    model_path = SAVED_DIR / "investment_ensemble.joblib"
    joblib.dump(ensemble, model_path)

    metadata = {
        "type": "investment_ensemble",
        "target": f"percentile_rank_av_real_prisstigning_aar+{HORIZON}",
        "target_horizon_years": HORIZON,
        "trained_at": datetime.now().isoformat(),
        "feature_cols_oh": list(X_oh.columns),
        "feature_cols_cat": list(X_cat.columns),
        "categorical_features": cat_feats,
        "n_train": int(len(X_tr_oh)),
        "n_val": int(len(X_val_oh)),
        "n_test": int(len(X_te_oh)),
        "n_test_kommune_year": int(len(ens_agg)),
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
            "naive": naive_m,
        },
        "library_versions": {
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "catboost": cb.__version__,
        },
        "device": "cuda",
    }
    meta_path = SAVED_DIR / "investment_ensemble_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nLagret: {model_path.name}")
    print(f"Lagret: {meta_path.name}")


if __name__ == "__main__":
    main()
