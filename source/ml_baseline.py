"""
Enkel ML-baseline: forutsi pris_kvm_alle_kommune fra de andre kolonnene.

Målet er å validere at datasettet faktisk inneholder signal — at en
modell kan slå en naiv predikering (median av treningsmål). XGBoost ble
valgt fordi det er bransjestandard for tabulær ML, takler manglende verdier
nativt, og gir feature importance gratis. Trening kjøres på GPU (CUDA) via
parameteren device='cuda'.

Splitt: tidsbasert (train ≤ 2018, val 2019-2020, test 2021-2024). Random
split ville lekket fremtidig prisutvikling inn i treningen.

Lekkasje-håndtering: kolonner som er direkte avledet av målet (de andre
pris_kvm_*-variantene, prisstigning, CAGR, volatilitet, Sharpe) droppes
før trening.

Bruk:
  python source/ml_baseline.py
"""

import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost varsler når predict-data ligger på CPU mens boosteren ligger på
# GPU. På små test-sett (~10k rader) er den ytelses-forskjellen mikro-
# sekunder, så vi filtrerer den informasjons-meldingen for å holde
# output ren.
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

FINAL = Path(__file__).parents[1] / "data" / "processed_data" / "final" / "boligdata_final.parquet"

TARGET = "pris_kvm_alle_kommune"

# Kolonner som er direkte avledet av målet. Må fjernes for å unngå at
# modellen "jukser" ved å lese målet ut av en annen kolonne.
LEAKAGE_COLS = [
    "pris_kvm_enebolig_kommune",
    "pris_kvm_smaahus_kommune",
    "pris_kvm_blokk_kommune",
    "prisstigning_nominal_pct",
    "prisstigning_real_pct",
    "cagr_5aar_real_pct",
    "volatilitet_5aar",
    "sharpe_5aar",
]

# Ikke-numeriske og ID-aktige kolonner. Vi holder baselinen til rene
# numeriske features for enkelhets skyld; en raffinert versjon kunne
# one-hot-kodet f.eks. naermeste_storby og modal_boligtype_kommune.
ID_COLS = [
    "postnummer",
    "geometry",
    "kommune_nr",
    "poststedsnavn",
    "kommunenavn",
    "naermeste_storby",
    "modal_boligtype_kommune",
    "met_stasjon_id",
    "andel_utdanning_uoppgitt_kommune",  # 100 % NaN, ingen informasjon
    "data_kvalitet_flagg",               # meta-kolonne, ikke en feature
]


def _print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Beregn tre standardmetrikker og skriv dem ut på én ryddig linje.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:8s}  MAE = {mae:>8,.0f} NOK/kvm   RMSE = {rmse:>8,.0f}   R² = {r2:>6.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    # Avbryt tidlig hvis pipelinen ikke er kjørt ennå — vi har ikke noe å trene på.
    if not FINAL.exists():
        print(f"FEIL: {FINAL} finnes ikke. Kjør 'python source/pipeline.py' først.")
        return

    # Header + versjons-info slik at loggen viser hvilken oppsetning vi kjørte med.
    print(f"=== ML-baseline: forutsi {TARGET} ===\n")
    print(f"XGBoost versjon: {xgb.__version__}")

    # Les hele det ferdige datasettet inn som GeoDataFrame (geometry-kolonnen
    # blir droppet senere som feature; vi trenger den ikke her).
    df = gpd.read_parquet(FINAL)
    print(f"Datasett: {df.shape[0]:,} rader × {df.shape[1]} kolonner")

    # Behold bare rader hvor målet er kjent — SSB har ikke pris-tall for alle år.
    df = df[df[TARGET].notna()].copy()
    print(f"Rader med kjent {TARGET}: {len(df):,}\n")

    # Tidsbasert splitt: train på fortiden, valider på de neste 2 årene,
    # test på fremtiden. Hardkodede grenser gir reproduserbarhet.
    train = df[df["aar"] <= 2018]
    val = df[(df["aar"] >= 2019) & (df["aar"] <= 2020)]
    test = df[df["aar"] >= 2021]
    print("Tidsbasert splitt:")
    print(f"  Train: {len(train):>6,} rader ({train['aar'].min()}-{train['aar'].max()})")
    print(f"  Val:   {len(val):>6,} rader ({val['aar'].min()}-{val['aar'].max()})")
    print(f"  Test:  {len(test):>6,} rader ({test['aar'].min()}-{test['aar'].max()})\n")

    # Bygg feature-listen: alt utenom mål, lekkasje-kolonner og ID-er.
    drop_cols = set([TARGET, *LEAKAGE_COLS, *ID_COLS])
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Antall features: {len(feature_cols)}")

    # Konverter til float32 numpy-arrays — XGBoost foretrekker det og det
    # halverer minnebruken sammenlignet med pandas-float64.
    X_train = train[feature_cols].astype(np.float32).to_numpy()
    y_train = train[TARGET].astype(np.float32).to_numpy()
    X_val = val[feature_cols].astype(np.float32).to_numpy()
    y_val = val[TARGET].astype(np.float32).to_numpy()
    X_test = test[feature_cols].astype(np.float32).to_numpy()
    y_test = test[TARGET].astype(np.float32).to_numpy()

    # Sett opp regressoren. tree_method='hist' er den GPU-kompatible
    # varianten i XGBoost 3.x. early_stopping_rounds gir automatisk
    # regulariseringseffekt — vi stopper når val-MAE ikke forbedres på 20 trær.
    print("\nTrener XGBoost på GPU (device='cuda')...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        device="cuda",
        tree_method="hist",
        early_stopping_rounds=20,
        eval_metric="mae",
        random_state=42,
    )

    # Tren modellen og mål veggklokke-tid — viser konkret at GPU-trening er rask.
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0

    # Rapporter hvor trening faktisk skjedde, hvor lang tid det tok, og hvor
    # early stopping fant beste iterasjon.
    used_device = model.get_xgb_params().get("device", "cpu")
    print(f"  Enhet:           {used_device}")
    print(f"  Treningstid:     {elapsed:.1f} s")
    print(f"  Beste iterasjon: {model.best_iteration} / {model.n_estimators}")

    # Prediker test-settet med modellen, og lag en naiv "gjett medianen"-
    # baseline til sammenligning.
    pred_test = model.predict(X_test)
    naive_pred = np.full_like(y_test, np.median(y_train))

    # Skriv ut metrikker for begge — naiv først så modellen, så de er enkle å sammenligne.
    print("\n=== Resultater på test-sett (2021-2024) ===")
    naive_m = _print_metrics("Naiv:",  y_test, naive_pred)
    model_m = _print_metrics("Modell:", y_test, pred_test)
    forbedring = (1 - model_m["mae"] / naive_m["mae"]) * 100
    print(f"\n  -> Modellen reduserer MAE med {forbedring:.0f} % vs. naiv prediksjon.")

    # Tolk R²-en for leseren — terskler valgt etter konvensjon, ikke matematikk.
    if model_m["r2"] > 0.5:
        print("  -> R² > 0.5: datasettet inneholder klart signal for prisprediksjon.")
    elif model_m["r2"] > 0.2:
        print("  -> R² mellom 0.2 og 0.5: noe signal, men begrenset prediksjonskraft.")
    else:
        print("  -> R² under 0.2: lite signal. Vurder å undersøke features eller mål.")

    # Top 15 features sortert på gain-importance. Stolpe-lengden er
    # normalisert mot toppscoren slik at relative bidrag er lett å lese.
    print("\n=== Top 15 features (XGBoost importance) ===")
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(15)
    for name, score in importance.items():
        bar = "#" * int(score * 30 / importance.max())
        print(f"  {score:.4f}  {name:38s} {bar}")


if __name__ == "__main__":
    main()
