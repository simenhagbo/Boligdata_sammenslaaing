"""
Delte hjelpefunksjoner og konstanter for begge modell-pipelinene.

Innhold:
  - PRICE_LEAKAGE_COLS, ID_COLS, CATEGORICAL_COLS — kolonne-grupper som
    behandles felles på tvers av begge modellene
  - filter_quality — fjerner rader med dårlig data-kvalitet
  - winsorize — klipper ekstremverdier for stabile mål
  - encode_categoricals — one-hot-encoder tekst-kolonner
  - choose_ensemble_weights — velger robuste ensemble-vekter på valideringssettet

load_dataset re-eksporteres fra data_loader for bakoverkompatibilitet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Datatilgang sentraliseres i data_loader — én sannhetskilde for alle modeller.
from data_loader import load_dataset  # noqa: F401  (re-eksporteres for kallere)

# Kolonner som er målet selv eller direkte avledet av kvm-pris.
# Pris-modellen dropper alle; investerings-modellen beholder de fleste.
PRICE_LEAKAGE_COLS = [
    "pris_kvm_enebolig_kommune",
    "pris_kvm_smaahus_kommune",
    "pris_kvm_blokk_kommune",
    "pris_kvm_alle_kommune",
    "prisstigning_nominal_pct",
    "prisstigning_real_pct",
    "cagr_5aar_real_pct",
    "volatilitet_5aar",
    "sharpe_5aar",
]

# ID-, meta- og tekst-kolonner som aldri skal være features
ID_COLS = [
    "postnummer",
    "geometry",
    "kommune_nr",
    "poststedsnavn",
    "kommunenavn",
    "met_stasjon_id",
    "andel_utdanning_uoppgitt_kommune",  # 100 % NaN, ingen informasjon
    "data_kvalitet_flagg",               # meta-kolonne, ikke en feature
]

# Kategoriske kolonner som one-hot-kodes (for XGBoost/LightGBM).
# CatBoost takler dem nativt og får dem som rå-tekst i sin egen pipeline.
CATEGORICAL_COLS = [
    "naermeste_storby",
    "modal_boligtype_kommune",
]


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Fjern rader med dårlig kvalitet (data_kvalitet_flagg=1)."""
    # Disse er typisk eldre år eller postnummer i kommuner som har endret
    # kommunenummer — over 50 % manglende verdier forstyrrer modellen.
    if "data_kvalitet_flagg" not in df.columns:
        return df
    before = len(df)
    df = df[df["data_kvalitet_flagg"] == 0].copy()
    dropped_pct = (before - len(df)) / before
    if dropped_pct > 0:
        print(f"  Filtrert ut {before - len(df):,} rader "
              f"({dropped_pct:.1%}) med data_kvalitet_flagg=1")
    return df


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Klipp de mest ekstreme verdiene til 1./99.-persentilen."""
    # Brukes på investerings-target. Noen få ekstremavkastninger fra
    # boomperioder eller små kommuner kan ellers dominere treningen.
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Gjør tekst-kolonner om til 0/1-kolonner per kategori."""
    # dummy_na=False unngår en falsk "NaN-kategori" — XGBoost/LightGBM
    # takler NaN selv, så vi trenger ikke kode det som egen verdi.
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        # Lag dummies og slå sammen — drop original-kolonnen etterpå
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False, dtype=np.float32)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def kommune_year_aggregator(
    df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Dedupliser df til (kommune_nr, aar)-nivå, sortert kronologisk per kommune.

    Brukes som første steg i feature-engineering der vi beregner lag,
    year-over-year-endringer eller relative verdier på kommune-år-nivå
    før vi merger tilbake til postnummer-nivå.
    """
    # Velg kun nødvendige kolonner og dedupliser
    cols = ["kommune_nr", "aar", value_col]
    return (
        df[cols]
        .drop_duplicates(["kommune_nr", "aar"])
        .sort_values(["kommune_nr", "aar"])
        .copy()
    )


def choose_ensemble_weights(
    val_preds: list[np.ndarray],
    y_val: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Velg ensemble-vekter som gir lavest MAE på valideringssettet.

    Prøver fire skjemaer (likevekt, invers-MAE, invers-MAE², beste
    enkeltmodell) og returnerer det som vinner på val. Garanterer at
    ensemblet aldri er dårligere enn beste enkeltmodell — hvis de svake
    modellene drar ned, faller vi tilbake til den beste alene.
    """
    # MAE per base-modell på val
    maes = np.array([mean_absolute_error(y_val, p) for p in val_preds])
    n = len(val_preds)

    # Fire kandidat-vekt-skjemaer
    candidates: dict[str, np.ndarray] = {}
    candidates["equal"] = np.ones(n) / n
    inv = 1.0 / maes
    candidates["inv_mae"] = inv / inv.sum()
    inv2 = 1.0 / (maes ** 2)
    candidates["inv_mae2"] = inv2 / inv2.sum()
    best_single = np.zeros(n)
    best_single[int(maes.argmin())] = 1.0
    candidates["best_single"] = best_single

    # Velg skjemaet med lavest val-MAE på den vektede blandingen
    best_name, best_w, best_score = "equal", candidates["equal"], np.inf
    for name, w in candidates.items():
        blend = sum(wi * p for wi, p in zip(w, val_preds))
        score = mean_absolute_error(y_val, blend)
        if score < best_score:
            best_name, best_w, best_score = name, w, score
    return best_w, best_name


def national_avg_price_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Beregn omsetnings-vektet kvm-pris for hele Norge per år."""
    # Dedupliser til (kommune, aar) for å unngå at hver kommune teller flere ganger
    base = (
        df[["kommune_nr", "aar", "pris_kvm_alle_kommune",
            "antall_omsetninger_kommune"]]
        .drop_duplicates(["kommune_nr", "aar"]).copy()
    )
    # Hopp over rader uten pris — de bidrar verken til snitt eller vekt
    base = base.dropna(subset=["pris_kvm_alle_kommune"])
    base["w"] = base["antall_omsetninger_kommune"].fillna(0)
    base["pw"] = base["pris_kvm_alle_kommune"] * base["w"]

    # Vektet snitt per år: sum(pris × omsetninger) / sum(omsetninger)
    per_year = base.groupby("aar").agg(
        pw_sum=("pw", "sum"),
        w_sum=("w", "sum"),
    ).reset_index()
    per_year["norge_pris"] = per_year["pw_sum"] / per_year["w_sum"].replace(0, np.nan)
    return per_year[["aar", "norge_pris"]]
