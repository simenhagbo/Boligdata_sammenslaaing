"""
Preprocessing-pipeline for pris-modellen.

Mål: pris_kvm_alle_kommune (rå NOK/kvm, log-transformeres i trener-en).

Engineerte features (7, alle look-ahead-sikre for prediksjon av pris ved år T):
  - pris_lag1_kommune          kvm-pris i år T-1
  - pris_lag2_kommune          kvm-pris i år T-2
  - pris_lag3_kommune          kvm-pris i år T-3
  - pris_lag5_kommune          kvm-pris i år T-5
  - inntekt_yoy_pct            inntekt: år-over-år endring
  - pris_til_inntekt_ratio_lag1 (pris_T-1 / inntekt_T-1)
  - pris_relativ_norge_lag1_pct kommune sin lag-pris som % av Norges-snitt det året

Hvorfor lag-versjoner: for pris-modellen ville bruk av nåværende kvm-pris
til å lage ratio/relativ-features vært direkte data-lekkasje. Lag-versjoner
unngår dette og fanger fortsatt det meste av signalet. Lag3 og lag5 fanger
opp lengre prissykler (boligmarkedet beveger seg ofte i 3-5-års bølger).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features_shared import (
    CATEGORICAL_COLS,
    ID_COLS,
    PRICE_LEAKAGE_COLS,
    encode_categoricals,
    national_avg_price_per_year,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legg til 7 engineerte features til df. Alle bygges på (kommune, aar)-nivå
    og merges tilbake til postnummer-radene."""
    # Bygg arbeidsbord på kommune-år-nivå
    base_cols = ["kommune_nr", "aar", "pris_kvm_alle_kommune",
                 "inntekt_etter_skatt", "antall_omsetninger_kommune"]
    base = (
        df[base_cols]
        .drop_duplicates(["kommune_nr", "aar"])
        .sort_values(["kommune_nr", "aar"])
        .copy()
    )

    # Lag-features for pris (T-1, T-2, T-3, T-5). Look-ahead-sikkert: bare bakover.
    # Lag3/lag5 fanger lengre prissykler enn lag1/lag2.
    grouped_pris = base.groupby("kommune_nr")["pris_kvm_alle_kommune"]
    base["pris_lag1_kommune"] = grouped_pris.shift(1)
    base["pris_lag2_kommune"] = grouped_pris.shift(2)
    base["pris_lag3_kommune"] = grouped_pris.shift(3)
    base["pris_lag5_kommune"] = grouped_pris.shift(5)

    # Lag av inntekt (mellomledd for de neste to beregningene)
    base["_inntekt_lag1"] = base.groupby("kommune_nr")["inntekt_etter_skatt"].shift(1)

    # År-over-år endring i inntekt (i prosent). Bruker inntekt ved T og T-1
    # — det er ikke lekkasje for pris-modellen siden inntekt ikke er målet.
    base["inntekt_yoy_pct"] = (
        (base["inntekt_etter_skatt"] - base["_inntekt_lag1"])
        / base["_inntekt_lag1"].replace(0, np.nan) * 100
    )

    # Pris/inntekt-ratio ved T-1 (lagget for å unngå lekkasje på pris-målet)
    base["pris_til_inntekt_ratio_lag1"] = (
        base["pris_lag1_kommune"] / base["_inntekt_lag1"].replace(0, np.nan)
    )

    # Kommunens lag-pris som % av Norges vektede snitt det forrige året
    norge = national_avg_price_per_year(df)
    # Skift Norge-snittet ett år frem så vi får T-1-snittet ved kolonnen for år T
    norge_lag = norge.copy()
    norge_lag["aar"] = norge_lag["aar"] + 1
    norge_lag = norge_lag.rename(columns={"norge_pris": "_norge_pris_lag1"})
    base = base.merge(norge_lag, on="aar", how="left")
    base["pris_relativ_norge_lag1_pct"] = (
        base["pris_lag1_kommune"] / base["_norge_pris_lag1"].replace(0, np.nan) * 100
    )

    # Merge nye kolonner tilbake til postnummer-nivå
    new_cols = [
        "pris_lag1_kommune",
        "pris_lag2_kommune",
        "pris_lag3_kommune",
        "pris_lag5_kommune",
        "inntekt_yoy_pct",
        "pris_til_inntekt_ratio_lag1",
        "pris_relativ_norge_lag1_pct",
    ]
    df = df.merge(
        base[["kommune_nr", "aar"] + new_cols],
        on=["kommune_nr", "aar"], how="left",
    )
    return df


def build_price_features(
    df: pd.DataFrame,
    target: str = "pris_kvm_alle_kommune",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Bygg X (features), X_cat (med rå-kategoriske for CatBoost) og y for pris-modellen.

    Returnerer tre objekter:
      X       — float32 med one-hot kategoriske (for XGBoost og LightGBM)
      X_cat   — samme rader, men beholder kategoriske som rå tekst (for CatBoost)
      y       — rå pris (log-transformeres i trener-en)
    """
    # Behold bare rader hvor målet er kjent
    df = df[df[target].notna()].copy()

    # Legg på engineerte features
    df = engineer_features(df)

    # Krev kun lag1 (det sterkeste signalet). Vi dropper dermed bare det
    # aller første året per kommune. lag2/lag3/lag5 får stå som NaN der de
    # mangler — XGBoost/LightGBM/CatBoost håndterer NaN nativt, så vi
    # beholder flere tidlige år enn om vi krevde lag2 også.
    # MERK: vi backfiller bevisst IKKE lag-kolonnene — å fylle lag1 for det
    # første året med neste års verdi ville satt lag1 = pris samme år, altså
    # direkte lekkasje av målet.
    df = df.dropna(subset=["pris_lag1_kommune"])

    # Drop alle pris-relaterte kolonner så modellen ikke leser målet
    # ut av en annen kolonne. Lag-pris er trygg og holdes utenfor lekkasje-listen.
    drop = set(ID_COLS + PRICE_LEAKAGE_COLS)
    drop.discard(target)  # behold målet i df, droppes som feature etterpå

    # CatBoost-variant: rå kategoriske kolonner beholdes.
    # CatBoost krever at kategoriske kolonner ikke har NaN/None — fyll med
    # sentinel-streng "ukjent" så manglende verdier blir sin egen kategori.
    df_cat = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].fillna("ukjent").astype(str)
    feature_cols_cat = [c for c in df_cat.columns if c not in drop and c != target]
    X_cat = df_cat[feature_cols_cat].copy()

    # XGBoost/LightGBM-variant: one-hot-encode kategoriske
    df_oh = encode_categoricals(df)
    feature_cols_oh = [c for c in df_oh.columns if c not in drop and c != target]
    X = df_oh[feature_cols_oh].astype(np.float32)

    y = df[target].astype(np.float32)
    return X, X_cat, y


def categorical_feature_names() -> list[str]:
    """Liste over kolonner som CatBoost skal behandle som kategoriske."""
    # CatBoost trenger eksplisitt liste — vi bruker samme som CATEGORICAL_COLS
    return list(CATEGORICAL_COLS)
