"""
Preprocessing-pipeline for investerings-modellen.

Mål: percentile rank av kommunens fremtidige real prisstigning (1 år frem)
     innen samme år. Skala 0-1 der 1 = beste avkastning den året.

Hvorfor ranking i stedet for absolutt avkastning:
- Eliminerer regime-skift-problemet. Med absolutt mål var den gamle
  modellen sjanseløs på 2022-2024 fordi den aldri hadde sett tilsvarende
  inflasjon i treningssettet. Rang er alltid 0-1 uansett hvilket regime.
- Fanger essensen i "investeringsattraktivitet": hvilke kommuner gjør det
  bedre enn andre dette året.
- Direkte brukbart i UI: høy predikert rank = vis denne kommunen øverst.

Engineerte features (10 totalt):
  Felles med pris-modellen (6):
    - pris_lag1_kommune, pris_lag2_kommune
    - inntekt_yoy_pct
    - pris_til_inntekt_ratio (T, ikke lag — trygt siden mål er forward)
    - pris_relativ_norge_pct (T, ikke lag)

  Investerings-spesifikke (4):
    - real_rente_pct          styringsrente − kpi_endring_pct
    - momentum_3y_real_pct    kumulativ real prisendring T-2 til T
    - momentum_5y_real_pct    kumulativ real prisendring T-4 til T
    - pris_relativ_norge_yoy  år-over-år endring i relativ-til-Norge
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features_shared import (
    CATEGORICAL_COLS,
    ID_COLS,
    encode_categoricals,
    national_avg_price_per_year,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legg til 10 engineerte features til df. Alle look-ahead-sikre for
    prediksjon av forward-avkastning fra år T."""
    # Bygg arbeidsbord på kommune-år-nivå
    base_cols = [
        "kommune_nr", "aar",
        "pris_kvm_alle_kommune", "inntekt_etter_skatt",
        "antall_omsetninger_kommune",
        "prisstigning_real_pct",
        "styringsrente", "kpi_endring_pct",
    ]
    base = (
        df[base_cols]
        .drop_duplicates(["kommune_nr", "aar"])
        .sort_values(["kommune_nr", "aar"])
        .copy()
    )

    # 1-2. Lag-features for kvm-pris (T-1 og T-2)
    grouped_pris = base.groupby("kommune_nr")["pris_kvm_alle_kommune"]
    base["pris_lag1_kommune"] = grouped_pris.shift(1)
    base["pris_lag2_kommune"] = grouped_pris.shift(2)

    # 3. Inntekt år-over-år (i prosent)
    base["_inntekt_lag1"] = base.groupby("kommune_nr")["inntekt_etter_skatt"].shift(1)
    base["inntekt_yoy_pct"] = (
        (base["inntekt_etter_skatt"] - base["_inntekt_lag1"])
        / base["_inntekt_lag1"].replace(0, np.nan) * 100
    )

    # 4. Pris/inntekt-ratio ved T (nåværende — ikke lekkasje siden mål er forward)
    base["pris_til_inntekt_ratio"] = (
        base["pris_kvm_alle_kommune"] / base["inntekt_etter_skatt"].replace(0, np.nan)
    )

    # 5. Kommunens pris som % av nasjonalt snitt det året
    norge = national_avg_price_per_year(df)
    base = base.merge(norge, on="aar", how="left")
    base["pris_relativ_norge_pct"] = (
        base["pris_kvm_alle_kommune"] / base["norge_pris"].replace(0, np.nan) * 100
    )

    # 6. Real rente (regime-indikator: høy positiv = stram pengepolitikk)
    base["real_rente_pct"] = base["styringsrente"] - base["kpi_endring_pct"]

    # 7. Momentum: kumulativ real prisendring siste 3 år, ender ved T
    grouped_pst = base.groupby("kommune_nr")["prisstigning_real_pct"]
    pst = [grouped_pst.shift(k) / 100 for k in range(5)]  # T, T-1, ..., T-4
    base["momentum_3y_real_pct"] = (
        (1 + pst[0]) * (1 + pst[1]) * (1 + pst[2]) - 1
    ) * 100

    # 8. Momentum over 5 år (T-4 til T). Fanger lengre prissykler enn 3-års.
    base["momentum_5y_real_pct"] = (
        (1 + pst[0]) * (1 + pst[1]) * (1 + pst[2]) * (1 + pst[3]) * (1 + pst[4]) - 1
    ) * 100

    # 9. År-over-år endring i pris_relativ_norge (fanger om kommunen
    # henger etter eller drar fra nasjonalt nivå)
    base["_relativ_norge_lag1"] = base.groupby("kommune_nr")["pris_relativ_norge_pct"].shift(1)
    base["pris_relativ_norge_yoy"] = (
        base["pris_relativ_norge_pct"] - base["_relativ_norge_lag1"]
    )

    # Merge alle nye kolonner tilbake til postnummer-nivå
    new_cols = [
        "pris_lag1_kommune",
        "pris_lag2_kommune",
        "inntekt_yoy_pct",
        "pris_til_inntekt_ratio",
        "pris_relativ_norge_pct",
        "real_rente_pct",
        "momentum_3y_real_pct",
        "momentum_5y_real_pct",
        "pris_relativ_norge_yoy",
    ]
    df = df.merge(
        base[["kommune_nr", "aar"] + new_cols],
        on=["kommune_nr", "aar"], how="left",
    )
    return df


def make_ranking_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Lag percentile-rank av fremtidig real prisstigning, innen hvert år."""
    # Bygg forward-avkastning på (kommune, aar)-nivå
    base = (
        df[["kommune_nr", "aar", "prisstigning_real_pct"]]
        .drop_duplicates(["kommune_nr", "aar"])
        .sort_values(["kommune_nr", "aar"])
        .copy()
    )
    # Shift -horizon: gir prisstigning ved (aar + horizon) innen kommune
    base["_fremtidig"] = base.groupby("kommune_nr")["prisstigning_real_pct"].shift(-horizon)

    # Per år: rank kommuner etter forward-avkastning, normalisert til [0, 1]
    base["rank_pct"] = base.groupby("aar")["_fremtidig"].rank(pct=True, method="average")

    # Merge tilbake til full df
    merged = df[["kommune_nr", "aar"]].merge(
        base[["kommune_nr", "aar", "rank_pct"]],
        on=["kommune_nr", "aar"], how="left",
    )
    merged.index = df.index
    return merged["rank_pct"]


def make_forward_return_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Lag rå forward-avkastning. Brukes til evaluering og top-N precision."""
    # Vi trenger den faktiske avkastningen i tillegg til rank-en for å
    # rapportere meningsfulle metrics som top-10 precision.
    base = (
        df[["kommune_nr", "aar", "prisstigning_real_pct"]]
        .drop_duplicates(["kommune_nr", "aar"])
        .sort_values(["kommune_nr", "aar"])
        .copy()
    )
    base["_fremtidig"] = base.groupby("kommune_nr")["prisstigning_real_pct"].shift(-horizon)
    merged = df[["kommune_nr", "aar"]].merge(
        base[["kommune_nr", "aar", "_fremtidig"]],
        on=["kommune_nr", "aar"], how="left",
    )
    merged.index = df.index
    return merged["_fremtidig"]


def build_investment_features(
    df: pd.DataFrame,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Bygg features og mål for investerings-modellen.

    Returnerer fem objekter:
      X        — float32 med one-hot kategoriske (for XGBoost og LightGBM)
      X_cat    — samme rader, men beholder kategoriske som rå tekst (CatBoost)
      y        — percentile rank (0-1) av forward-avkastning (mål for modellen)
      y_raw    — rå forward-avkastning i prosent (kun for evaluering)
      kommune  — kommune_nr per rad (brukes til kommune-nivå-evaluering)
    """
    # Legg på engineerte features først (trenger prisstigning_real_pct osv.)
    df = engineer_features(df)

    # Bygg mål og rå avkastning
    rank_all = make_ranking_target(df, horizon=horizon)
    raw_all = make_forward_return_target(df, horizon=horizon)

    # Behold kun rader med kjent mål og hvor lag1 finnes. Vi krever ikke
    # lag2/momentum — de kan stå som NaN (tre-modellene håndterer det), så
    # vi beholder flere tidlige år. Ingen lekkasje her uansett, siden målet
    # er forward (T+1) og ikke knyttet til lag-prisene.
    mask = rank_all.notna() & df["pris_lag1_kommune"].notna()
    df = df[mask].copy()
    y = rank_all[mask].astype(np.float32)
    y_raw = raw_all[mask].astype(np.float32)
    # Returner kommune_nr separat siden den droppes fra feature-matrisene
    kommune = df["kommune_nr"].copy()

    # Drop ID-kolonner + prisstigning_real_pct (brukt til å bygge mål — fjern
    # for å tvinge modellen til å bruke fundamentale features framfor momentum)
    drop = set(ID_COLS + ["prisstigning_real_pct"])

    # CatBoost-variant: beholdes med rå kategoriske.
    # CatBoost takler ikke NaN i kategoriske felter — fyll med sentinel-streng.
    df_cat = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].fillna("ukjent").astype(str)
    feature_cols_cat = [c for c in df_cat.columns if c not in drop]
    X_cat = df_cat[feature_cols_cat].copy()

    # XGBoost/LightGBM-variant: one-hot-encode kategoriske
    df_oh = encode_categoricals(df)
    feature_cols_oh = [c for c in df_oh.columns if c not in drop]
    X = df_oh[feature_cols_oh].astype(np.float32)

    return X, X_cat, y, y_raw, kommune


def categorical_feature_names() -> list[str]:
    """Liste over kolonner som CatBoost skal behandle som kategoriske."""
    return list(CATEGORICAL_COLS)
