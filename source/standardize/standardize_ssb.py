"""
Parser SSB-tabellene fra JSON-stat til Parquet med postnummer-nøkkel.

SSB er på kommunenivå — alle postnummer i samme kommune får samme verdi.
JSON-stat-formatet lagrer verdier som en flat liste der rekkefølgen tilsvarer
det kartesiske produktet av dimensjonene. itertools.product gjenoppretter
indeksene.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "ssb"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
OUT_DIR = STD_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_jsonstat(path: Path) -> pd.DataFrame:
    """Flate ut JSON-stat2 til en DataFrame der hver rad er én dimensjonskombinasjon."""
    with open(path, encoding="utf-8") as f:
        js = json.load(f)

    dims = js["id"]
    labels = js["dimension"]
    values = js["value"]

    # En liste av (kode, etikett)-par per dimensjon. dict bevarer
    # insertion order, som matcher SSBs JSON-stat-rekkefølge.
    cats = [list(labels[d]["category"]["label"].items()) for d in dims]

    rows = []
    for combo, val in zip(itertools.product(*cats), values):
        row = {dim: code for dim, (code, _) in zip(dims, combo)}
        row["value"] = val
        rows.append(row)

    return pd.DataFrame(rows)


def load_kommune_mapping() -> pd.DataFrame:
    mapping_path = STD_DIR / "postnummer_kommune_mapping.parquet"
    if not mapping_path.exists():
        raise FileNotFoundError("Kjør standardize_kartverket.py før standardize_ssb.py")
    return pd.read_parquet(mapping_path)[["postnummer", "kommune_nr"]]


def _find_col(df: pd.DataFrame, name_lower: str) -> str:
    for c in df.columns:
        if c.lower() == name_lower:
            return c
    raise KeyError(f"Fant ikke kolonne '{name_lower}' i {list(df.columns)}")


# --- BygnType-fordeling og total fra tabell 06265 ---

def standardize_boliger_per_type(mapping: pd.DataFrame) -> pd.DataFrame:
    """Totaltall, modal og andeler per BygnType per kommune (tabell 06265)."""
    src = RAW_DIR / "boliger_per_type_06265.json"
    cols_out = [
        "postnummer", "antall_boliger", "modal_boligtype_kommune",
        "andel_bygntype_01_kommune", "andel_bygntype_02_kommune",
        "andel_bygntype_03_kommune", "andel_bygntype_04_kommune",
        "andel_bygntype_05_kommune",
    ]
    if not src.exists():
        return mapping.assign(**{c: pd.NA for c in cols_out[1:]})[cols_out]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "bygntype")

    # Behold bare hovedtypene 01-05 (drop 999=annet) for andeler
    main_types = ["01", "02", "03", "04", "05"]
    main = df[df[type_col].isin(main_types)].copy()

    # Pivot til wide: én kolonne per BygnType
    wide = main.pivot_table(
        index=region_col, columns=type_col, values="value", aggfunc="sum"
    ).fillna(0)
    # Sørg for at alle fem typer finnes som kolonner
    for t in main_types:
        if t not in wide.columns:
            wide[t] = 0.0
    wide = wide[main_types]

    totals = wide.sum(axis=1).rename("antall_boliger")
    andeler = wide.div(totals.replace(0, np.nan), axis=0)
    andeler.columns = [f"andel_bygntype_{t}_kommune" for t in main_types]
    modal = wide.idxmax(axis=1).rename("modal_boligtype_kommune")

    result = pd.concat([totals, modal, andeler], axis=1).reset_index()
    result = result.rename(columns={region_col: "kommune_nr"})
    result["kommune_nr"] = result["kommune_nr"].astype(str).str.zfill(4)

    merged = mapping.merge(result, on="kommune_nr", how="left")
    return merged[cols_out]


# --- Folkemengde (07459) og inntekt (12558) ---

def standardize_folkemengde(mapping: pd.DataFrame) -> pd.DataFrame:
    """Folkemengde per kommune (tabell 07459) — summerer over Kjonn og Alder."""
    src = RAW_DIR / "folkemengde_07459.json"
    if not src.exists():
        return mapping.assign(befolkning=pd.NA)[["postnummer", "befolkning"]]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")

    totals = df.groupby(region_col)["value"].sum().reset_index()
    totals = totals.rename(columns={region_col: "kommune_nr", "value": "befolkning"})
    totals["kommune_nr"] = totals["kommune_nr"].astype(str).str.zfill(4)

    merged = mapping.merge(totals, on="kommune_nr", how="left")
    return merged[["postnummer", "befolkning"]]


def standardize_inntekt(mapping: pd.DataFrame) -> pd.DataFrame:
    """Inntekt etter skatt per kommune (tabell 12558, InntektSkatt=00S)."""
    src = RAW_DIR / "inntekt_12558.json"
    if not src.exists():
        return mapping.assign(inntekt_etter_skatt=pd.NA)[["postnummer", "inntekt_etter_skatt"]]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")

    df = df.rename(columns={region_col: "kommune_nr", "value": "inntekt_etter_skatt"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df = df[["kommune_nr", "inntekt_etter_skatt"]].dropna()
    df = df.drop_duplicates("kommune_nr", keep="first")

    merged = mapping.merge(df, on="kommune_nr", how="left")
    return merged[["postnummer", "inntekt_etter_skatt"]]


# --- Priser fra tabell 06035 ---

# Mapping Boligtype-kode → kolonne-suffiks
_PRIS_TYPE_LABEL = {"01": "enebolig", "02": "smaahus", "03": "blokk"}


def standardize_priser(mapping: pd.DataFrame) -> pd.DataFrame:
    """Kvm-pris og antall omsetninger per kommune (tabell 06035)."""
    src = RAW_DIR / "priser_06035.json"
    cols_out = [
        "postnummer",
        "pris_kvm_enebolig_kommune", "pris_kvm_smaahus_kommune", "pris_kvm_blokk_kommune",
        "pris_kvm_alle_kommune", "antall_omsetninger_kommune",
    ]
    if not src.exists():
        return mapping.assign(**{c: pd.NA for c in cols_out[1:]})[cols_out]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "boligtype")
    content_col = _find_col(df, "contentscode")

    # Pivot til (region, type) × ContentsCode
    pivoted = df.pivot_table(
        index=[region_col, type_col], columns=content_col, values="value", aggfunc="sum"
    ).reset_index()

    # Plukk ut kvm-pris og omsetninger som egne kolonner
    pivoted = pivoted.rename(columns={"KvPris": "kvpris", "Omsetninger": "omsetninger"})

    # Beregn vektet gj.snitt pris over typene, vekt = omsetninger
    pivoted["pris_x_vekt"] = pivoted["kvpris"] * pivoted["omsetninger"]
    alle = pivoted.groupby(region_col).agg(
        pris_x_vekt_sum=("pris_x_vekt", "sum"),
        omsetninger_sum=("omsetninger", "sum"),
    ).reset_index()
    alle["pris_kvm_alle_kommune"] = (
        alle["pris_x_vekt_sum"] / alle["omsetninger_sum"].replace(0, np.nan)
    )
    alle = alle.rename(columns={region_col: "kommune_nr", "omsetninger_sum": "antall_omsetninger_kommune"})
    alle["kommune_nr"] = alle["kommune_nr"].astype(str).str.zfill(4)
    alle = alle[["kommune_nr", "pris_kvm_alle_kommune", "antall_omsetninger_kommune"]]

    # Per-type kolonner via pivot
    per_type = pivoted.pivot_table(
        index=region_col, columns=type_col, values="kvpris", aggfunc="first"
    )
    per_type = per_type.rename(columns={k: f"pris_kvm_{v}_kommune" for k, v in _PRIS_TYPE_LABEL.items()})
    # Garanter alle tre kolonner finnes
    for label in _PRIS_TYPE_LABEL.values():
        col = f"pris_kvm_{label}_kommune"
        if col not in per_type.columns:
            per_type[col] = np.nan
    per_type = per_type.reset_index().rename(columns={region_col: "kommune_nr"})
    per_type["kommune_nr"] = per_type["kommune_nr"].astype(str).str.zfill(4)

    combined = per_type.merge(alle, on="kommune_nr", how="outer")
    merged = mapping.merge(combined, on="kommune_nr", how="left")
    return merged[cols_out]


# --- Byggeår og bruksareal (06266, 06513) ---

# BygnAr-kode → (periode-label, midtpunkt-år for median-beregning)
_BYGGEAAR_BUCKETS = {
    "01": ("for1946", 1880),
    "02": ("for1946", 1910),
    "03": ("for1946", 1930),
    "04": ("for1946", 1943),
    "06": ("1946_1970", 1953),
    "07": ("1946_1970", 1965),
    "08": ("1971_1990", 1975),
    "09": ("1971_1990", 1985),
    "10": ("1991_2010", 1995),
    "11": ("1991_2010", 2005),
    "12": ("etter2010", 2015),
    "13": ("etter2010", 2022),
    # 99 = ukjent — ekskluderes
}
_BYGGEAAR_PERIODER = ["for1946", "1946_1970", "1971_1990", "1991_2010", "etter2010"]

# BruksAreal-kode → (gruppe-label, midtpunkt-kvm)
_BRUKSAREAL_BUCKETS = {
    "1": ("under60", 20),
    "2": ("under60", 35),
    "3": ("under60", 45),
    "4": ("under60", 55),
    "5": ("60_99", 70),
    "6": ("60_99", 90),
    "50": ("100_159", 110),
    "51": ("100_159", 130),
    "52": ("100_159", 150),
    "53": ("160_249", 180),
    "54": ("160_249", 225),
    "55": ("over250", 275),
    "56": ("over250", 325),
    "57": ("over250", 400),
}
_BRUKSAREAL_GRUPPER = ["under60", "60_99", "100_159", "160_249", "over250"]


def _vektet_median(midtpunkter: np.ndarray, vekter: np.ndarray) -> float:
    """Median fra sortert (midtpunkt, vekt)-serie."""
    total = vekter.sum()
    if total == 0:
        return np.nan
    kumulativ = np.cumsum(vekter)
    idx = np.searchsorted(kumulativ, total / 2)
    idx = min(idx, len(midtpunkter) - 1)
    return float(midtpunkter[idx])


def _andeler_og_median(
    df: pd.DataFrame,
    region_col: str,
    bucket_col: str,
    buckets: dict[str, tuple[str, float]],
    gruppe_navn: list[str],
    prefix: str,
    median_col_name: str,
) -> pd.DataFrame:
    """
    Beregn andeler per gruppe + vektet median per kommune.

    `buckets` mapper kode i `bucket_col` til (gruppenavn, midtpunkt).
    `prefix` brukes som kolonne-prefiks for andelene.
    """
    work = df[df[bucket_col].isin(buckets)].copy()
    work["gruppe"] = work[bucket_col].map(lambda c: buckets[c][0])
    work["midtpunkt"] = work[bucket_col].map(lambda c: buckets[c][1])

    # Andeler per gruppe
    grupped = work.groupby([region_col, "gruppe"])["value"].sum().unstack(fill_value=0)
    for g in gruppe_navn:
        if g not in grupped.columns:
            grupped[g] = 0.0
    grupped = grupped[gruppe_navn]
    totals = grupped.sum(axis=1)
    andeler = grupped.div(totals.replace(0, np.nan), axis=0)
    andeler.columns = [f"{prefix}_{g}_kommune" for g in gruppe_navn]

    # Vektet median per kommune. Sortér etter midtpunkt for korrekt kumulering.
    work_sorted = work.sort_values([region_col, "midtpunkt"])
    medians = work_sorted.groupby(region_col).apply(
        lambda g: _vektet_median(g["midtpunkt"].values, g["value"].values),
        include_groups=False,
    ).rename(median_col_name)

    return pd.concat([andeler, medians], axis=1).reset_index()


def standardize_byggeaar(mapping: pd.DataFrame) -> pd.DataFrame:
    """Byggeår-fordeling per kommune (tabell 06266) — fem perioder + median."""
    src = RAW_DIR / "byggeaar_06266.json"
    cols_out = ["postnummer"] + [f"andel_byggeaar_{p}_kommune" for p in _BYGGEAAR_PERIODER] + ["median_byggeaar_kommune"]
    if not src.exists():
        return mapping.assign(**{c: pd.NA for c in cols_out[1:]})[cols_out]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    bucket_col = _find_col(df, "bygnar")

    result = _andeler_og_median(
        df, region_col, bucket_col,
        _BYGGEAAR_BUCKETS, _BYGGEAAR_PERIODER,
        prefix="andel_byggeaar",
        median_col_name="median_byggeaar_kommune",
    )
    result = result.rename(columns={region_col: "kommune_nr"})
    result["kommune_nr"] = result["kommune_nr"].astype(str).str.zfill(4)
    merged = mapping.merge(result, on="kommune_nr", how="left")
    return merged[cols_out]


def standardize_bruksareal(mapping: pd.DataFrame) -> pd.DataFrame:
    """Bruksareal-fordeling per kommune (tabell 06513) — fem grupper + median."""
    src = RAW_DIR / "bruksareal_06513.json"
    cols_out = ["postnummer"] + [f"andel_areal_{g}_kommune" for g in _BRUKSAREAL_GRUPPER] + ["median_bruksareal_kommune"]
    if not src.exists():
        return mapping.assign(**{c: pd.NA for c in cols_out[1:]})[cols_out]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    bucket_col = _find_col(df, "bruksareal")

    result = _andeler_og_median(
        df, region_col, bucket_col,
        _BRUKSAREAL_BUCKETS, _BRUKSAREAL_GRUPPER,
        prefix="andel_areal",
        median_col_name="median_bruksareal_kommune",
    )
    result = result.rename(columns={region_col: "kommune_nr"})
    result["kommune_nr"] = result["kommune_nr"].astype(str).str.zfill(4)
    merged = mapping.merge(result, on="kommune_nr", how="left")
    return merged[cols_out]


def main() -> None:
    print("=== Standardisering: SSB ===")
    mapping = load_kommune_mapping()

    parts = [
        standardize_boliger_per_type(mapping),
        standardize_folkemengde(mapping),
        standardize_inntekt(mapping),
        standardize_priser(mapping),
        standardize_byggeaar(mapping),
        standardize_bruksareal(mapping),
    ]

    df = mapping.copy()
    for p in parts:
        df = df.merge(p, on="postnummer", how="left")

    out = OUT_DIR / "ssb_bolig_demografi.parquet"
    df.to_parquet(out, index=False)
    print(f"  ssb_bolig_demografi.parquet: {len(df)} rader, {len(df.columns)} kolonner")
    # Vis dekning på de mest interessante kolonnene
    for col in [
        "antall_boliger", "modal_boligtype_kommune", "befolkning",
        "inntekt_etter_skatt", "pris_kvm_alle_kommune",
        "median_byggeaar_kommune", "median_bruksareal_kommune",
    ]:
        if col in df.columns:
            print(f"    {col}: {df[col].notna().mean():.1%}")
    print("=== SSB standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
