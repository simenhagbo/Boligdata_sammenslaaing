"""
Parser SSB-rådata til ett tidsserie-Parquet på (postnummer, år)-format.

Alle postnummer i samme kommune deler SSB-verdiene for et gitt år — det er
en bevisst forenkling siden SSB ikke publiserer på postnummer-nivå.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "ssb"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)


# ----- JSON-stat parsing -----

def parse_jsonstat(path: Path) -> pd.DataFrame:
    """Flat ut JSON-stat2 — én rad per dimensjonskombinasjon."""
    with open(path, encoding="utf-8") as f:
        js = json.load(f)

    dims = js["id"]
    labels = js["dimension"]
    values = js["value"]
    cats = [list(labels[d]["category"]["label"].items()) for d in dims]

    rows = []
    for combo, val in zip(itertools.product(*cats), values):
        row = {dim: code for dim, (code, _) in zip(dims, combo)}
        row["value"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def parse_jsonstat_chunks(prefix: str) -> pd.DataFrame:
    """Les alle `{prefix}_partNN.json`-filer og slå sammen."""
    files = sorted(RAW_DIR.glob(f"{prefix}_part*.json"))
    if not files:
        # Fallback: hvis tabellen ikke ble chunked, prøv enkeltfil
        single = RAW_DIR / f"{prefix}.json"
        if single.exists():
            return parse_jsonstat(single)
        return pd.DataFrame()
    return pd.concat([parse_jsonstat(f) for f in files], ignore_index=True)


def _find_col(df: pd.DataFrame, name_lower: str) -> str:
    for c in df.columns:
        if c.lower() == name_lower:
            return c
    raise KeyError(f"Fant ikke kolonne '{name_lower}' i {list(df.columns)}")


# ----- Backbone: postnummer × år -----

def load_backbone() -> pd.DataFrame:
    """Bygger backbone som postnummer × år (kryss-produkt). 23 år × 3378 pnr."""
    mapping_path = STD_DIR / "postnummer_kommune_mapping.parquet"
    if not mapping_path.exists():
        raise FileNotFoundError("Kjør standardize_kartverket.py før denne")
    mapping = pd.read_parquet(mapping_path)[["postnummer", "kommune_nr"]]

    aar = pd.DataFrame({"aar": list(range(2002, 2025))})
    backbone = mapping.merge(aar, how="cross")
    return backbone


# ----- Boliger per type (06265) -----

def standardize_boliger_per_type() -> pd.DataFrame:
    """Totaltall, modal og andeler per BygnType per (kommune, år) fra 06265."""
    src = RAW_DIR / "boliger_per_type_06265.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "bygntype")
    tid_col = _find_col(df, "tid")

    # Hovedtypene 01-05; vi dropper 999=annet før vi regner andeler
    main_types = ["01", "02", "03", "04", "05"]
    main = df[df[type_col].isin(main_types)].copy()

    wide = main.pivot_table(
        index=[region_col, tid_col], columns=type_col,
        values="value", aggfunc="sum",
    ).fillna(0)
    for t in main_types:
        if t not in wide.columns:
            wide[t] = 0.0
    wide = wide[main_types]

    totals = wide.sum(axis=1).rename("antall_boliger")
    andeler = wide.div(totals.replace(0, np.nan), axis=0)
    andeler.columns = [f"andel_bygntype_{t}_kommune" for t in main_types]
    modal = wide.idxmax(axis=1).rename("modal_boligtype_kommune")

    out = pd.concat([totals, modal, andeler], axis=1).reset_index()
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Folkemengde (06913) -----

def standardize_folkemengde() -> pd.DataFrame:
    """Folkemengde per kommune × år fra 06913 (ContentsCode=Folkemengde)."""
    src = RAW_DIR / "folkemengde_06913.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar", "value": "befolkning"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    return df[["kommune_nr", "aar", "befolkning"]]


# ----- Inntekt (12558) -----

def standardize_inntekt() -> pd.DataFrame:
    """Median inntekt etter skatt per kommune × år (12558, Desil 5)."""
    src = RAW_DIR / "inntekt_12558.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar",
                            "value": "inntekt_etter_skatt"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    df = df[["kommune_nr", "aar", "inntekt_etter_skatt"]].dropna()
    return df.drop_duplicates(["kommune_nr", "aar"])


# ----- Priser (06035) -----

# Mapping fra Boligtype-kode til kolonne-suffiks
_PRIS_TYPE_LABEL = {"01": "enebolig", "02": "smaahus", "03": "blokk"}


def standardize_priser() -> pd.DataFrame:
    """Kvm-pris per boligtype + vektet snitt og antall omsetninger per (kommune, år)."""
    src = RAW_DIR / "priser_06035.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "boligtype")
    content_col = _find_col(df, "contentscode")
    tid_col = _find_col(df, "tid")

    # Vi vil ha to kolonner (kvpris, omsetninger) per (Region, Boligtype, Tid).
    # pivot_table med aggfunc="sum" konverterer NaN til 0 — gir falske nuller.
    # Bruker derfor merge på filtrerte deler for å bevare NaN korrekt.
    kvpris_df = df[df[content_col] == "KvPris"][
        [region_col, type_col, tid_col, "value"]
    ].rename(columns={"value": "kvpris"})
    oms_df = df[df[content_col] == "Omsetninger"][
        [region_col, type_col, tid_col, "value"]
    ].rename(columns={"value": "omsetninger"})
    pivoted = kvpris_df.merge(oms_df, on=[region_col, type_col, tid_col], how="outer")

    # SSB rapporterer omsetninger=0 og kvpris=NaN for sensurerte små kommuner.
    # Vi setter kvpris=NaN også der omsetninger=0 for konsistens.
    pivoted.loc[pivoted["omsetninger"].fillna(0) == 0, "kvpris"] = np.nan

    # Totalt antall omsetninger per (kommune, år) — uavhengig av om vi har pris
    oms_total = pivoted.groupby([region_col, tid_col])["omsetninger"].sum().reset_index()
    oms_total = oms_total.rename(columns={"omsetninger": "antall_omsetninger_kommune"})

    # Vektet snitt over typene. Vi bruker bare rader der både kvpris og
    # omsetninger finnes — ellers risikerer vi 0 i teller mens nevner er
    # positiv (= falsk 0-pris for kommuner som har sensurert pris).
    valid = pivoted.dropna(subset=["kvpris"]).copy()
    valid["pris_x_vekt"] = valid["kvpris"] * valid["omsetninger"]
    alle = valid.groupby([region_col, tid_col]).agg(
        pris_x_vekt_sum=("pris_x_vekt", "sum"),
        omsetninger_med_pris=("omsetninger", "sum"),
    ).reset_index()
    alle["pris_kvm_alle_kommune"] = (
        alle["pris_x_vekt_sum"] / alle["omsetninger_med_pris"].replace(0, np.nan)
    )
    alle = alle.merge(oms_total, on=[region_col, tid_col], how="outer")
    alle = alle.rename(columns={region_col: "kommune_nr", tid_col: "aar"})[
        ["kommune_nr", "aar", "pris_kvm_alle_kommune", "antall_omsetninger_kommune"]
    ]
    alle["kommune_nr"] = alle["kommune_nr"].astype(str).str.zfill(4)
    alle["aar"] = alle["aar"].astype(int)

    # Per-type kvm-pris
    per_type = pivoted.pivot_table(
        index=[region_col, tid_col], columns=type_col,
        values="kvpris", aggfunc="first",
    )
    per_type = per_type.rename(
        columns={k: f"pris_kvm_{v}_kommune" for k, v in _PRIS_TYPE_LABEL.items()}
    )
    for label in _PRIS_TYPE_LABEL.values():
        col = f"pris_kvm_{label}_kommune"
        if col not in per_type.columns:
            per_type[col] = np.nan
    per_type = per_type.reset_index().rename(
        columns={region_col: "kommune_nr", tid_col: "aar"}
    )
    per_type["kommune_nr"] = per_type["kommune_nr"].astype(str).str.zfill(4)
    per_type["aar"] = per_type["aar"].astype(int)

    return per_type.merge(alle, on=["kommune_nr", "aar"], how="outer")


# ----- Byggeår og bruksareal-fordeling -----

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
}
_BYGGEAAR_PERIODER = ["for1946", "1946_1970", "1971_1990", "1991_2010", "etter2010"]

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
    """Median fra (midtpunkt, vekt)-serie sortert etter midtpunkt."""
    total = vekter.sum()
    if total == 0:
        return np.nan
    kumulativ = np.cumsum(vekter)
    idx = min(int(np.searchsorted(kumulativ, total / 2)), len(midtpunkter) - 1)
    return float(midtpunkter[idx])


def _andeler_og_median_tidsserie(
    df: pd.DataFrame, region_col: str, tid_col: str, bucket_col: str,
    buckets: dict, gruppe_navn: list[str], prefix: str, median_col: str,
) -> pd.DataFrame:
    """Beregn andeler + vektet median per (kommune, år)."""
    work = df[df[bucket_col].isin(buckets)].copy()
    work["gruppe"] = work[bucket_col].map(lambda c: buckets[c][0])
    work["midtpunkt"] = work[bucket_col].map(lambda c: buckets[c][1])

    grupped = work.groupby([region_col, tid_col, "gruppe"])["value"].sum().unstack(fill_value=0)
    for g in gruppe_navn:
        if g not in grupped.columns:
            grupped[g] = 0.0
    grupped = grupped[gruppe_navn]
    totals = grupped.sum(axis=1)
    andeler = grupped.div(totals.replace(0, np.nan), axis=0)
    andeler.columns = [f"{prefix}_{g}_kommune" for g in gruppe_navn]

    work_sorted = work.sort_values([region_col, tid_col, "midtpunkt"])
    medians = work_sorted.groupby([region_col, tid_col]).apply(
        lambda g: _vektet_median(g["midtpunkt"].values, g["value"].values),
        include_groups=False,
    ).rename(median_col)

    return pd.concat([andeler, medians], axis=1).reset_index()


def standardize_byggeaar() -> pd.DataFrame:
    """Byggeår-fordeling i fem perioder + median per (kommune, år)."""
    df = parse_jsonstat_chunks("byggeaar_06266")
    if df.empty:
        return pd.DataFrame()

    region_col = _find_col(df, "region")
    bucket_col = _find_col(df, "bygnar")
    tid_col = _find_col(df, "tid")

    out = _andeler_og_median_tidsserie(
        df, region_col, tid_col, bucket_col,
        _BYGGEAAR_BUCKETS, _BYGGEAAR_PERIODER,
        prefix="andel_byggeaar", median_col="median_byggeaar_kommune",
    )
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


def standardize_bruksareal() -> pd.DataFrame:
    """Bruksareal-fordeling i fem grupper + median per (kommune, år)."""
    df = parse_jsonstat_chunks("bruksareal_06513")
    if df.empty:
        return pd.DataFrame()

    region_col = _find_col(df, "region")
    bucket_col = _find_col(df, "bruksareal")
    tid_col = _find_col(df, "tid")

    out = _andeler_og_median_tidsserie(
        df, region_col, tid_col, bucket_col,
        _BRUKSAREAL_BUCKETS, _BRUKSAREAL_GRUPPER,
        prefix="andel_areal", median_col="median_bruksareal_kommune",
    )
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Utdanning (09429) -----

# Mapping fra Nivaa-kode til kolonne-suffiks. "00" (i alt) brukes som nevner.
_UTD_NIVAA_LABEL = {
    "01": "grunnskole",
    "02a": "videregaaende",
    "11": "fagskole",
    "03a": "uh_kort",
    "04a": "uh_lang",
    "09": "uoppgitt",
}


def standardize_utdanning() -> pd.DataFrame:
    """Andel per utdanningsnivå per (kommune, år) fra 09429."""
    src = RAW_DIR / "utdanning_09429.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    nivaa_col = _find_col(df, "nivaa")
    tid_col = _find_col(df, "tid")

    # ContentsCode er allerede filtrert til PersonerProsent — value er prosent
    df["andel"] = df["value"] / 100.0  # 0-1 i stedet for 0-100

    main = df[df[nivaa_col].isin(_UTD_NIVAA_LABEL)].copy()
    wide = main.pivot_table(
        index=[region_col, tid_col], columns=nivaa_col,
        values="andel", aggfunc="first",
    )
    wide = wide.rename(columns={k: f"andel_utdanning_{v}_kommune"
                                 for k, v in _UTD_NIVAA_LABEL.items()})
    for v in _UTD_NIVAA_LABEL.values():
        col = f"andel_utdanning_{v}_kommune"
        if col not in wide.columns:
            wide[col] = np.nan

    out = wide.reset_index().rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Sysselsetting (07984) -----

def standardize_sysselsetting() -> pd.DataFrame:
    """Antall sysselsatte (15-74 år, alle næringer) per (kommune, år)."""
    src = RAW_DIR / "sysselsetting_07984.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar",
                            "value": "antall_sysselsatte"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    return df[["kommune_nr", "aar", "antall_sysselsatte"]].drop_duplicates(
        ["kommune_nr", "aar"]
    )


# ----- Main -----

def main() -> None:
    print("=== Standardisering: SSB (tidsserie 2002-2024) ===")
    backbone = load_backbone()
    print(f"  Backbone: {len(backbone):,} rader (postnummer × år)")

    parts = {
        "boliger_per_type": standardize_boliger_per_type(),
        "folkemengde": standardize_folkemengde(),
        "inntekt": standardize_inntekt(),
        "priser": standardize_priser(),
        "byggeaar": standardize_byggeaar(),
        "bruksareal": standardize_bruksareal(),
        "utdanning": standardize_utdanning(),
        "sysselsetting": standardize_sysselsetting(),
    }
    for name, p in parts.items():
        print(f"  {name}: {len(p):,} kommune-år rader, {len(p.columns)} kol")

    # Join alle på (kommune_nr, aar). Hver del er allerede på riktig nivå.
    df = backbone.copy()
    for name, p in parts.items():
        if p.empty:
            continue
        df = df.merge(p, on=["kommune_nr", "aar"], how="left")

    out = STD_DIR / "ssb_bolig_demografi.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  ssb_bolig_demografi.parquet: {len(df):,} rader, {len(df.columns)} kolonner")
    # Vis dekning på de viktigste kolonnene
    for col in [
        "antall_boliger", "befolkning", "inntekt_etter_skatt",
        "pris_kvm_alle_kommune", "median_byggeaar_kommune",
        "median_bruksareal_kommune", "andel_utdanning_uh_kort_kommune",
        "antall_sysselsatte",
    ]:
        if col in df.columns:
            print(f"    {col}: {df[col].notna().mean():.1%}")
    print("=== SSB standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
