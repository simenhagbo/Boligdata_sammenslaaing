"""
Parser de tre SSB-tabellene fra JSON-stat til Parquet med postnummer-nøkkel.

SSB er på kommunenivå — alle postnummer i samme kommune får samme verdi.
JSON-stat-formatet lagrer verdier som en flat liste der rekkefølgen tilsvarer
det kartesiske produktet av dimensjonene. itertools.product gjenoppretter
indeksene.
"""

import itertools
import json
from pathlib import Path

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


def standardize_boliger_per_type(mapping: pd.DataFrame) -> pd.DataFrame:
    """Antall boliger og modal boligtype per kommune (tabell 06265)."""
    src = RAW_DIR / "boliger_per_type_06265.json"
    if not src.exists():
        return mapping.assign(
            antall_boliger=pd.NA, modal_boligtype_kommune=pd.NA
        )[["postnummer", "antall_boliger", "modal_boligtype_kommune"]]

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "bygntype")

    # Total: sum over alle BygnType per region
    totals = df.groupby(region_col)["value"].sum().reset_index()
    totals = totals.rename(columns={region_col: "kommune_nr", "value": "antall_boliger"})
    totals["kommune_nr"] = totals["kommune_nr"].astype(str).str.zfill(4)

    # Modal: typen med høyest antall per region. Sorterer synkende og tar
    # første rad per kommune.
    typed = df[[region_col, type_col, "value"]].sort_values(
        [region_col, "value"], ascending=[True, False]
    )
    modal = typed.drop_duplicates(region_col, keep="first").rename(
        columns={region_col: "kommune_nr", type_col: "modal_boligtype_kommune"}
    )
    modal["kommune_nr"] = modal["kommune_nr"].astype(str).str.zfill(4)
    modal = modal[["kommune_nr", "modal_boligtype_kommune"]]

    combined = totals.merge(modal, on="kommune_nr", how="left")
    merged = mapping.merge(combined, on="kommune_nr", how="left")
    return merged[["postnummer", "antall_boliger", "modal_boligtype_kommune"]]


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


def main() -> None:
    print("=== Standardisering: SSB ===")
    mapping = load_kommune_mapping()

    boliger_df = standardize_boliger_per_type(mapping)
    folkemengde_df = standardize_folkemengde(mapping)
    inntekt_df = standardize_inntekt(mapping)

    df = mapping.copy()
    df = df.merge(boliger_df, on="postnummer", how="left")
    df = df.merge(folkemengde_df, on="postnummer", how="left")
    df = df.merge(inntekt_df, on="postnummer", how="left")

    out = OUT_DIR / "ssb_bolig_demografi.parquet"
    df.to_parquet(out, index=False)
    print(f"  ssb_bolig_demografi.parquet: {len(df)} rader")
    print(f"    antall_boliger:           {df['antall_boliger'].notna().mean():.1%}")
    print(f"    modal_boligtype_kommune:  {df['modal_boligtype_kommune'].notna().mean():.1%}")
    print(f"    befolkning:               {df['befolkning'].notna().mean():.1%}")
    print(f"    inntekt_etter_skatt:      {df['inntekt_etter_skatt'].notna().mean():.1%}")
    print("=== SSB standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
