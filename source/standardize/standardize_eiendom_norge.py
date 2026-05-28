"""
Plukker ut siste måneds pris per region fra Eiendom Norge-Excelen og kobler
til postnummer via kommunenavn.

Dette er den mest skjøre delen av pipelinen — Eiendom Norge bruker bynavn
("Oslo", "Bergen") i stedet for kommunenummer, og kolonnenavnene i Excel-filen
endrer seg mellom utgaver. Søker derfor dynamisk etter relevante kolonner og
matcher på normaliserte navn. Loggen viser hvor mange postnummer som fikk
treff — er det lavt, gå inn i Excel-filen og se hva som har endret seg.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "eiendom_norge"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
OUT_DIR = STD_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_excel() -> pd.DataFrame:
    src = RAW_DIR / "prisstatistikk.xlsx"
    if not src.exists():
        raise FileNotFoundError(f"Mangler {src} — kjør collect_eiendom_norge.py først")
    try:
        return pd.read_excel(src, sheet_name="Månedlig", header=0)
    except Exception:
        # "Månedlig"-arket har ikke alltid det navnet — fall tilbake til det første
        return pd.read_excel(src, sheet_name=0, header=0)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Gjør kolonnenavn til lowercase snake_case uten norske tegn."""
    df.columns = (
        df.columns.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[æ]", "ae", regex=True)
        .str.replace(r"[ø]", "oe", regex=True)
        .str.replace(r"[å]", "aa", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def extract_latest_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Trekk ut siste måneds median kvm-pris per region."""
    df = normalize_columns(df)

    pris_col = next(
        (c for c in df.columns if "pris" in c and ("m2" in c or "kvm" in c or "kvad" in c)),
        None,
    )
    if pris_col is None:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if not numeric_cols:
            raise ValueError(f"Fant ingen priskolonne. Tilgjengelige: {list(df.columns)}")
        pris_col = numeric_cols[0]

    region_col = next(
        (c for c in df.columns if c in ("region", "omraade", "by", "omrade")),
        df.columns[0],
    )
    dato_col = next(
        (c for c in df.columns if "dato" in c or "mnd" in c or "maaned" in c or "aar" in c),
        None,
    )

    result = df[[region_col, pris_col]].copy()
    if dato_col:
        result[dato_col] = df[dato_col]
        result = result.sort_values(dato_col, ascending=False)

    result = result.rename(columns={region_col: "region_navn", pris_col: "median_pris_m2"})
    result = result.drop_duplicates("region_navn", keep="first")
    result["median_pris_m2"] = pd.to_numeric(result["median_pris_m2"], errors="coerce")
    return result[["region_navn", "median_pris_m2"]].dropna()


def koble_til_postnummer(priser: pd.DataFrame) -> pd.DataFrame:
    """Match Eiendom Norge-regioner til kommunenavn fra Kartverket-mappingen."""
    mapping_path = STD_DIR / "postnummer_kommune_mapping.parquet"
    if not mapping_path.exists():
        raise FileNotFoundError("Kjør standardize_kartverket.py før denne")
    mapping = pd.read_parquet(mapping_path)

    def norm(s: pd.Series) -> pd.Series:
        return s.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

    mapping["_kommunenavn_norm"] = norm(mapping["kommunenavn"])
    priser["_region_norm"] = norm(priser["region_navn"])

    merged = mapping.merge(
        priser.rename(columns={"_region_norm": "_kommunenavn_norm"}),
        on="_kommunenavn_norm",
        how="left",
    )

    matched = merged["median_pris_m2"].notna().mean()
    print(f"  Eiendom Norge kobling: {matched:.1%} av postnummer fikk pris")

    merged = merged.drop(columns=["_kommunenavn_norm", "region_navn"], errors="ignore")
    return merged[["postnummer", "kommune_nr", "median_pris_m2"]]


def main() -> None:
    print("=== Standardisering: Eiendom Norge ===")
    if not (RAW_DIR / "prisstatistikk.xlsx").exists():
        print("  Hopper over: prisstatistikk.xlsx ikke funnet")
        print("=== Eiendom Norge standardisering ferdig ===\n")
        return

    raw_df = load_excel()
    priser = extract_latest_prices(raw_df)
    result = koble_til_postnummer(priser)

    out = OUT_DIR / "eiendom_norge_priser.parquet"
    result.to_parquet(out, index=False)
    print(f"  eiendom_norge_priser.parquet: {len(result)} rader")
    print("=== Eiendom Norge standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
