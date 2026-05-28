"""
Aggregerer Matrikkelen-Bygningspunkt fra bygningsnivå til kommune-nivå.

Kjøres bare hvis --include-matrikkelen er satt. Leser de paginerte GML-filene
fra raw_data/matrikkelen/fylke_*/page_*.gml og produserer to kolonner:
antall_bygninger_kommune (count) og modal_bygningstype_kommune (modus).
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "matrikkelen"
OUT_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GML-schemaet bruker varierende casing avhengig av versjon
_TYPE_COLS = ["bygningstype", "bygningsType", "bygningstypekode"]
_KOMMUNE_COLS = ["kommunenummer", "kommuneNummer"]


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_all_properties() -> pd.DataFrame:
    """Les alle paginerte GML-filer og slå sammen til én DataFrame uten geometri."""
    files = sorted(RAW_DIR.glob("fylke_*/page_*.gml"))
    if not files:
        # Bakoverkompatibel sti for gamle .geojson-filer
        files = sorted(RAW_DIR.glob("bygninger_fylke_*.geojson"))
        if not files:
            raise FileNotFoundError(
                f"Ingen matrikkelfiler i {RAW_DIR} — kjør collect_matrikkelen.py først"
            )

    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            gdf = gpd.read_file(fp)
        except Exception as e:
            print(f"    ADVARSEL: kunne ikke lese {fp.name}: {e}")
            continue

        type_col = _find_col(list(gdf.columns), _TYPE_COLS)
        kom_col = _find_col(list(gdf.columns), _KOMMUNE_COLS)
        if kom_col is None:
            continue

        keep_cols = [c for c in [type_col, kom_col] if c is not None]
        frames.append(pd.DataFrame(gdf[keep_cols]))

    if not frames:
        raise RuntimeError("Ingen lesbare GML-filer hadde forventede attributter")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Lastet {len(df):,} bygninger fra {len(files)} GML-filer")
    return df


def aggregate_per_kommune(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    type_col = _find_col(cols, _TYPE_COLS)
    kom_col = _find_col(cols, _KOMMUNE_COLS)
    if kom_col is None:
        raise ValueError(f"Fant ikke kommunenummer-kolonne. Tilgjengelige: {cols[:20]}")

    df = df.copy()
    df["kommune_nr"] = df[kom_col].astype(str).str.zfill(4)
    df["bygningstype"] = df[type_col].astype(str) if type_col else "ukjent"

    def modal(s: pd.Series) -> str:
        counts = s.value_counts()
        return str(counts.index[0]) if len(counts) > 0 else "ukjent"

    return df.groupby("kommune_nr").agg(
        antall_bygninger_kommune=("bygningstype", "count"),
        modal_bygningstype_kommune=("bygningstype", modal),
    ).reset_index()


def main() -> None:
    print("=== Standardisering: Matrikkelen ===")
    raw = load_all_properties()
    aggregated = aggregate_per_kommune(raw)

    out = OUT_DIR / "matrikkelen_aggregert.parquet"
    aggregated.to_parquet(out, index=False)
    print(f"  matrikkelen_aggregert.parquet: {len(aggregated)} kommuner")
    print("=== Matrikkelen standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
