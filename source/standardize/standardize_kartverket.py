"""
Konverterer rådata fra Kartverket og Bring til Parquet med postnummer som nøkkel.

To outputfiler:
  postnummer_geometri.parquet         postnummer + polygon (EPSG:4326)
  postnummer_kommune_mapping.parquet  postnummer → kommune + navn
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "kartverket"
OUT_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def standardize_postnummer_geometri() -> gpd.GeoDataFrame:
    src = RAW_DIR / "postnummer_wfs.gml"
    out = OUT_DIR / "postnummer_geometri.parquet"

    gdf = gpd.read_file(src)
    gdf = gdf.to_crs("EPSG:4326")
    gdf.columns = [c.lower() for c in gdf.columns]

    # WFS-versjoner gir litt ulike navn på postnummer-kolonnen
    pnr_col = next((c for c in gdf.columns if "postnummer" in c or "postnum" in c), None)
    if pnr_col is None:
        raise ValueError(f"Fant ikke postnummer-kolonne i {list(gdf.columns)}")

    gdf = gdf.rename(columns={pnr_col: "postnummer"})
    gdf["postnummer"] = gdf["postnummer"].astype(str).str.zfill(4)
    gdf = gdf[["postnummer", "geometry"]].drop_duplicates("postnummer")

    gdf.to_parquet(out, index=False)
    print(f"  postnummer_geometri.parquet: {len(gdf)} postnummer")
    return gdf


def standardize_postnummer_kommune_mapping() -> pd.DataFrame:
    """Flat tabell: postnummer → kommunenummer + navn. Brukes av SSB og Eiendom Norge."""
    src = RAW_DIR / "postnummer_registry.json"
    out = OUT_DIR / "postnummer_kommune_mapping.parquet"

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["postnummer"] = df["postnummer"].astype(str).str.zfill(4)
    df["kommune_nr"] = df["kommunenummer"].astype(str).str.zfill(4)

    # Bring lagrer steds- og kommunenavn i UPPERCASE
    df["poststedsnavn"] = df["poststedsnavn"].astype(str).str.title()
    df["kommunenavn"] = df["kommunenavn"].astype(str).str.title()

    df = df[["postnummer", "poststedsnavn", "kommune_nr", "kommunenavn"]].drop_duplicates("postnummer")
    df.to_parquet(out, index=False)
    print(f"  postnummer_kommune_mapping.parquet: {len(df)} postnummer")
    return df


def main() -> None:
    print("=== Standardisering: Kartverket ===")
    standardize_postnummer_geometri()
    standardize_postnummer_kommune_mapping()
    print("=== Kartverket standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
