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
    """Les Kartverket-GML, konverter til EPSG:4326 og lagre som Parquet.

    EPSG:4326 (WGS84, lat/lon) er det vanligste CRS-et for geo-data og
    matcher det de fleste andre verktøy forventer. Rådata fra WFS er i
    EPSG:25833 (UTM 33N).
    """
    src = RAW_DIR / "postnummer_wfs.gml"
    out = OUT_DIR / "postnummer_geometri.parquet"

    gdf = gpd.read_file(src)
    gdf = gdf.to_crs("EPSG:4326")
    # Lowercase alt for konsistens — GML kan returnere blandet casing
    gdf.columns = [c.lower() for c in gdf.columns]

    # Feltnavnet for postnummer varierer mellom WFS-versjoner og GML-schema.
    # Søk dynamisk etter en kolonne som inneholder "postnummer" eller "postnum"
    # i stedet for å hardkode et navn som kanskje endrer seg.
    pnr_col = next((c for c in gdf.columns if "postnummer" in c or "postnum" in c), None)
    if pnr_col is None:
        raise ValueError(f"Fant ikke postnummer-kolonne i {list(gdf.columns)}")

    gdf = gdf.rename(columns={pnr_col: "postnummer"})
    # Postnummer skal alltid være 4 sifre — zero-pad strenger som "1" → "0001"
    gdf["postnummer"] = gdf["postnummer"].astype(str).str.zfill(4)
    # Behold kun det vi trenger; drop_duplicates som sikkerhetssjekk
    gdf = gdf[["postnummer", "geometry"]].drop_duplicates("postnummer")

    gdf.to_parquet(out, index=False)
    print(f"  postnummer_geometri.parquet: {len(gdf)} postnummer")
    return gdf


def standardize_postnummer_kommune_mapping() -> pd.DataFrame:
    """Flat tabell: postnummer → kommunenummer + navn. Brukes av SSB-joinen."""
    src = RAW_DIR / "postnummer_registry.json"
    out = OUT_DIR / "postnummer_kommune_mapping.parquet"

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Zero-pad både postnummer og kommunenummer til 4 siffer.
    # Eksempel: kommunenummer "301" (Oslo) skal være "0301" for konsistent join.
    df["postnummer"] = df["postnummer"].astype(str).str.zfill(4)
    df["kommune_nr"] = df["kommunenummer"].astype(str).str.zfill(4)

    # Bring lagrer alt i UPPERCASE ("OSLO", "BERGEN"). Title-case er bedre
    # for ML-features og kart-visualisering ("Oslo", "Bergen").
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
